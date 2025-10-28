"""
Training utilities for ViT-UDIP.

This module provides training functions, validation utilities,
and training loop implementations.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def validate_one_epoch(model, dataloader, device):
    """Validate model for one epoch.
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device to run validation on
        
    Returns:
        tuple: (avg_loss, avg_psnr, avg_ssim)
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples = 0
    
    val_pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in val_pbar:
            x_T1, mask = batch
            x_T1 = x_T1.to(device)
            mask = mask.to(device)

            # Model forward pass returns loss directly
            _, loss, pred_patches, batch_mask = model(x_T1, mask)
            
            # The loss is already the mean over the batch
            total_loss += loss.item()
            val_pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})

            # Unpatchify for metrics calculation
            m = model.module if hasattr(model, 'module') else model
            recon_T1 = m.decoder.unpatchify(pred_patches, batch_mask, x_T1)

            # Metrics per sample
            for i in range(x_T1.shape[0]):
                gt_T1 = x_T1[i].cpu().numpy()  # (H, D, W)
                pred_T1 = recon_T1[i].detach().cpu().numpy()  # (D, H, W)
                msk_3d = mask[i].cpu().numpy().astype(bool)  # (H, D, W)
                if msk_3d.sum() > 0:
                    gt_T1_masked = gt_T1[msk_3d]
                    pred_T1_masked = pred_T1[msk_3d]
                    dr_T1 = gt_T1_masked.max() - gt_T1_masked.min()
                    psnr_T1 = compare_psnr(gt_T1_masked, pred_T1_masked, data_range=dr_T1) if dr_T1 > 0 else 0.0
                    try:
                        dr_T1_full = gt_T1.max() - gt_T1.min()
                        ssim_T1 = compare_ssim(gt_T1, pred_T1, data_range=dr_T1_full, mask=msk_3d) if dr_T1_full > 0 else 0.0
                    except Exception:
                        ssim_T1 = 0.0
                    total_psnr += psnr_T1
                    total_ssim += ssim_T1
                n_samples += 1
                
    # Calculate average loss per sample processed by this GPU
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / n_samples if n_samples > 0 else 0.0
    avg_ssim = total_ssim / n_samples if n_samples > 0 else 0.0
    
    # Synchronize validation metrics across all GPUs
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        device = next(model.parameters()).device
        
        loss_tensor = torch.tensor(avg_loss, device=device)
        psnr_tensor = torch.tensor(avg_psnr, device=device)
        ssim_tensor = torch.tensor(avg_ssim, device=device)
        
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(psnr_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ssim_tensor, op=torch.distributed.ReduceOp.SUM)
        
        avg_loss = loss_tensor.item() / world_size
        avg_psnr = psnr_tensor.item() / world_size
        avg_ssim = ssim_tensor.item() / world_size
    
    return avg_loss, avg_psnr, avg_ssim


def train_model(model, train_dataset, val_dataset=None, num_epochs=300, 
                batch_size=4, num_workers=4, device=None, save_dir=None):
    """Train the ViT-UDIP model.
    
    Args:
        model (UDIPViT_engine): Model to train
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset, optional): Validation dataset
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        num_workers (int): Number of data loader workers
        device (torch.device, optional): Device to train on
        save_dir (str, optional): Directory to save checkpoints
        
    Returns:
        UDIPViT_engine: Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, 
        num_workers=num_workers, shuffle=True, drop_last=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False, drop_last=False
        )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for i, (x_T1, mask) in enumerate(pbar):
            x_T1 = x_T1.to(device)
            mask = mask.to(device)
            
            with torch.cuda.amp.autocast():
                _, loss, _, _ = model(x_T1, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        
        # Validation
        if val_loader is not None:
            avg_val_loss, avg_psnr, avg_ssim = validate_one_epoch(model, val_loader, device)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, Val PSNR: {avg_psnr:.4f}, Val SSIM: {avg_ssim:.4f}")
            
            # Save checkpoint
            if save_dir is not None:
                is_best = avg_val_loss < best_val_loss
                best_val_loss = min(avg_val_loss, best_val_loss)
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'avg_train_loss': avg_train_loss,
                    'avg_val_loss': avg_val_loss,
                    'avg_psnr': avg_psnr,
                    'avg_ssim': avg_ssim
                }
                
                # Save latest checkpoint
                torch.save(checkpoint, f"{save_dir}/latest_checkpoint.pth")
                
                # Save best model
                if is_best:
                    torch.save(checkpoint, f"{save_dir}/best_model.pth")
                    print(f"New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
    
    return model
