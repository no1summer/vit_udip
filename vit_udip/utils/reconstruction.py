"""
Reconstruction utilities for ViT-UDIP.

This module provides functions for reconstructing images using trained models,
including saving reconstructed volumes and slices.
"""

import os
import numpy as np
import nibabel as nib
import torch
import imageio
from torchvision.utils import save_image
from typing import Optional, Union


def minmax_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] range using min-max scaling.
    
    Args:
        volume: Input volume array
        
    Returns:
        Normalized volume array
    """
    v = volume.astype(np.float32)
    mn, mx = v.min(), v.max()
    if mx - mn < 1e-8:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


def save_slices(vol: np.ndarray, outdir: str, prefix: str) -> None:
    """Save individual slices of a volume as PNG images.
    
    Args:
        vol: Volume array of shape (D, H, W)
        outdir: Output directory for slice images
        prefix: Prefix for slice filenames
    """
    os.makedirs(outdir, exist_ok=True)
    v_norm = minmax_normalize(vol)
    depth = v_norm.shape[0]
    
    for z in range(depth):
        slice_img = (v_norm[z] * 255).astype(np.uint8)
        imageio.imwrite(
            os.path.join(outdir, f"{prefix}_slice_{z:03d}.png"), 
            slice_img
        )


def reconstruct_image(
    model: torch.nn.Module,
    image_path: str,
    output_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    save_slices_flag: bool = False,
    save_center_slice: bool = True
) -> np.ndarray:
    """Reconstruct an image using a trained ViT-UDIP model.
    
    Args:
        model: Trained UDIPViT_engine model
        image_path: Path to input NIfTI image
        output_path: Path to save reconstructed NIfTI (optional)
        device: Device to run inference on (default: auto-detect)
        save_slices_flag: Whether to save individual slices as PNG
        save_center_slice: Whether to save center slice as PNG
        
    Returns:
        Reconstructed volume as numpy array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess image
    img_nib = nib.load(image_path)
    img_data = img_nib.get_fdata().astype(np.float32)
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_data).unsqueeze(0)  # (1, D, H, W)
    img_tensor = img_tensor.to(device)
    
    # Create mask (non-zero voxels)
    mask = (img_tensor != 0).float()
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        compute_pool, loss, pred_patches, batch_mask = model(img_tensor, mask)
        
        # Unpatchify reconstruction
        m = model.module if hasattr(model, 'module') else model
        recon_vols = m.decoder.unpatchify(pred_patches, batch_mask, img_tensor)
    
    # Convert back to numpy
    reconstructed = recon_vols[0].detach().cpu().float().numpy()  # (D, H, W)
    
    # Save reconstructed volume if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        affine = img_nib.affine
        nib.save(nib.Nifti1Image(reconstructed, affine), output_path)
        print(f"Saved reconstructed volume to: {output_path}")
    
    # Save center slice if requested
    if save_center_slice and output_path:
        center_idx = reconstructed.shape[0] // 2
        center_slice = minmax_normalize(reconstructed[center_idx])
        center_slice_tensor = torch.from_numpy(center_slice).unsqueeze(0)
        
        center_slice_path = output_path.replace('.nii.gz', '_center_slice.png')
        save_image(center_slice_tensor, center_slice_path)
        print(f"Saved center slice to: {center_slice_path}")
    
    # Save individual slices if requested
    if save_slices_flag and output_path:
        slices_dir = output_path.replace('.nii.gz', '_slices')
        save_slices(reconstructed, slices_dir, 'recon')
        print(f"Saved individual slices to: {slices_dir}")
    
    return reconstructed


def reconstruct_batch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_dir: str,
    num_samples: int = 5,
    device: Optional[Union[str, torch.device]] = None,
    save_slices_flag: bool = False
) -> None:
    """Reconstruct multiple images from a dataloader.
    
    Args:
        model: Trained UDIPViT_engine model
        dataloader: DataLoader containing images to reconstruct
        output_dir: Directory to save reconstructions
        num_samples: Number of samples to process
        device: Device to run inference on
        save_slices_flag: Whether to save individual slices
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    sample_count = 0
    for batch_idx, (img, mask) in enumerate(dataloader):
        if sample_count >= num_samples:
            break
            
        img = img.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            compute_pool, loss, pred_patches, batch_mask = model(img, mask)
            m = model.module if hasattr(model, 'module') else model
            recon_vols = m.decoder.unpatchify(pred_patches, batch_mask, img)
        
        # Process each sample in the batch
        for i in range(img.shape[0]):
            if sample_count >= num_samples:
                break
                
            reconstructed = recon_vols[i].detach().cpu().float().numpy()
            original = img[i].detach().cpu().float().numpy()
            
            # Save reconstructed volume
            recon_path = os.path.join(output_dir, f'recon_sample_{sample_count}.nii.gz')
            orig_path = os.path.join(output_dir, f'orig_sample_{sample_count}.nii.gz')
            
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(reconstructed, affine), recon_path)
            nib.save(nib.Nifti1Image(original, affine), orig_path)
            
            # Save center slices
            center_idx = reconstructed.shape[0] // 2
            recon_center = minmax_normalize(reconstructed[center_idx])
            orig_center = minmax_normalize(original[center_idx])
            
            save_image(
                torch.from_numpy(recon_center).unsqueeze(0),
                os.path.join(output_dir, f'recon_center_slice_{sample_count}.png')
            )
            save_image(
                torch.from_numpy(orig_center).unsqueeze(0),
                os.path.join(output_dir, f'orig_center_slice_{sample_count}.png')
            )
            
            # Save individual slices if requested
            if save_slices_flag:
                recon_slices_dir = os.path.join(output_dir, f'recon_slices_{sample_count}')
                orig_slices_dir = os.path.join(output_dir, f'orig_slices_{sample_count}')
                save_slices(reconstructed, recon_slices_dir, 'recon')
                save_slices(original, orig_slices_dir, 'orig')
            
            sample_count += 1
            print(f"Processed sample {sample_count}/{num_samples}")
    
    print(f"Saved {sample_count} reconstructions to {output_dir}")
