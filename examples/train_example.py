"""
Example training script for ViT-UDIP.

This script demonstrates how to train the ViT-UDIP model on medical images.
"""

import os
import torch
from vit_udip.models import UDIPViT_engine
from vit_udip.data import MedicalImageDataset
from vit_udip.training import train_model


def main():
    """Main training function."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = UDIPViT_engine(
        lr=0.001,
        patch_size=14,
        tubelet_size=16,
        img_size=182,
        num_frames=224,
        in_chans=1,
        encoder_embed_dim=128,
        decoder_embed_dim=64,
        encoder_depth=12,
        decoder_depth=12,
        num_heads=8,
        non_zero_patch_opt=True,
        use_patchwise_loss=True,
    )
    
    print("Model created successfully")
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        datafile="path/to/train.csv",  # Update with your data path
        modality="T1_unbiased_linear"
    )
    
    val_dataset = MedicalImageDataset(
        datafile="path/to/val.csv",  # Update with your data path
        modality="T1_unbiased_linear"
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create save directory
    save_dir = "output/vit_udip_training"
    os.makedirs(save_dir, exist_ok=True)
    
    # Train model
    trained_model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=300,
        batch_size=4,
        num_workers=4,
        device=device,
        save_dir=save_dir
    )
    
    print("Training completed!")
    print(f"Model saved to: {save_dir}")


if __name__ == "__main__":
    main()
