"""
Example inference script for ViT-UDIP.

This script demonstrates how to use a trained ViT-UDIP model for inference
and feature extraction.
"""

import torch
import numpy as np
import os
from vit_udip.models import UDIPViT_engine
from vit_udip.utils.feature_extraction import extract_features, extract_features_batch


def main():
    """Main inference function."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model from checkpoint
    checkpoint_path = "path/to/best_model.pth"  # Update with your checkpoint path
    model = UDIPViT_engine.from_checkpoint(
        checkpoint_path,
        lr=0.001,
        encoder_embed_dim=128,
        decoder_embed_dim=64,
        encoder_depth=12,
        decoder_depth=12,
        num_heads=8
    )
    
    print("Model loaded successfully")
    
    # Extract features from a single image
    image_path = "path/to/image.nii.gz"  # Update with your image path
    features = extract_features(model, image_path, device)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Features mean: {features.mean().item():.4f}")
    print(f"Features std: {features.std().item():.4f}")
    
    # Save features
    torch.save(features, "extracted_features.pt")
    print("Features saved to: extracted_features.pt")
    
    # Example: Extract features from multiple images
    image_paths = [
        "path/to/image1.nii.gz",
        "path/to/image2.nii.gz",
        "path/to/image3.nii.gz"
    ]
    
    # Note: Update image_paths with actual paths
    if all(os.path.exists(path) for path in image_paths):
        all_features = extract_features_batch(model, image_paths, device, batch_size=2)
        print(f"Extracted features from {len(all_features)} images")
        
        # Stack features
        stacked_features = torch.stack(all_features)
        print(f"Stacked features shape: {stacked_features.shape}")
        
        # Save stacked features
        torch.save(stacked_features, "batch_features.pt")
        print("Batch features saved to: batch_features.pt")


if __name__ == "__main__":
    main()
