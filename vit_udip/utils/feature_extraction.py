"""
Utility functions for ViT-UDIP.

This module provides utility functions for feature extraction
and other common operations.
"""

import torch
import nibabel as nib
import numpy as np
from ..models import UDIPViT_engine


def extract_features(model, image_path, device=None):
    """Extract features from a single image using the trained model.
    
    Args:
        model (UDIPViT_engine): Trained model
        image_path (str): Path to the image file
        device (torch.device, optional): Device to run inference on
        
    Returns:
        torch.Tensor: Extracted features
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    img = nib.load(image_path)
    img = img.get_fdata()
    img = torch.from_numpy(img)
    img = torch.nn.functional.pad(img, (0,0,3,3,0,0))  # padding image from 182x218x182 to 182x224x182
    mask = img != 0
    img = (img - img[img != 0].mean()) / img[img != 0].std()
    img = img.type(torch.float)
    
    # Add batch dimension and move to device
    img = img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features, _, _, _ = model(img, mask)
    
    return features.squeeze(0)  # Remove batch dimension


def extract_features_batch(model, image_paths, device=None, batch_size=1):
    """Extract features from multiple images.
    
    Args:
        model (UDIPViT_engine): Trained model
        image_paths (list): List of image file paths
        device (torch.device, optional): Device to run inference on
        batch_size (int): Batch size for processing
        
    Returns:
        list: List of extracted features
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_masks = []
        
        # Load batch
        for path in batch_paths:
            img = nib.load(path)
            img = img.get_fdata()
            img = torch.from_numpy(img)
            img = torch.nn.functional.pad(img, (0,0,3,3,0,0))
            mask = img != 0
            img = (img - img[img != 0].mean()) / img[img != 0].std()
            img = img.type(torch.float)
            
            batch_images.append(img)
            batch_masks.append(mask)
        
        # Stack batch
        batch_images = torch.stack(batch_images).to(device)
        batch_masks = torch.stack(batch_masks).to(device)
        
        # Extract features
        with torch.no_grad():
            features, _, _, _ = model(batch_images, batch_masks)
        
        all_features.extend([f.squeeze(0) for f in features])
    
    return all_features
