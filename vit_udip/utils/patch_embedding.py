"""
Patch embedding module for 3D medical images.

This module provides 3D patch embedding functionality for converting
volumetric medical images into patch tokens for Vision Transformer processing.
"""

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """3D Volume to Patch Embedding.
    
    Converts 3D medical images into patch tokens by applying 3D convolution
    with non-overlapping patches.
    
    Args:
        patch_size (int): Size of patches in height and width dimensions
        tubelet_size (int): Size of patches in depth dimension
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension for output tokens
    """
    
    def __init__(self, patch_size=14, tubelet_size=16, in_chans=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # Use (tubelet, patch, patch) ordering so temporal/depth dimension comes first
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size, tubelet_size, patch_size),
            stride=(patch_size, tubelet_size, patch_size)
        )

    def forward(self, x, **kwargs):
        """Forward pass through patch embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, D, W)
            
        Returns:
            torch.Tensor: Patch tokens of shape (B, num_patches, embed_dim)
        """
        # The UDIP model expects (B, C, H, D, W)
        # Our data loader provides (B, H, D, W), so we add the channel dim
        x = x.unsqueeze(1)  # Add channel dimension if needed
       
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, C, H, D, W) -> (B, C, H * D * W) -> (B, C, num_patches)
        return x
