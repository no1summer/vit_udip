"""
Vision Transformer decoder for medical image reconstruction.

This module provides the decoder component of the Vision Transformer
architecture, including positional token concatenation and patch reconstruction.
"""

import math
import torch
import torch.nn as nn
from functools import partial

from ..utils.positional_encoding import get_3d_sincos_pos_embed, trunc_normal_
from ..utils.attention import TransformerBlock


class Decoder(nn.Module):
    """UDIP-style decoder with positional token concatenation.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        num_frames (int): Number of frames (depth)
        tubelet_size (int): Tubelet size for depth dimension
        embed_dim (int): Encoder embedding dimension
        decoder_embed_dim (int): Decoder embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_scale (float, optional): Scale factor for QK
        drop_rate (float): Dropout probability
        attn_drop_rate (float): Attention dropout probability
        norm_layer (nn.Module): Normalization layer
        init_std (float): Initialization standard deviation
    """
    
    def __init__(self, img_size=182, patch_size=14, num_frames=224, tubelet_size=16,
                 embed_dim=128, decoder_embed_dim=64, depth=12, num_heads=8,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        self.num_patches = (self.input_size // self.patch_size) ** 2 * (self.num_frames // self.tubelet_size)
        self.grid_size = self.input_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        
        # Positional embedding for decoder
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.num_patches, self.decoder_embed_dim), requires_grad=False)
        self._init_pos_embed(self.pos_emb.data)
        
        # Decoder layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                act_layer=nn.GELU, grid_size=self.grid_size, grid_depth=self.grid_depth,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(self.decoder_embed_dim)
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        
        # Output projection
        patch_numel = self.patch_size ** 2 * self.tubelet_size
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, patch_numel, bias=True)
        
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """Rescale block weights."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_pos_embed(self, pos_embed):
        """Initialize positional embeddings."""
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_depth, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def unpatchify(self, x, batch_mask, imgs):
        """Reconstruct image from patches.
        
        Args:
            x (torch.Tensor): Patch predictions
            batch_mask (torch.Tensor): Batch mask for active patches
            imgs (torch.Tensor): Original images for reference
            
        Returns:
            torch.Tensor: Reconstructed images
        """
        B = x.shape[0]
        # Create a full tensor of zeros for all patches
        full_patches = torch.zeros(B, self.num_patches, x.shape[-1], device=x.device)
        full_patches.fill_(imgs[0,0,0,0])  # Fill with a constant value (e.g., zero)
        # Place the predicted patches into the correct positions
        full_patches[:, batch_mask, :] = x
        
        # Reshape to image dimensions
        x = full_patches.view(
            B, self.grid_size, self.grid_depth, self.grid_size, 
            self.patch_size, self.tubelet_size, self.patch_size
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.input_size, self.num_frames, self.input_size)
        return x

    def forward(self, x, batch_mask):
        """Forward pass through decoder.
        
        Args:
            x (torch.Tensor): Encoded features
            batch_mask (torch.Tensor): Batch mask for active patches
            
        Returns:
            torch.Tensor: Reconstructed patch predictions
        """
        # Get positional embeddings for active patches
        pos_emb = self.pos_emb[:, batch_mask, :].repeat(x.shape[0], 1, 1)
        num_latent = x.shape[1]
        
        # Project encoder output
        x = self.decoder_embed(x)
        
        # Concatenate memory tokens with positional tokens (UDIP strategy)
        x = torch.cat([x, pos_emb], dim=1)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        # Predict from positional tokens only
        x = self.decoder_pred(x[:, num_latent:, :])
        return x
