"""
Vision Transformer encoder for medical image processing.

This module provides the encoder component of the Vision Transformer
architecture, including patch embedding, positional encoding, and
transformer blocks.
"""

import math
import torch
import torch.nn as nn
from functools import partial

from ..utils.positional_encoding import get_3d_sincos_pos_embed, trunc_normal_
from ..utils.patch_embedding import PatchEmbed3D
from ..utils.attention import TransformerBlock


class VisionTransformer(nn.Module):
    """Vision Transformer encoder with optional non-zero patch optimization.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size
        num_frames (int): Number of frames (depth)
        tubelet_size (int): Tubelet size for depth dimension
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_scale (float, optional): Scale factor for QK
        drop_rate (float): Dropout probability
        attn_drop_rate (float): Attention dropout probability
        norm_layer (nn.Module): Normalization layer
        init_std (float): Initialization standard deviation
        out_layers (list, optional): Output layers
        uniform_power (bool): Whether to use uniform power distribution
        non_zero_patch_opt (bool): Whether to use non-zero patch optimization
    """
    
    def __init__(self, img_size=182, patch_size=14, num_frames=224, tubelet_size=16,
                 in_chans=1, embed_dim=128, depth=12, num_heads=8, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm, init_std=0.02, out_layers=None,
                 uniform_power=False, non_zero_patch_opt=True, **kwargs):
        super().__init__()
        self.non_zero_patch_opt = non_zero_patch_opt
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.grid_size = self.input_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim)
        
        self.num_patches = self.grid_size * self.grid_depth * self.grid_size

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                act_layer=nn.GELU, grid_size=self.grid_size, grid_depth=self.grid_depth,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        """Initialize positional embeddings with sine-cosine."""
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_depth, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        """Initialize module weights."""
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
        """Rescale transformer block weights for better initialization."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def non_zero_patch(self, x, img_mask):
        """Filter out patches that are all zeros across the batch.
        
        Args:
            x (torch.Tensor): Patch tokens
            img_mask (torch.Tensor): Mask from input image
            
        Returns:
            tuple: (filtered_tokens, batch_mask, non_zero_mask)
        """
        non_zero_mask = img_mask.view(
            -1, self.grid_size, self.patch_size, self.grid_depth, self.tubelet_size,
            self.grid_size, self.patch_size).sum((2,4,6)) != 0
        non_zero_mask = non_zero_mask.view(-1, self.grid_size * self.grid_depth * self.grid_size)
        batch_mask = non_zero_mask.max(0)[0]
        x = x[:, batch_mask, :]
        return x, batch_mask, non_zero_mask

    def forward(self, x, img_mask):
        """Forward pass through vision transformer.
        
        Args:
            x (torch.Tensor): Input image tensor
            img_mask (torch.Tensor): Mask from input image
            
        Returns:
            tuple: (encoded_features, batch_mask, non_zero_mask)
        """
        x = self.patch_embed(x)
        x += self.pos_embed
        if self.non_zero_patch_opt:
            x, batch_mask, non_zero_patch = self.non_zero_patch(x, img_mask)
        else:
            batch_mask = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
            non_zero_patch = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)
        return x, batch_mask, non_zero_patch
