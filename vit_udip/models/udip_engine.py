"""
Main UDIP-ViT engine for medical image processing.

This module provides the main model wrapper that combines the encoder
and decoder components for unsupervised deep image processing.
"""

import torch
import torch.nn as nn
from functools import partial

from .vit_encoder import VisionTransformer
from .vit_decoder import Decoder


class UDIPViT_engine(nn.Module):
    """
    Wrapper for UDIP-ViT components, designed to be a drop-in replacement for engine_AE.
    It handles the single-modality case and matches the forward pass of udip_vit_merged.py.
    
    Args:
        lr (float): Learning rate
        patch_size (int): Patch size
        tubelet_size (int): Tubelet size for depth dimension
        img_size (int): Input image size
        num_frames (int): Number of frames (depth)
        in_chans (int): Number of input channels
        encoder_embed_dim (int): Encoder embedding dimension
        decoder_embed_dim (int): Decoder embedding dimension
        encoder_depth (int): Number of encoder transformer blocks
        decoder_depth (int): Number of decoder transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_scale (float, optional): Scale factor for QK
        drop_rate (float): Dropout probability
        attn_drop_rate (float): Attention dropout probability
        norm_layer (nn.Module): Normalization layer
        init_std (float): Initialization standard deviation
        non_zero_patch_opt (bool): Whether to use non-zero patch optimization
        concat_modalities (bool): Whether to concatenate modalities (unused)
        use_modality (str): Modality to use (unused)
        use_sincos_pos_embed (bool): Whether to use sinusoidal positional embedding (unused)
        use_patchwise_loss (bool): Whether to use patchwise loss (unused)
    """
    
    def __init__(self, lr, patch_size=14, tubelet_size=16,
                 img_size=182, num_frames=224, in_chans=1, 
                 encoder_embed_dim=128, decoder_embed_dim=64, 
                 encoder_depth=12, decoder_depth=12, num_heads=8, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm, init_std=0.02, non_zero_patch_opt=True,
                 # Unused args for compatibility
                 concat_modalities=False, use_modality='T1', use_sincos_pos_embed=True, 
                 use_patchwise_loss=True):
        super().__init__()
        self.lr = lr # for optimizer
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.image_size = img_size
        self.num_frames = num_frames
        
        self.grid_size = self.image_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        
        # Encoder
        self.encoder = VisionTransformer(
            img_size, patch_size, num_frames, tubelet_size, in_chans, encoder_embed_dim,
            encoder_depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
            attn_drop_rate, norm_layer, init_std, None, False, non_zero_patch_opt)
        
        # Decoder
        self.decoder = Decoder(
            img_size=img_size, patch_size=patch_size, num_frames=num_frames, tubelet_size=tubelet_size,
            embed_dim=encoder_embed_dim, decoder_embed_dim=decoder_embed_dim, 
            depth=decoder_depth, num_heads=num_heads)

    def forward_loss(self, pred, imgs, batch_mask, non_zero_mask):
        """Compute forward loss for reconstruction.
        
        Args:
            pred (torch.Tensor): Predicted patches
            imgs (torch.Tensor): Target images
            batch_mask (torch.Tensor): Batch mask for active patches
            non_zero_mask (torch.Tensor): Non-zero mask for patches
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        target = imgs.view(
            (-1, self.grid_size, self.patch_size, self.grid_depth, self.tubelet_size,
             self.grid_size, self.patch_size)).permute(0,1,3,5,2,4,6).reshape((-1, 13*14*13, 14*16*14))
        
        # Select only active patches
        target = target[:, batch_mask, :]
        mask = non_zero_mask[:, batch_mask]
        
        # MSE loss
        loss = ((target - pred)**2).mean(-1)
        return (loss * mask).sum() / mask.sum()
    
    def forward(self, imgs, img_mask):
        """Forward pass; expects x_T1 shape (B,H,D,W) like reference UDIP pipeline.
        
        Args:
            imgs (torch.Tensor): Input images
            img_mask (torch.Tensor): Mask from input image
            
        Returns:
            tuple: (pooled_features, loss, predictions, batch_mask)
        """
        # Encode (imgs kept in original ordering expected by patch_embed; y_for_mask used only for zero-patch masking)
        latent, batch_mask, non_zero_mask = self.encoder(imgs, img_mask)
        
        # Average pool latent representations
        compute_pool = latent.mean(1, keepdim=True)
        
        # Decode
        pred = self.decoder(compute_pool, batch_mask)
        
        # Compute loss
        loss = self.forward_loss(pred, imgs, batch_mask, non_zero_mask)
        
        return compute_pool, loss, pred, batch_mask

    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            **kwargs: Additional arguments for model initialization
            
        Returns:
            UDIPViT_engine: Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model parameters from checkpoint if available
        if 'model_state_dict' in checkpoint:
            # Try to infer parameters from the checkpoint
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'module.' prefix if present (from DDP)
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            # Create model with default parameters
            model = cls(lr=0.001, **kwargs)
            
            # Load state dict
            model.load_state_dict(state_dict)
            
            return model
        else:
            raise ValueError("Checkpoint does not contain model_state_dict")
