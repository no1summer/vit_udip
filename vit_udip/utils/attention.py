"""
Attention mechanisms for Vision Transformer.

This module provides multi-head self-attention and MLP components
used in the Vision Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-Layer Perceptron.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features
        out_features (int, optional): Number of output features
        act_layer (nn.Module): Activation layer
        drop (float): Dropout probability
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass through MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_scale (float, optional): Scale factor for QK
        attn_drop (float): Attention dropout probability
        proj_drop (float): Projection dropout probability
        use_sdpa (bool): Whether to use scaled dot product attention
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        # Try to use SDPA, but fall back if it's not available (e.g., older PyTorch)
        self.use_sdpa = use_sdpa and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mask=None):
        """Forward pass through attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: (output_tensor, attention_weights)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            # scaled_dot_product_attention does not return attention weights
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_scale (float, optional): Scale factor for QK
        drop (float): Dropout probability
        attn_drop (float): Attention dropout probability
        act_layer (nn.Module): Activation layer
        norm_layer (nn.Module): Normalization layer
        grid_size (int, optional): Grid size for positional encoding
        grid_depth (int, optional): Grid depth for positional encoding
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 grid_size=None, grid_depth=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        """Forward pass through transformer block.
        
        Args:
            x (torch.Tensor): Input tensor
            return_attention (bool): Whether to return attention weights
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor or tuple: Output tensor or (output, attention_weights)
        """
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
