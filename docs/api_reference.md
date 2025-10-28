"""
API Reference for ViT-UDIP.

This document provides detailed API documentation for all
classes and functions in the ViT-UDIP package.
"""

# ViT-UDIP API Reference

## Models

### UDIPViT_engine

Main model class for Vision Transformer unsupervised deep image processing.

```python
class UDIPViT_engine(nn.Module):
    def __init__(self, lr, patch_size=14, tubelet_size=16, img_size=182, 
                 num_frames=224, in_chans=1, encoder_embed_dim=128, 
                 decoder_embed_dim=64, encoder_depth=12, decoder_depth=12, 
                 num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, 
                 drop_rate=0.0, attn_drop_rate=0.0, norm_layer=nn.LayerNorm, 
                 init_std=0.02, non_zero_patch_opt=True, **kwargs):
```

**Parameters:**
- `lr` (float): Learning rate for optimizer
- `patch_size` (int): Size of patches in height and width dimensions
- `tubelet_size` (int): Size of patches in depth dimension
- `img_size` (int): Input image size
- `num_frames` (int): Number of frames (depth)
- `in_chans` (int): Number of input channels
- `encoder_embed_dim` (int): Encoder embedding dimension
- `decoder_embed_dim` (int): Decoder embedding dimension
- `encoder_depth` (int): Number of encoder transformer blocks
- `decoder_depth` (int): Number of decoder transformer blocks
- `num_heads` (int): Number of attention heads
- `mlp_ratio` (float): MLP expansion ratio
- `qkv_bias` (bool): Whether to use bias in QKV projection
- `qk_scale` (float, optional): Scale factor for QK
- `drop_rate` (float): Dropout probability
- `attn_drop_rate` (float): Attention dropout probability
- `norm_layer` (nn.Module): Normalization layer
- `init_std` (float): Initialization standard deviation
- `non_zero_patch_opt` (bool): Whether to use non-zero patch optimization

**Methods:**
- `forward(imgs, mask)`: Forward pass through the model
- `from_checkpoint(checkpoint_path, **kwargs)`: Load model from checkpoint

### VisionTransformer

Vision Transformer encoder component.

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=182, patch_size=14, num_frames=224, 
                 tubelet_size=16, in_chans=1, embed_dim=128, depth=12, 
                 num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, 
                 drop_rate=0.0, attn_drop_rate=0.0, norm_layer=nn.LayerNorm, 
                 init_std=0.02, out_layers=None, uniform_power=False, 
                 non_zero_patch_opt=True, **kwargs):
```

### Decoder

Vision Transformer decoder component.

```python
class Decoder(nn.Module):
    def __init__(self, img_size=182, patch_size=14, num_frames=224, 
                 tubelet_size=16, embed_dim=128, decoder_embed_dim=64, 
                 depth=12, num_heads=8, mlp_ratio=4.0, qkv_bias=True, 
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, 
                 norm_layer=nn.LayerNorm, init_std=0.02):
```

## Data

### MedicalImageDataset

Dataset class for medical images with padding and normalization.

```python
class MedicalImageDataset(Dataset):
    def __init__(self, datafile, modality):
```

**Parameters:**
- `datafile` (str): Path to CSV file containing image paths
- `modality` (str): Column name containing image paths

**Methods:**
- `__len__()`: Return dataset size
- `__getitem__(idx)`: Get item at index

## Training

### train_model

Train the ViT-UDIP model.

```python
def train_model(model, train_dataset, val_dataset=None, num_epochs=300, 
                batch_size=4, num_workers=4, device=None, save_dir=None):
```

**Parameters:**
- `model` (UDIPViT_engine): Model to train
- `train_dataset` (Dataset): Training dataset
- `val_dataset` (Dataset, optional): Validation dataset
- `num_epochs` (int): Number of training epochs
- `batch_size` (int): Batch size
- `num_workers` (int): Number of data loader workers
- `device` (torch.device, optional): Device to train on
- `save_dir` (str, optional): Directory to save checkpoints

**Returns:**
- `UDIPViT_engine`: Trained model

### validate_one_epoch

Validate model for one epoch.

```python
def validate_one_epoch(model, dataloader, device):
```

**Parameters:**
- `model` (nn.Module): Model to validate
- `dataloader` (DataLoader): Validation data loader
- `device` (torch.device): Device to run validation on

**Returns:**
- `tuple`: (avg_loss, avg_psnr, avg_ssim)

## Utils

### extract_features

Extract features from a single image.

```python
def extract_features(model, image_path, device=None):
```

**Parameters:**
- `model` (UDIPViT_engine): Trained model
- `image_path` (str): Path to the image file
- `device` (torch.device, optional): Device to run inference on

**Returns:**
- `torch.Tensor`: Extracted features

### extract_features_batch

Extract features from multiple images.

```python
def extract_features_batch(model, image_paths, device=None, batch_size=1):
```

**Parameters:**
- `model` (UDIPViT_engine): Trained model
- `image_paths` (list): List of image file paths
- `device` (torch.device, optional): Device to run inference on
- `batch_size` (int): Batch size for processing

**Returns:**
- `list`: List of extracted features

### Positional Encoding Functions

#### get_3d_sincos_pos_embed

Generate 3D sinusoidal positional embeddings.

```python
def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, 
                           uniform_power=False):
```

**Parameters:**
- `embed_dim` (int): Embedding dimension
- `grid_size` (int): Grid size for height and width
- `grid_depth` (int): Grid depth
- `cls_token` (bool): Whether to include CLS token
- `uniform_power` (bool): Whether to use uniform power distribution

**Returns:**
- `np.ndarray`: 3D positional embeddings

#### trunc_normal_

Truncated normal initialization.

```python
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
```

**Parameters:**
- `tensor` (torch.Tensor): Tensor to initialize
- `mean` (float): Mean of normal distribution
- `std` (float): Standard deviation of normal distribution
- `a` (float): Lower bound
- `b` (float): Upper bound

**Returns:**
- `torch.Tensor`: Initialized tensor

## Attention Components

### Attention

Multi-head self-attention module.

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., use_sdpa=True):
```

### TransformerBlock

Transformer block with self-attention and MLP.

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, grid_size=None, grid_depth=None):
```

### MLP

Multi-Layer Perceptron module.

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
```

## Patch Embedding

### PatchEmbed3D

3D Volume to Patch Embedding.

```python
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=14, tubelet_size=16, in_chans=1, embed_dim=384):
```

**Parameters:**
- `patch_size` (int): Size of patches in height and width dimensions
- `tubelet_size` (int): Size of patches in depth dimension
- `in_chans` (int): Number of input channels
- `embed_dim` (int): Embedding dimension for output tokens
