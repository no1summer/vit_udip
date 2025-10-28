"""
Architecture documentation for ViT-UDIP.

This document describes the architecture and design decisions
of the Vision Transformer for Unsupervised Deep Image Processing.
"""

# ViT-UDIP Architecture

## Overview

ViT-UDIP is a Vision Transformer-based autoencoder designed for unsupervised learning on medical images. The architecture consists of an encoder-decoder structure where both components use transformer blocks.

## Key Components

### 1. Patch Embedding (PatchEmbed3D)

- **Purpose**: Converts 3D medical images into patch tokens
- **Input**: Volumetric images of shape (B, H, D, W)
- **Process**: 
  - Adds channel dimension: (B, H, D, W) → (B, 1, H, D, W)
  - Applies 3D convolution with non-overlapping patches
  - Flattens and transposes to token format
- **Output**: Patch tokens of shape (B, num_patches, embed_dim)

### 2. Positional Encoding

- **Type**: 3D sinusoidal positional embeddings
- **Purpose**: Provides spatial information to transformer blocks
- **Implementation**: 
  - Generates embeddings for height, width, and depth dimensions
  - Concatenates embeddings from all three dimensions
  - Uses sine and cosine functions for smooth interpolation

### 3. Vision Transformer Encoder

- **Architecture**: Standard transformer blocks with self-attention
- **Components**:
  - Patch embedding layer
  - Positional encoding
  - Multiple transformer blocks
  - Layer normalization
- **Optimization**: Non-zero patch filtering to reduce computational overhead

### 4. Vision Transformer Decoder

- **Architecture**: UDIP-style decoder with positional token concatenation
- **Key Features**:
  - Projects encoder features to decoder embedding space
  - Concatenates memory tokens with positional tokens
  - Uses transformer blocks for reconstruction
  - Predicts patches from positional tokens only

### 5. Loss Function

- **Type**: Mean Squared Error (MSE) loss
- **Scope**: Patch-wise reconstruction loss
- **Masking**: Only computes loss on non-zero patches

## Design Decisions

### Why Vision Transformer?

1. **Global Context**: Self-attention allows the model to capture long-range dependencies in medical images
2. **Scalability**: Transformer architecture scales well with model size
3. **Flexibility**: Can handle variable input sizes through patch-based processing

### Why 3D Patches?

1. **Medical Images**: Volumetric data requires 3D processing
2. **Efficiency**: 3D patches reduce computational complexity compared to pixel-wise processing
3. **Context**: Larger patches capture more anatomical context

### Why Unsupervised Learning?

1. **Data Efficiency**: No need for labeled data
2. **Representation Learning**: Learns meaningful features through reconstruction
3. **Generalization**: Can be applied to various medical imaging tasks

## Model Parameters

### Default Configuration

- **Patch Size**: 14×14×16 (height×width×depth)
- **Input Size**: 182×224×182 voxels
- **Encoder Embedding**: 128 dimensions
- **Decoder Embedding**: 64 dimensions
- **Encoder Depth**: 12 transformer blocks
- **Decoder Depth**: 12 transformer blocks
- **Attention Heads**: 8 heads

### Memory Requirements

- **Training**: ~8GB GPU memory (batch size 4)
- **Inference**: ~2GB GPU memory (single image)
- **Model Size**: ~50MB parameters

## Training Strategy

### Data Preprocessing

1. **Padding**: Images padded from 182×218×182 to 182×224×182
2. **Normalization**: Zero-mean, unit-variance normalization
3. **Masking**: Background voxels masked out

### Training Configuration

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 0.001 with cosine annealing
- **Batch Size**: 4 (adjustable based on GPU memory)
- **Epochs**: 300 (with early stopping)

### Validation Metrics

- **Loss**: Reconstruction MSE loss
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

## Usage Patterns

### Training

```python
from vit_udip.models import UDIPViT_engine
from vit_udip.training import train_model

model = UDIPViT_engine(encoder_embed_dim=128, decoder_embed_dim=64)
train_model(model, train_dataset, val_dataset)
```

### Inference

```python
from vit_udip.utils import extract_features

features = extract_features(model, "image.nii.gz")
```

### Feature Extraction

The model can be used for:
- **Representation Learning**: Extract meaningful features for downstream tasks
- **Image Reconstruction**: Generate high-quality reconstructions
- **Anomaly Detection**: Identify unusual patterns through reconstruction error
- **Image Enhancement**: Improve image quality through reconstruction
