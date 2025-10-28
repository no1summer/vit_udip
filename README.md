# ViT-UDIP: Vision Transformer for Unsupervised Deep Image Processing

A PyTorch implementation of Vision Transformer (ViT) for unsupervised deep image processing, specifically designed for medical image reconstruction and analysis.

## Overview

ViT-UDIP is a Vision Transformer-based autoencoder that learns meaningful representations from medical images through unsupervised reconstruction. The model uses 3D patch embedding, positional encoding, and transformer blocks to process volumetric medical data.

## Key Features

- **3D Vision Transformer Architecture**: Processes volumetric medical images using 3D patches
- **Unsupervised Learning**: Learns representations through reconstruction without labels
- **Efficient Patch Processing**: Optimized for non-zero patches to reduce computational overhead
- **Distributed Training**: Supports multi-GPU training with PyTorch DDP
- **Medical Image Focus**: Designed specifically for brain MRI data processing

## Architecture

- **Encoder**: Vision Transformer with 3D patch embedding and positional encoding
- **Decoder**: Transformer-based decoder with positional token concatenation
- **Patch Size**: 14×14×16 (height×width×depth)
- **Input Size**: 182×224×182 voxels
- **Embedding Dimensions**: Configurable (default: 128 encoder, 64 decoder)

## Installation

### Option 1: Install from source (recommended)

```bash
# Clone the repository
git clone https://github.com/no1summer/vit-udip.git
cd vit-udip

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option 2: Install dependencies manually

```bash
pip install torch>=1.12.0 torchvision>=0.13.0
pip install nibabel>=3.2.0 pandas>=1.3.0 numpy>=1.21.0
pip install scikit-image>=0.19.0 tqdm>=4.62.0
pip install tensorboard>=2.8.0
```

### Option 3: Using requirements.txt

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```python
from vit_udip.models import UDIPViT_engine
from vit_udip.data import MedicalImageDataset
from vit_udip.training import train_model

# Initialize model
model = UDIPViT_engine(
    lr=0.001,
    encoder_embed_dim=128,
    decoder_embed_dim=64,
    encoder_depth=12,
    decoder_depth=12,
    num_heads=8
)

# Load data
train_dataset = MedicalImageDataset(
    datafile="path/to/train.csv",
    modality="T1_unbiased_linear"
)

# Train model
train_model(model, train_dataset, num_epochs=300)
```

### Feature Extraction

```python
from vit_udip.models import UDIPViT_engine
from vit_udip.utils.feature_extraction import extract_features

# Load trained model
model = UDIPViT_engine.from_checkpoint("path/to/checkpoint.pth")

# Extract features
features = extract_features(model, "path/to/image.nii.gz")
```

### Reconstruction

```python
from vit_udip.models import UDIPViT_engine
from vit_udip.utils.reconstruction import reconstruct_image

# Load trained model
model = UDIPViT_engine.from_checkpoint("path/to/checkpoint.pth")

# Reconstruct image
reconstructed = reconstruct_image(model, "path/to/image.nii.gz", output_path="reconstruction.nii.gz")
```

## File Structure

```
vit_udip/
├── models/           # Model definitions
│   ├── __init__.py
│   ├── vit_encoder.py
│   ├── vit_decoder.py
│   └── udip_engine.py
├── utils/            # Utility functions
│   ├── __init__.py
│   ├── positional_encoding.py
│   ├── patch_embedding.py
│   └── attention.py
├── data/             # Data handling
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
├── training/          # Training scripts
│   ├── __init__.py
│   ├── trainer.py
│   └── validation.py
├── examples/         # Example scripts
│   ├── train_example.py
│   └── inference_example.py
└── docs/            # Documentation
    ├── architecture.md
    └── api_reference.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the Vision Transformer architecture
- Inspired by UDIP (Unsupervised Deep Image Processing) methodology
- Designed for medical image analysis applications
