# ViT-UDIP: Vision Transformer for Unsupervised Deep Image Processing

A PyTorch implementation of Vision Transformer (ViT) for unsupervised image deep learning, specifically designed for extracting meaningful features from medical images for genetic discovery and GWAS analysis.

## Overview

ViT-UDIP is a Vision Transformer-based autoencoder that learns meaningful representations from medical images through unsupervised reconstruction. The model uses 3D patch embedding, positional encoding, and transformer blocks to process volumetric medical data and extract deep features for downstream genetic analysis.

## Key Features

- **3D Vision Transformer Architecture**: Processes volumetric medical images using 3D patches
- **Unsupervised Feature Learning**: Learns meaningful representations through reconstruction without labels
- **Genetic Discovery Focus**: Designed specifically for extracting features for GWAS analysis
- **Efficient Patch Processing**: Optimized for non-zero patches to reduce computational overhead
- **Distributed Training**: Supports multi-GPU training with PyTorch DDP
- **Medical Image Processing**: Specialized for brain MRI data analysis

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

### Training for Feature Extraction

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

# Train model for feature extraction
train_model(model, train_dataset, num_epochs=300)
```

### Feature Extraction for GWAS

```python
from vit_udip.models import UDIPViT_engine
from vit_udip.utils.feature_extraction import extract_features

# Load trained model
model = UDIPViT_engine.from_checkpoint("path/to/checkpoint.pth")

# Extract features from medical images
features = extract_features(model, "path/to/image.nii.gz")

# Features can now be used for GWAS analysis
print(f"Extracted features shape: {features.shape}")
```

### Batch Feature Extraction

```python
from vit_udip.utils.feature_extraction import extract_features_batch

# Extract features from multiple images
image_paths = ["path/to/image1.nii.gz", "path/to/image2.nii.gz"]
all_features = extract_features_batch(model, image_paths, device="cuda")

# Stack features for analysis
stacked_features = torch.stack(all_features)
print(f"Batch features shape: {stacked_features.shape}")
```

## Use Cases

### 1. Genetic Discovery Pipeline

```python
# 1. Train ViT-UDIP on medical images
model = train_model(train_dataset)

# 2. Extract features from all subjects
features = extract_features_batch(model, all_image_paths)

# 3. Save features for GWAS analysis
torch.save(features, "extracted_features.pt")

# 4. Use features in downstream genetic analysis
# Features can be correlated with genetic variants, phenotypes, etc.
```

### 2. Phenotype-Genotype Association

The extracted features can be used to:
- **Identify genetic variants** associated with brain structure
- **Discover novel phenotypes** from medical imaging data
- **Perform GWAS analysis** using deep learning features
- **Study population genetics** across different ethnicities

### 3. Multi-Modal Analysis

```python
# Extract features from different modalities
t1_features = extract_features(model_t1, t1_images)
t2_features = extract_features(model_t2, t2_images)

# Combine features for comprehensive analysis
combined_features = torch.cat([t1_features, t2_features], dim=1)
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
│   ├── attention.py
│   ├── feature_extraction.py
│   └── reconstruction.py
├── data/             # Data handling
│   ├── __init__.py
│   └── dataset.py
├── training/          # Training scripts
│   ├── __init__.py
│   └── trainer.py
├── examples/         # Example scripts
│   ├── train_example.py
│   └── inference_example.py
└── docs/            # Documentation
    ├── architecture.md
    └── api_reference.md
```

## Research Applications

This package is designed for research in:

- **Medical Imaging Genetics**: Discovering genetic variants that influence brain structure
- **Population Genetics**: Studying genetic diversity across populations
- **Phenotype Discovery**: Identifying novel imaging-based phenotypes
- **GWAS Analysis**: Using deep learning features for genome-wide association studies
- **Multi-Ethnic Studies**: Analyzing genetic effects across different ethnic groups

## Performance

- **Feature Extraction**: ~128-dimensional features per subject
- **Training Time**: ~300 epochs for convergence
- **Memory Usage**: Optimized for GPU training with batch processing
- **Scalability**: Supports large-scale population studies

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vit_udip_2024,
  title={ViT-UDIP: Vision Transformer for Unsupervised Deep Image Processing in Genetic Discovery},
  author={no1summer},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on the Vision Transformer architecture
- Inspired by UDIP (Unsupervised Deep Image Processing) methodology
- Designed for medical image analysis and genetic discovery applications