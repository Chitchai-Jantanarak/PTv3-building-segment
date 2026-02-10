# Urban Building Point Cloud Pipeline

A multi-stage deep learning pipeline for urban point cloud processing, building segmentation, inpainting, and FEMA/HAZUS building classification.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a 4-stage training pipeline for urban building analysis:

| Stage | Name | Description | Input | Output |
|-------|------|-------------|-------|--------|
| **0** | MAE Pretraining | Self-supervised learning | XYZ + relZ | Pretrained encoder |
| **1** | SEG-A | Semantic segmentation | Frozen encoder | Per-point labels |
| **2** | SEG-B | Building inpainting | Building points | Reconstructed geometry |
| **3** | FEMA/HAZUS | Building classification | Clusters + features | Building codes |

## Features

- **Multi-stage Training Pipeline**: MAE → SEG-A → SEG-B → FEMA
- **Hydra Configuration**: Single YAML-based control for all operations
- **Multiple Datasets**: SensatUrban, WHU 3D, generic LAS/LAZ support
- **Block-level Masking**: Structured masking for better MAE pretraining
- **Hierarchical Classification**: HAZUS building codes (RES, COM, IND, GOV, EDU, AGR, REL)
- **DEM Integration**: Relative height computation from Digital Elevation Models

## Installation

### Prerequisites

- NVIDIA GPU with CUDA 11.8+
- Conda (Miniconda or Anaconda)
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/PTv3-building-segment.git
cd PTv3-building-segment/urban-building

# Run setup script
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Activate environment
conda activate urban-building
```

### Manual Setup

```bash
# Create conda environment
conda create -n urban-building python=3.10 -y
conda activate urban-building

# Install PyTorch with CUDA
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install GDAL
conda install gdal -c conda-forge

# Install project dependencies
pip install -e ".[all]"
```

## Directory Structure

```
urban-building/
├── main.py                 # 🔑 Single Hydra entry point
├── pyproject.toml            # Dependencies
│
├── configs/                  # 🔧 All configuration
│   ├── config.yaml           # Main controller
│   ├── task/
│   │   ├── mae.yaml          # Stage 0: MAE pretraining
│   │   ├── seg_a.yaml        # Stage 1: Segmentation
│   │   ├── seg_b_geom.yaml   # Stage 2: Geometry inpainting
│   │   ├── seg_b_color.yaml  # Stage 2: Color inpainting
│   │   └── fema.yaml         # Stage 3: HAZUS classification
│   ├── model/
│   │   ├── ptv3.yaml         # PointTransformerV3 config
│   │   └── mae.yaml          # MAE model config
│   └── data/
│       ├── sensat.yaml       # SensatUrban dataset
│       ├── whu.yaml          # WHU 3D dataset
│       └── las.yaml          # Generic LAS/DEM
│
├── data/
│   ├── raw/                  # Original datasets
│   │   ├── sensat/
│   │   ├── whu/
│   │   └── las/
│   ├── processed/            # Preprocessed .pth files
│   └── cache/
│
├── checkpoints/              # Model weights
│   ├── mae/
│   ├── seg_a/
│   ├── seg_b_geom/
│   ├── seg_b_color/
│   └── fema/
│
├── src/
│   ├── core/                 # Core utilities
│   │   ├── io/               # LAS/PLY/DEM readers
│   │   ├── preprocessing/    # Voxelization, features
│   │   └── utils/            # Logging, memory, seed
│   │
│   ├── datasets/             # Dataset classes
│   │   ├── base.py
│   │   └── builder.py
│   │
│   ├── models/
│   │   ├── encoders/ptv3/    # PTv3 backbone
│   │   ├── mae/              # MAE model
│   │   ├── seg_heads/        # Segmentation heads
│   │   └── hazus_head/       # FEMA classifier
│   │
│   ├── train/                # Training scripts
│   │   ├── mae.py
│   │   ├── seg_a.py
│   │   ├── seg_b_geom.py
│   │   └── fema.py
│   │
│   └── infer/                # Inference scripts
│
└── scripts/                  # Shell scripts
    └── setup_env.sh
```

## Configuration

All behavior is controlled via Hydra configuration files. The main entry point is `configs/config.yaml`.

### Basic Usage

```bash
# Default: MAE pretraining with SensatUrban
python main.py

# Specify task
python main.py task=seg_a

# Change dataset
python main.py task=seg_a data=whu

# Override parameters
python main.py task=mae training.epochs=100 data.batch_size=4

# Preprocessing mode
python main.py mode=preprocess data=sensat

# Inference mode
python main.py mode=infer task=seg_a
```

### Configuration Hierarchy

```yaml
defaults:
  - _self_
  - task: mae           # Task-specific settings
  - model: ptv3         # Model architecture
  - data: sensat        # Dataset configuration

mode: train             # preprocess | train | infer | pipeline
```

## Training Pipeline

### Stage 0: MAE Pretraining (Foundation)

Self-supervised pretraining using masked autoencoder approach with block-level masking.

```bash
python main.py task=mae
```

**Key Settings:**
- Input: XYZ + relative height (relZ)
- Masking: 70% block-level masking (not random points)
- Output: Pretrained encoder weights

### Stage 1: SEG-A (Semantic Segmentation)

Urban semantic segmentation following ScanNet/RandLA-Net style.

```bash
python main.py task=seg_a
```

**Classes (SensatUrban):**
| ID | Class | ID | Class |
|----|-------|----|----|
| 0 | Ground | 7 | Traffic Roads |
| 1 | High Vegetation | 8 | Street Furniture |
| 2 | **Buildings** | 9 | Cars |
| 3 | Walls | 10 | Footpath |
| 4 | Bridge | 11 | Bikes |
| 5 | Parking | 12 | Water |
| 6 | Rail | | |

### Stage 2: SEG-B (Building Inpainting)

Geometry completion for buildings using structured masking (walls, roofs).

```bash
# Geometry inpainting
python main.py task=seg_b_geom

# Color inpainting (requires color data)
python main.py task=seg_b_color
```

**Key Features:**
- Input: Building-only points (extracted via SEG-A)
- Masking: Structured removal (wall slabs, roof sections, corners)
- Output: Reconstructed geometry + anomaly scores

### Stage 3: FEMA/HAZUS Classification

Hierarchical building classification following FEMA HAZUS standards.

```bash
python main.py task=fema
```

**Classification Hierarchy:**

| Main Category | Sub-categories |
|--------------|----------------|
| **RES** (Residential) | RES1-RES6 (Single family to High-rise) |
| **COM** (Commercial) | COM1-COM10 (Retail, Wholesale, Banks, etc.) |
| **IND** (Industrial) | IND1-IND6 (Heavy, Light, Food, Metals, etc.) |
| **GOV** (Government) | GOV1-GOV2 (Services, Emergency) |
| **EDU** (Education) | EDU1-EDU2 (Schools, Universities) |
| **AGR** (Agriculture) | AGR1 |
| **REL** (Religious) | REL1 |

**Additional Attributes:**
- Stories: Low-rise (1-2), Mid-rise (3-5), High-rise (6+)
- Basement: Yes / No

## Datasets

### SensatUrban

Urban-scale outdoor point cloud dataset with 13 classes.

```bash
# Download from: https://github.com/QingyongHu/SensatUrban
# Place in: data/raw/sensat/

# Preprocess
python main.py mode=preprocess data=sensat
```

### WHU 3D

Chinese urban point cloud dataset with MLS (Mobile Laser Scanning) focus.

```bash
# Download from: https://whu3d.com/
# Place in: data/raw/whu/mls/

# Preprocess
python main.py mode=preprocess data=whu
```

### Custom LAS/LAZ Data

For inference on real-world LiDAR data:

```bash
# Place files in: data/raw/las/
# Optional DEM in: data/raw/dem/

# Preprocess
python main.py mode=preprocess data=las

# Run full pipeline
python main.py mode=pipeline
```

## Model Architecture

### PointTransformerV3 Encoder

The backbone encoder uses PointTransformerV3 with:
- Z-order serialization for efficient processing
- Multi-scale feature extraction
- Configurable depth and attention heads

### MAE Model

```
Input (XYZ + relZ) → Block Masking (70%) → PTv3 Encoder → Decoder → Reconstruction
```

### HAZUS Head

```
DLP Features (32-dim) → Shared MLP → Main Head (7 classes)
                                   ↓
                            Main Embedding
                                   ↓
                      Conditioned Sub Head (28 classes)
                                   ↓
                      Attribute Heads (Stories, Basement)
```

## References

- [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)
- [SensatUrban](https://github.com/QingyongHu/SensatUrban)
- [WHU 3D Dataset](https://whu3d.com/)
- [FEMA HAZUS Technical Manual](https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus)

## License

MIT License - see [LICENSE](LICENSE) for details.