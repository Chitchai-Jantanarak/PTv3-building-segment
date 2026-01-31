"""
Datasets module for the Urban Building Pipeline.

Provides:
- Base dataset classes for point cloud data
- Task-specific dataset implementations:
    - MAEDataset: Self-supervised pretraining
    - SegADataset: Semantic segmentation
    - SegBDataset: Building inpainting
    - FEMADataset: HAZUS building classification
- Dataset builder functions
- Custom collate functions for variable-length point clouds

Usage:
    from src.datasets import build_dataset, build_dataloader

    # Build dataset from config
    dataset = build_dataset(cfg, split='train')

    # Build dataloader
    dataloader = build_dataloader(cfg, split='train')

    # Or use specific dataset classes
    from src.datasets import SensatUrbanDataset
    dataset = SensatUrbanDataset(root='data/processed', split='train')
"""

from .base import (
    BasePointCloudDataset,
    SimplePointCloudDataset,
    collate_fn,
    worker_init_fn,
)
from .builder import (
    # Registry
    DATASET_REGISTRY,
    FEMADataset,
    # Dataset classes
    GenericDataset,
    LASDataset,
    MAEDataset,
    SegADataset,
    SegBDataset,
    SensatUrbanDataset,
    WHUDataset,
    build_dataloader,
    # Builder functions
    build_dataset,
    get_available_datasets,
    get_dataset_class,
    get_num_classes,
    register_dataset,
)

__all__ = [
    # Base classes
    "BasePointCloudDataset",
    "SimplePointCloudDataset",
    # Dataset implementations
    "GenericDataset",
    "SensatUrbanDataset",
    "WHUDataset",
    "LASDataset",
    "MAEDataset",
    "SegADataset",
    "SegBDataset",
    "FEMADataset",
    # Registry
    "DATASET_REGISTRY",
    "register_dataset",
    "get_dataset_class",
    "get_available_datasets",
    # Builder functions
    "build_dataset",
    "build_dataloader",
    "get_num_classes",
    # Utilities
    "collate_fn",
    "worker_init_fn",
]
