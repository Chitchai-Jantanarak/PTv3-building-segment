"""
Dataset builder for the Urban Building Pipeline.

Provides unified interface to construct datasets based on Hydra configuration.
Supports multiple dataset types and task-specific data loading.

Usage:
    from src.datasets.builder import build_dataset, build_dataloader

    dataset = build_dataset(cfg, split='train')
    dataloader = build_dataloader(cfg, split='train')
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .base import BasePointCloudDataset, SimplePointCloudDataset, collate_fn

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class."""

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset_class(name: str) -> type:
    """Get dataset class by name."""
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")
    return DATASET_REGISTRY[name]


# =============================================================================
# Task-specific Dataset Classes
# =============================================================================


@register_dataset("generic")
class GenericDataset(SimplePointCloudDataset):
    """Generic dataset for preprocessed .pth files."""

    pass


@register_dataset("sensat")
class SensatUrbanDataset(BasePointCloudDataset):
    """
    SensatUrban dataset for urban semantic segmentation.

    Classes:
        0: Ground, 1: High Vegetation, 2: Buildings, 3: Walls,
        4: Bridge, 5: Parking, 6: Rail, 7: Traffic Roads,
        8: Street Furniture, 9: Cars, 10: Footpath, 11: Bikes, 12: Water
    """

    CLASSES = {
        0: "Ground",
        1: "High Vegetation",
        2: "Buildings",
        3: "Walls",
        4: "Bridge",
        5: "Parking",
        6: "Rail",
        7: "Traffic Roads",
        8: "Street Furniture",
        9: "Cars",
        10: "Footpath",
        11: "Bikes",
        12: "Water",
    }

    NUM_CLASSES = 13
    BUILDING_CLASS = 2

    def _load_file_list(self) -> list[Path]:
        """Load SensatUrban data files."""
        # Try split directory first
        split_dir = self.root / self.split
        if not split_dir.exists():
            split_dir = self.root / "sensat" / self.split
        if not split_dir.exists():
            split_dir = self.root

        files = sorted(split_dir.glob("*.pth"))

        if not files:
            # Try to find by pattern
            files = sorted(split_dir.glob("**/cambridge*.pth"))
            files.extend(sorted(split_dir.glob("**/birmingham*.pth")))

        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return self.CLASSES, None


@register_dataset("whu")
class WHUDataset(BasePointCloudDataset):
    """
    WHU 3D dataset for urban point cloud processing.

    Supports MLS (Mobile Laser Scanning) mode with simplified class mapping.
    """

    # Simplified 9-class mapping
    SIMPLIFIED_CLASSES = {
        0: "Ground",
        1: "Vegetation",
        2: "Building",
        3: "Pole",
        4: "Vehicle",
        5: "Pedestrian",
        6: "Street Furniture",
        7: "Infrastructure",
        8: "Others",
    }

    NUM_SIMPLIFIED_CLASSES = 9
    BUILDING_CLASS = 2

    def __init__(
        self,
        root: Union[str, Path],
        mode: str = "mls",  # als | mls | mls-w
        use_simplified: bool = True,
        **kwargs,
    ):
        self.mode = mode
        self.use_simplified = use_simplified
        super().__init__(root=root, **kwargs)

    def _load_file_list(self) -> list[Path]:
        """Load WHU data files for specified mode."""
        # Try mode-specific directory
        mode_dir = self.root / self.mode / self.split
        if not mode_dir.exists():
            mode_dir = self.root / "whu" / self.mode / self.split
        if not mode_dir.exists():
            mode_dir = self.root / self.split

        files = sorted(mode_dir.glob("*.pth"))
        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        if self.use_simplified:
            return self.SIMPLIFIED_CLASSES, None
        return None, None


@register_dataset("las")
class LASDataset(BasePointCloudDataset):
    """
    Generic LAS/LAZ dataset for inference.

    Loads preprocessed .pth files from LAS point clouds.
    Typically used for inference pipeline.
    """

    def _load_file_list(self) -> list[Path]:
        """Load LAS data files."""
        las_dir = self.root / "las" / self.split
        if not las_dir.exists():
            las_dir = self.root / self.split
        if not las_dir.exists():
            las_dir = self.root

        files = sorted(las_dir.glob("*.pth"))
        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return None, None


# =============================================================================
# Task-specific Datasets
# =============================================================================


@register_dataset("mae")
class MAEDataset(BasePointCloudDataset):
    """
    Dataset for MAE (Masked Autoencoder) pretraining.

    Returns point clouds without labels for self-supervised learning.
    Supports block-level masking.
    """

    def __init__(
        self,
        root: Union[str, Path],
        mask_ratio: float = 0.7,
        block_size: float = 0.5,
        **kwargs,
    ):
        self.mask_ratio = mask_ratio
        self.block_size = block_size

        # MAE doesn't use labels
        kwargs["ignore_index"] = -1
        super().__init__(root=root, **kwargs)

    def _load_file_list(self) -> list[Path]:
        """Load files for MAE training (uses all available data)."""
        # MAE can use all data regardless of labels
        all_files = []
        for pattern in ["*.pth", "**/*.pth"]:
            all_files.extend(self.root.glob(pattern))

        # Remove duplicates and sort
        return sorted(set(all_files))

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return None, None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get sample with masking applied."""
        data = super().__getitem__(idx)

        # Remove labels if present (not needed for MAE)
        data.pop("labels", None)

        return data


@register_dataset("seg_a")
class SegADataset(BasePointCloudDataset):
    """
    Dataset for SEG-A (semantic segmentation) task.

    Returns point clouds with semantic labels.
    Uses SensatUrban or WHU class definitions.
    """

    def __init__(
        self,
        root: Union[str, Path],
        dataset_type: str = "sensat",  # sensat | whu
        **kwargs,
    ):
        self.dataset_type = dataset_type

        if dataset_type == "sensat":
            self._class_names = SensatUrbanDataset.CLASSES
            self._num_classes = SensatUrbanDataset.NUM_CLASSES
        elif dataset_type == "whu":
            self._class_names = WHUDataset.SIMPLIFIED_CLASSES
            self._num_classes = WHUDataset.NUM_SIMPLIFIED_CLASSES
        else:
            self._class_names = None
            self._num_classes = 0

        super().__init__(root=root, **kwargs)

    def _load_file_list(self) -> list[Path]:
        """Load files for segmentation."""
        split_dir = self.root / self.split
        if not split_dir.exists():
            split_dir = self.root / self.dataset_type / self.split
        if not split_dir.exists():
            split_dir = self.root

        return sorted(split_dir.glob("*.pth"))

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return self._class_names, None


@register_dataset("seg_b")
@register_dataset("seg_b_geom")
@register_dataset("seg_b_color")
class SegBDataset(BasePointCloudDataset):
    """
    Dataset for SEG-B (building inpainting) task.

    Returns building-only points with structured masking for inpainting.
    Separates visible and target (masked) points.
    """

    def __init__(
        self,
        root: Union[str, Path],
        building_class: int = 2,
        mask_ratio: float = 0.75,
        mask_type: str = "structured",  # structured | random
        include_color: bool = False,
        **kwargs,
    ):
        self.building_class = building_class
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.include_color = include_color
        super().__init__(root=root, **kwargs)

    def _load_file_list(self) -> list[Path]:
        """Load files for building inpainting."""
        split_dir = self.root / self.split
        if not split_dir.exists():
            split_dir = self.root

        files = sorted(split_dir.glob("*.pth"))

        # Filter to files that contain buildings
        # (In practice, this filtering should be done during preprocessing)
        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return None, None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get sample with building filtering and masking."""
        data = super().__getitem__(idx)

        # Filter to building points if labels available
        if "labels" in data:
            building_mask = data["labels"] == self.building_class

            if building_mask.sum() > 100:  # Enough building points
                data["points"] = data["points"][building_mask]
                data["coords"] = data["coords"][building_mask]
                data.pop("labels", None)  # No longer needed

        # Apply structured masking
        data = self._apply_masking(data)

        return data

    def _apply_masking(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply structured masking for inpainting."""
        coords = data["coords"]
        points = data["points"]
        n = coords.shape[0]

        if n < 64:
            # Not enough points, return as-is
            data["visible"] = points
            data["target"] = points
            data["mask"] = torch.zeros(n, dtype=torch.bool)
            return data

        # Create mask based on type
        if self.mask_type == "structured":
            mask = self._structured_mask(coords)
        else:
            mask = self._random_mask(n)

        # Split into visible and target
        visible_mask = ~mask

        data["visible"] = points[visible_mask]
        data["visible_coords"] = coords[visible_mask]
        data["target"] = points[mask]
        data["target_coords"] = coords[mask]
        data["mask"] = mask

        return data

    def _structured_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """Create structured mask (wall/roof removal)."""
        n = coords.shape[0]
        z = coords[:, 2]

        # Randomly choose mask strategy
        strategy = torch.randint(0, 3, (1,)).item()

        if strategy == 0:
            # Horizontal slice removal (walls)
            z_range = z.max() - z.min()
            slice_z = z.min() + torch.rand(1).item() * z_range
            slice_thickness = z_range * self.mask_ratio * 0.5
            mask = torch.abs(z - slice_z) < slice_thickness

        elif strategy == 1:
            # Top removal (roof)
            threshold = torch.quantile(z, 1.0 - self.mask_ratio)
            mask = z >= threshold

        else:
            # Quadrant removal
            x, y = coords[:, 0], coords[:, 1]
            x_mid, y_mid = x.median(), y.median()

            quadrant = torch.randint(0, 4, (1,)).item()
            if quadrant == 0:
                mask = (x > x_mid) & (y > y_mid)
            elif quadrant == 1:
                mask = (x > x_mid) & (y <= y_mid)
            elif quadrant == 2:
                mask = (x <= x_mid) & (y > y_mid)
            else:
                mask = (x <= x_mid) & (y <= y_mid)

        # Ensure we don't mask everything
        if mask.sum() > n * 0.95:
            keep_idx = torch.randperm(mask.sum())[: int(n * 0.05)]
            masked_indices = torch.where(mask)[0]
            mask[masked_indices[keep_idx]] = False

        return mask

    def _random_mask(self, n: int) -> torch.Tensor:
        """Create random point mask."""
        num_mask = int(n * self.mask_ratio)
        mask = torch.zeros(n, dtype=torch.bool)
        mask_idx = torch.randperm(n)[:num_mask]
        mask[mask_idx] = True
        return mask


@register_dataset("fema")
@register_dataset("hazus")
class FEMADataset(Dataset):
    """
    Dataset for FEMA/HAZUS building classification.

    Returns pre-extracted building features (geometric + MAE error)
    for hierarchical classification.
    """

    MAIN_CLASSES = ["RES", "COM", "IND", "GOV", "EDU", "AGR", "REL"]
    NUM_MAIN_CLASSES = 7
    NUM_SUB_CLASSES = 28

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        feature_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.feature_dim = feature_dim

        # Load pre-extracted building features
        self.samples = self._load_samples()

        logger.info(
            f"Initialized FEMADataset with {len(self.samples)} building samples"
        )

    def _load_samples(self) -> list[dict[str, Any]]:
        """Load pre-extracted building feature samples."""
        samples = []

        feature_dir = self.root / "fema_features" / self.split
        if not feature_dir.exists():
            feature_dir = self.root / self.split

        # Load all feature files
        for file_path in sorted(feature_dir.glob("*.pth")):
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=False)

                # Handle both single building and multi-building files
                if "dlp" in data:
                    # Single building
                    samples.append(data)
                elif "buildings" in data:
                    # Multiple buildings per file
                    samples.extend(data["buildings"])

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a building sample."""
        sample = self.samples[idx]

        # Extract DLP (Detailed Level Parameters) features
        dlp = sample.get("dlp", sample.get("features"))
        if isinstance(dlp, np.ndarray):
            dlp = torch.from_numpy(dlp).float()
        elif not isinstance(dlp, torch.Tensor):
            dlp = torch.tensor(dlp, dtype=torch.float32)

        # Pad or truncate to expected dimension
        if dlp.shape[0] < self.feature_dim:
            dlp = torch.nn.functional.pad(dlp, (0, self.feature_dim - dlp.shape[0]))
        elif dlp.shape[0] > self.feature_dim:
            dlp = dlp[: self.feature_dim]

        result = {"dlp": dlp}

        # Labels
        if "main_label" in sample:
            result["main_label"] = torch.tensor(sample["main_label"], dtype=torch.long)
        if "sub_label" in sample:
            result["sub_label"] = torch.tensor(sample["sub_label"], dtype=torch.long)
        if "label" in sample:
            # Generic label (backward compatibility)
            result["label"] = torch.tensor(sample["label"], dtype=torch.long)

        # Optional attributes
        if "stories" in sample:
            result["stories"] = torch.tensor(sample["stories"], dtype=torch.long)
        if "basement" in sample:
            result["basement"] = torch.tensor(sample["basement"], dtype=torch.long)

        return result


# =============================================================================
# Builder Functions
# =============================================================================


def build_dataset(
    cfg: DictConfig,
    split: str = "train",
    task: Optional[str] = None,
    transform: Optional[Callable] = None,
) -> Dataset:
    """
    Build dataset from configuration.

    Args:
        cfg: Hydra configuration
        split: Data split ('train', 'val', 'test')
        task: Task name override (uses cfg.task.name if None)
        transform: Optional data transform

    Returns:
        Dataset instance
    """
    # Determine task
    if task is None:
        task = cfg.task.name if hasattr(cfg, "task") else "generic"

    # Get data config
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg
    task_cfg = cfg.task if hasattr(cfg, "task") else {}

    # Determine root directory
    if hasattr(cfg, "paths"):
        root = Path(cfg.paths.processed)
    else:
        root = Path(data_cfg.get("root", "data/processed"))

    # Common kwargs
    kwargs = {
        "root": root,
        "split": split,
        "voxel_size": data_cfg.get("voxel_size", 0.04),
        "max_points": data_cfg.get("max_points", None),
        "transform": transform,
        "ignore_index": data_cfg.get("ignore_index", -1),
    }

    # Task-specific dataset selection
    if task in ["mae"]:
        dataset_cls = get_dataset_class("mae")
        kwargs["mask_ratio"] = task_cfg.get("model", {}).get("mask_ratio", 0.7)

    elif task in ["seg_a"]:
        dataset_cls = get_dataset_class("seg_a")
        kwargs["dataset_type"] = data_cfg.get("name", "sensat")

    elif task in ["seg_b", "seg_b_geom", "seg_b_color"]:
        dataset_cls = get_dataset_class("seg_b")
        kwargs["building_class"] = data_cfg.get("filter_class", 2)
        kwargs["mask_ratio"] = (
            task_cfg.get("model", {}).get("masking", {}).get("ratio", 0.75)
        )
        kwargs["include_color"] = "color" in task

    elif task in ["fema", "hazus"]:
        dataset_cls = get_dataset_class("fema")
        # FEMA has different kwargs
        return dataset_cls(
            root=root,
            split=split,
            feature_dim=task_cfg.get("features", {}).get("input_dim", 32),
        )

    else:
        # Try to get dataset by name
        dataset_name = data_cfg.get("name", "generic")
        try:
            dataset_cls = get_dataset_class(dataset_name)
        except ValueError:
            dataset_cls = get_dataset_class("generic")

    return dataset_cls(**kwargs)


def build_dataloader(
    cfg: DictConfig,
    split: str = "train",
    task: Optional[str] = None,
    transform: Optional[Callable] = None,
    collate: Optional[Callable] = None,
) -> DataLoader:
    """
    Build DataLoader from configuration.

    Args:
        cfg: Hydra configuration
        split: Data split
        task: Task name override
        transform: Optional data transform
        collate: Optional custom collate function

    Returns:
        DataLoader instance
    """
    # Build dataset
    dataset = build_dataset(cfg, split=split, task=task, transform=transform)

    # Get data config
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg

    # DataLoader kwargs
    is_train = split == "train"

    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 0)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": is_train,
        "num_workers": num_workers,
        "pin_memory": data_cfg.get("pin_memory", True),
        "drop_last": is_train and len(dataset) > batch_size,
        "persistent_workers": num_workers > 0,
    }

    # Use custom collate for variable-length point clouds
    if collate is None:
        # FEMA dataset doesn't need special collate
        task_name = task or (cfg.task.name if hasattr(cfg, "task") else "")
        if task_name not in ["fema", "hazus"]:
            collate = collate_fn

    if collate is not None:
        loader_kwargs["collate_fn"] = collate

    return DataLoader(dataset, **loader_kwargs)


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_datasets() -> list[str]:
    """Get list of registered dataset names."""
    return list(DATASET_REGISTRY.keys())


def get_num_classes(cfg: DictConfig) -> int:
    """Get number of classes from configuration."""
    # Check task config first
    if hasattr(cfg, "task") and hasattr(cfg.task, "dataset"):
        return cfg.task.dataset.get("num_classes", 0)

    # Check data config
    if hasattr(cfg, "data") and hasattr(cfg.data, "classes"):
        return cfg.data.classes.get("num_classes", 0)

    # Dataset-specific defaults
    data_name = cfg.data.name if hasattr(cfg, "data") else "generic"

    if data_name == "sensat":
        return SensatUrbanDataset.NUM_CLASSES
    elif data_name == "whu":
        return WHUDataset.NUM_SIMPLIFIED_CLASSES

    return 0


# Import numpy for type checking
import numpy as np
