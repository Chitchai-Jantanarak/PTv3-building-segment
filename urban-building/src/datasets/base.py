"""
Base dataset class for point cloud processing.

Provides common functionality for all datasets:
- Loading preprocessed .pth files
- Voxelization and feature handling
- Data augmentation
- Batching utilities

All task-specific datasets should inherit from this class.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BasePointCloudDataset(Dataset, ABC):
    """
    Abstract base class for point cloud datasets.

    Subclasses must implement:
        - _load_file_list(): Returns list of data files
        - _get_class_info(): Returns class names and mapping
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        voxel_size: float = 0.04,
        max_points: Optional[int] = None,
        transform: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None,
        ignore_index: int = -1,
        cache_data: bool = False,
    ):
        """
        Args:
            root: Root directory containing processed data
            split: Data split ('train', 'val', 'test')
            voxel_size: Voxel size for grid sampling (0 to disable)
            max_points: Maximum number of points per sample (None for no limit)
            transform: Optional transform function
            feature_names: List of feature names to use
            ignore_index: Label value to ignore in loss computation
            cache_data: Whether to cache loaded data in memory
        """
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.transform = transform
        self.feature_names = feature_names or ["x", "y", "z", "rel_z"]
        self.ignore_index = ignore_index
        self.cache_data = cache_data

        # Data cache
        self._cache: Dict[int, Dict[str, Any]] = {}

        # Load file list
        self.file_list = self._load_file_list()

        if len(self.file_list) == 0:
            logger.warning(f"No data files found in {self.root} for split '{split}'")

        # Get class information
        self.class_names, self.class_mapping = self._get_class_info()
        self.num_classes = len(self.class_names) if self.class_names else 0

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.file_list)} samples "
            f"(split: {split}, voxel_size: {voxel_size})"
        )

    @abstractmethod
    def _load_file_list(self) -> List[Path]:
        """
        Load list of data files for the current split.

        Returns:
            List of file paths
        """
        raise NotImplementedError

    @abstractmethod
    def _get_class_info(
        self,
    ) -> Tuple[Optional[Dict[int, str]], Optional[Dict[int, int]]]:
        """
        Get class names and optional label mapping.

        Returns:
            class_names: Dict mapping class ID to name, or None
            class_mapping: Dict mapping original labels to new labels, or None
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - 'points': (N, D) point features
                - 'coords': (N, 3) coordinates
                - 'labels': (N,) semantic labels (if available)
                - Additional fields depending on dataset
        """
        # Check cache
        if self.cache_data and idx in self._cache:
            data = self._cache[idx]
        else:
            # Load from file
            file_path = self.file_list[idx]
            data = self._load_sample(file_path)

            # Cache if enabled
            if self.cache_data:
                self._cache[idx] = data

        # Make a copy to avoid modifying cached data
        data = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()
        }

        # Apply voxelization
        if self.voxel_size > 0:
            data = self._voxelize(data)

        # Apply point limit
        if self.max_points is not None and len(data["coords"]) > self.max_points:
            data = self._random_sample(data, self.max_points)

        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)

        # Convert to tensors
        data = self._to_tensors(data)

        return data

    def _load_sample(self, file_path: Path) -> Dict[str, np.ndarray]:
        """
        Load a single sample from file.

        Args:
            file_path: Path to .pth file

        Returns:
            Dictionary with numpy arrays
        """
        try:
            raw_data = torch.load(file_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

        # Extract fields
        coords = np.asarray(raw_data["coords"], dtype=np.float32)
        features = np.asarray(raw_data["features"], dtype=np.float32)

        data = {
            "coords": coords,
            "features": features,
            "file_path": str(file_path),
        }

        # Labels
        if "labels" in raw_data and raw_data["labels"] is not None:
            labels = np.asarray(raw_data["labels"], dtype=np.int64)

            # Apply label mapping if defined
            if self.class_mapping is not None:
                new_labels = np.full_like(labels, self.ignore_index)
                for old_label, new_label in self.class_mapping.items():
                    new_labels[labels == old_label] = new_label
                labels = new_labels

            data["labels"] = labels

        # Instance labels
        if "instance" in raw_data and raw_data["instance"] is not None:
            data["instance"] = np.asarray(raw_data["instance"], dtype=np.int64)

        # Additional metadata
        if "feature_names" in raw_data:
            data["feature_names"] = raw_data["feature_names"]

        return data

    def _voxelize(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply voxelization to the sample.

        Args:
            data: Sample dictionary

        Returns:
            Voxelized sample dictionary
        """
        coords = data["coords"]
        features = data["features"]

        # Compute voxel indices
        voxel_idx = np.floor(coords / self.voxel_size).astype(np.int64)

        # Shift to non-negative
        voxel_idx = voxel_idx - voxel_idx.min(axis=0)

        # Hash to unique keys
        max_idx = voxel_idx.max(axis=0) + 1
        keys = (
            voxel_idx[:, 0] * max_idx[1] * max_idx[2]
            + voxel_idx[:, 1] * max_idx[2]
            + voxel_idx[:, 2]
        )

        # Find unique voxels and randomly select one point per voxel
        unique_keys, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )

        # Random selection within each voxel
        selected_indices = np.zeros(len(unique_keys), dtype=np.int64)
        for i, key in enumerate(unique_keys):
            point_indices = np.where(keys == key)[0]
            selected_indices[i] = np.random.choice(point_indices)

        # Apply selection
        data["coords"] = coords[selected_indices]
        data["features"] = features[selected_indices]

        if "labels" in data:
            data["labels"] = data["labels"][selected_indices]

        if "instance" in data:
            data["instance"] = data["instance"][selected_indices]

        return data

    def _random_sample(
        self, data: Dict[str, np.ndarray], num_points: int
    ) -> Dict[str, np.ndarray]:
        """
        Randomly sample points from the data.

        Args:
            data: Sample dictionary
            num_points: Number of points to sample

        Returns:
            Sampled data dictionary
        """
        n = len(data["coords"])
        if n <= num_points:
            return data

        # Random selection
        indices = np.random.choice(n, num_points, replace=False)
        indices.sort()  # Keep spatial order

        data["coords"] = data["coords"][indices]
        data["features"] = data["features"][indices]

        if "labels" in data:
            data["labels"] = data["labels"][indices]

        if "instance" in data:
            data["instance"] = data["instance"][indices]

        return data

    def _to_tensors(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert numpy arrays to PyTorch tensors.

        Args:
            data: Sample dictionary with numpy arrays

        Returns:
            Sample dictionary with tensors
        """
        result = {}

        # Points/features
        result["points"] = torch.from_numpy(data["features"]).float()
        result["coords"] = torch.from_numpy(data["coords"]).float()

        # Labels
        if "labels" in data:
            result["labels"] = torch.from_numpy(data["labels"]).long()

        if "instance" in data:
            result["instance"] = torch.from_numpy(data["instance"]).long()

        # Preserve non-array fields
        for key in ["file_path", "feature_names"]:
            if key in data:
                result[key] = data[key]

        return result

    def get_class_weights(self, method: str = "inverse_freq") -> torch.Tensor:
        """
        Compute class weights for imbalanced training.

        Args:
            method: Weighting method ('inverse_freq', 'sqrt_inverse_freq', 'median_freq')

        Returns:
            (C,) tensor of class weights
        """
        if self.num_classes == 0:
            raise ValueError("No class information available")

        # Count labels across all samples
        label_counts = np.zeros(self.num_classes, dtype=np.int64)

        for idx in range(len(self)):
            file_path = self.file_list[idx]
            data = torch.load(file_path, map_location="cpu", weights_only=False)

            if "labels" in data and data["labels"] is not None:
                labels = np.asarray(data["labels"])
                labels = labels[labels >= 0]  # Ignore invalid labels
                labels = labels[labels < self.num_classes]

                for label in labels:
                    label_counts[label] += 1

        # Compute weights
        total = label_counts.sum()

        if method == "inverse_freq":
            weights = total / (self.num_classes * label_counts + 1e-6)
        elif method == "sqrt_inverse_freq":
            weights = np.sqrt(total / (self.num_classes * label_counts + 1e-6))
        elif method == "median_freq":
            median = np.median(label_counts[label_counts > 0])
            weights = median / (label_counts + 1e-6)
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Normalize
        weights = weights / weights.sum() * self.num_classes

        return torch.from_numpy(weights).float()

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "num_samples": len(self),
            "split": self.split,
            "voxel_size": self.voxel_size,
            "max_points": self.max_points,
            "num_classes": self.num_classes,
            "feature_names": self.feature_names,
        }

        if self.class_names:
            stats["class_names"] = self.class_names

        return stats


class SimplePointCloudDataset(BasePointCloudDataset):
    """
    Simple dataset implementation that loads all .pth files from a directory.

    Useful for generic datasets without specific structure.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        num_classes: int = 0,
        class_names: Optional[Dict[int, str]] = None,
        **kwargs,
    ):
        """
        Args:
            root: Root directory (should contain train/val/test subdirectories)
            split: Data split
            num_classes: Number of classes (for labels)
            class_names: Optional class name mapping
            **kwargs: Additional arguments for BasePointCloudDataset
        """
        self._num_classes = num_classes
        self._class_names = class_names
        super().__init__(root=root, split=split, **kwargs)

    def _load_file_list(self) -> List[Path]:
        """Load all .pth files from the split directory."""
        split_dir = self.root / self.split

        if not split_dir.exists():
            # Try loading directly from root
            split_dir = self.root

        files = sorted(split_dir.glob("*.pth"))
        return files

    def _get_class_info(
        self,
    ) -> Tuple[Optional[Dict[int, str]], Optional[Dict[int, int]]]:
        """Return class info passed to constructor."""
        return self._class_names, None


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for point cloud batches.

    Handles variable-length point clouds by creating batch indices.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with:
            - 'points': (B*N, D) concatenated features
            - 'coords': (B*N, 3) concatenated coordinates
            - 'batch': (B*N,) batch indices
            - 'labels': (B*N,) concatenated labels (if present)
            - 'offset': (B,) cumulative point counts
    """
    points_list = []
    coords_list = []
    labels_list = []
    batch_indices = []
    offset = [0]

    for i, sample in enumerate(batch):
        n = sample["points"].shape[0]

        points_list.append(sample["points"])
        coords_list.append(sample["coords"])
        batch_indices.append(torch.full((n,), i, dtype=torch.long))

        if "labels" in sample:
            labels_list.append(sample["labels"])

        offset.append(offset[-1] + n)

    result = {
        "points": torch.cat(points_list, dim=0),
        "coords": torch.cat(coords_list, dim=0),
        "batch": torch.cat(batch_indices, dim=0),
        "offset": torch.tensor(offset[1:], dtype=torch.long),
    }

    if labels_list:
        result["labels"] = torch.cat(labels_list, dim=0)

    return result


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker process with unique random seed.

    Use this as worker_init_fn in DataLoader for reproducibility.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
