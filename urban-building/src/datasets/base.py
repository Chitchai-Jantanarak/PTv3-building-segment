# src/dataset/base.py
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BasePointCloudDataset(Dataset, ABC):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        voxel_size: float = 0.04,
        max_points: Optional[int] = None,
        transform: Optional[Callable] = None,
        feature_names: Optional[list[str]] = None,
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

        self._cache: dict[int, dict[str, Any]] = {}

        self.file_list = self._load_file_list()

        if len(self.file_list) == 0:
            logger.warning(f"No data files found in {self.root} for split '{split}'")

        self.class_names, self.class_mapping = self._get_class_info()
        self.num_classes = len(self.class_names) if self.class_names else 0

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.file_list)} samples "
            f"(split: {split}, voxel_size: {voxel_size})"
        )

    @abstractmethod
    def _load_file_list(self) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache_data and idx in self._cache:
            data = self._cache[idx]
        else:
            file_path = self.file_list[idx]
            data = self._load_sample(file_path)

            if self.cache_data:
                self._cache[idx] = data

        data = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()
        }

        if self.max_points is not None and len(data["coords"]) > self.max_points:
            data = self._random_sample(data, self.max_points)

        if self.voxel_size > 0:
            data = self._voxelize(data)

        if self.transform is not None:
            data = self.transform(data)

        data = self._to_tensors(data)

        return data

    def _load_sample(self, file_path: Path) -> dict[str, np.ndarray]:
        try:
            if file_path.suffix == ".npz":
                # Lazy NpzFile: only arrays accessed via [] are decompressed
                raw_data = np.load(file_path, allow_pickle=True)
            else:
                raw_data = torch.load(file_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

        coords_key = "coords" if "coords" in raw_data else "xyz"
        coords = np.asarray(raw_data[coords_key], dtype=np.float32)
        features = np.asarray(raw_data["features"], dtype=np.float32)

        data = {
            "coords": coords,
            "features": features,
            "file_path": str(file_path),
        }

        if "labels" in raw_data:
            labels_raw = raw_data["labels"]
            if labels_raw is not None:
                labels = np.asarray(labels_raw, dtype=np.int64)

                if self.class_mapping is not None:
                    new_labels = np.full_like(labels, self.ignore_index)
                    for old_label, new_label in self.class_mapping.items():
                        new_labels[labels == old_label] = new_label
                    labels = new_labels

                data["labels"] = labels

        if "instance" in raw_data:
            inst = raw_data["instance"]
            if inst is not None:
                data["instance"] = np.asarray(inst, dtype=np.int64)

        if "rgb" in raw_data:
            rgb_raw = raw_data["rgb"]
            if rgb_raw is not None:
                rgb = np.asarray(rgb_raw, dtype=np.float32)
                if rgb.shape[0] != coords.shape[0] and "indices" in raw_data:
                    idx = np.asarray(raw_data["indices"], dtype=np.int64)
                    rgb = rgb[idx]
                if rgb.shape[0] == coords.shape[0]:
                    data["rgb"] = rgb

        if "feature_names" in raw_data:
            data["feature_names"] = raw_data["feature_names"]

        if hasattr(raw_data, "close"):
            raw_data.close()

        return data

    def _voxelize(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        coords = data["coords"]

        voxel_idx = np.floor(coords / self.voxel_size).astype(np.int64)
        voxel_idx = voxel_idx - voxel_idx.min(axis=0)

        max_idx = voxel_idx.max(axis=0) + 1
        keys = (
            voxel_idx[:, 0] * max_idx[1] * max_idx[2]
            + voxel_idx[:, 1] * max_idx[2]
            + voxel_idx[:, 2]
        )

        order = np.argsort(keys)
        sorted_keys = keys[order]
        breaks = np.flatnonzero(np.diff(sorted_keys)) + 1
        groups = np.split(order, breaks)

        selected_indices = np.array(
            [g[np.random.randint(len(g))] for g in groups], dtype=np.int64
        )

        data["coords"] = coords[selected_indices]
        data["features"] = data["features"][selected_indices]

        if "labels" in data:
            data["labels"] = data["labels"][selected_indices]
        if "instance" in data:
            data["instance"] = data["instance"][selected_indices]
        if "rgb" in data:
            data["rgb"] = data["rgb"][selected_indices]

        return data

    def _random_sample(
        self, data: dict[str, np.ndarray], num_points: int
    ) -> dict[str, np.ndarray]:
        n = len(data["coords"])
        if n <= num_points:
            return data

        indices = np.random.choice(n, num_points, replace=False)
        indices.sort()

        data["coords"] = data["coords"][indices]
        data["features"] = data["features"][indices]

        if "labels" in data:
            data["labels"] = data["labels"][indices]

        if "instance" in data:
            data["instance"] = data["instance"][indices]

        if "rgb" in data:
            data["rgb"] = data["rgb"][indices]

        return data

    def _to_tensors(self, data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        result = {}

        coords = data["coords"]
        centroid = coords.mean(axis=0)
        coords = coords - centroid

        result["points"] = torch.from_numpy(data["features"]).float()
        result["coords"] = torch.from_numpy(coords).float()

        if "labels" in data:
            result["labels"] = torch.from_numpy(data["labels"]).long()

        if "instance" in data:
            result["instance"] = torch.from_numpy(data["instance"]).long()

        if "rgb" in data:
            result["rgb"] = torch.from_numpy(data["rgb"]).float()

        for key in ["file_path", "feature_names"]:
            if key in data:
                result[key] = data[key]

        return result

    def get_class_weights(self, method: str = "inverse_freq") -> torch.Tensor:
        if self.num_classes == 0:
            raise ValueError("No class information available")

        label_counts = np.zeros(self.num_classes, dtype=np.int64)

        for idx in range(len(self)):
            file_path = self.file_list[idx]
            data = torch.load(file_path, map_location="cpu", weights_only=False)

            if "labels" in data and data["labels"] is not None:
                labels = np.asarray(data["labels"])

                if self.class_mapping is not None:
                    mapped = np.full_like(labels, -1)
                    for old_label, new_label in self.class_mapping.items():
                        mapped[labels == old_label] = new_label
                    labels = mapped

                labels = labels[labels >= 0]
                labels = labels[labels < self.num_classes]

                for label in labels:
                    label_counts[label] += 1

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

        weights = weights / weights.sum() * self.num_classes

        return torch.from_numpy(weights).float()

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
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
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        num_classes: int = 0,
        class_names: Optional[dict[int, str]] = None,
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

    def _load_file_list(self) -> list[Path]:
        split_dir = self.root / self.split

        if not split_dir.exists():
            split_dir = self.root

        files = sorted(split_dir.glob("*.pth")) + sorted(split_dir.glob("*.npz"))
        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return self._class_names, None


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
            - 'visible'/'visible_coords'/'target'/'target_coords'/'mask':
              (present when SegBDataset provides masking output)
    """
    points_list = []
    coords_list = []
    labels_list = []
    rgb_list = []
    batch_indices = []
    offset = [0]

    # Seg-B masking keys
    _MASK_KEYS = ("visible", "visible_coords", "target", "target_coords", "mask")
    mask_lists: dict[str, list] = {k: [] for k in _MASK_KEYS}
    mask_batch_lists: dict[str, list] = {k: [] for k in _MASK_KEYS}

    for i, sample in enumerate(batch):
        n = sample["points"].shape[0]

        points_list.append(sample["points"])
        coords_list.append(sample["coords"])
        batch_indices.append(torch.full((n,), i, dtype=torch.long))

        if "labels" in sample:
            labels_list.append(sample["labels"])

        if "rgb" in sample:
            rgb_list.append(sample["rgb"])

        for k in _MASK_KEYS:
            if k in sample:
                t = sample[k]
                mask_lists[k].append(t)
                mask_batch_lists[k].append(
                    torch.full((t.shape[0],), i, dtype=torch.long)
                )

        offset.append(offset[-1] + n)

    result = {
        "points": torch.cat(points_list, dim=0),
        "coords": torch.cat(coords_list, dim=0),
        "batch": torch.cat(batch_indices, dim=0),
        "offset": torch.tensor(offset[1:], dtype=torch.long),
    }

    if labels_list:
        result["labels"] = torch.cat(labels_list, dim=0)

    if rgb_list:
        result["rgb"] = torch.cat(rgb_list, dim=0)

    for k in _MASK_KEYS:
        if mask_lists[k]:
            result[k] = torch.cat(mask_lists[k], dim=0)
            result[f"{k}_batch"] = torch.cat(mask_batch_lists[k], dim=0)

    return result


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)
