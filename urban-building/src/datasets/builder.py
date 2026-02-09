# src/dataset/base.py
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .base import BasePointCloudDataset, SimplePointCloudDataset, collate_fn


def _has_hydra_cwd() -> bool:
    try:
        import hydra.utils

        hydra.utils.get_original_cwd()
        return True
    except (ImportError, ValueError):
        return False


logger = logging.getLogger(__name__)


DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset_class(name: str) -> type:
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")
    return DATASET_REGISTRY[name]


@register_dataset("generic")
class GenericDataset(SimplePointCloudDataset):
    pass


@register_dataset("sensat")
class SensatUrbanDataset(BasePointCloudDataset):
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
    BINARY_MAPPING = {
        0: 0,
        1: 0,
        2: 1,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
    }

    def _load_file_list(self) -> list[Path]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            split_dir = self.root / "sensat" / self.split
        if not split_dir.exists():
            split_dir = self.root

        files = sorted(split_dir.glob("*.pth"))

        if not files:
            files = sorted(split_dir.glob("**/cambridge*.pth"))
            files.extend(sorted(split_dir.glob("**/birmingham*.pth")))

        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return self.CLASSES, None


@register_dataset("whu")
class WHUDataset(BasePointCloudDataset):
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
    RAW_TO_SIMPLIFIED = {
        100300: 0,
        102400: 1,
        102600: 1,
        104202: 1,
        102000: 2,
        102100: 2,
        105800: 3,
        100600: 4,
        100500: 5,
        100100: 6,
        100400: 6,
    }
    BINARY_MAPPING = {
        102000: 1,
        102100: 1,
        100300: 0,
        100100: 0,
        100400: 0,
        102400: 0,
        102600: 0,
        104202: 0,
        105800: 0,
        100500: 0,
        100600: 0,
    }

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
            return self.SIMPLIFIED_CLASSES, self.RAW_TO_SIMPLIFIED
        return None, None


@register_dataset("las")
class LASDataset(BasePointCloudDataset):
    def _load_file_list(self) -> list[Path]:
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


@register_dataset("mae")
class MAEDataset(BasePointCloudDataset):
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
        split_dir = self.root / self.split
        if split_dir.exists():
            files = sorted(split_dir.glob("*.pth")) + sorted(split_dir.glob("*.npz"))
            if files:
                return files

        files = sorted(self.root.glob("*.pth")) + sorted(self.root.glob("*.npz"))
        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return None, None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = super().__getitem__(idx)

        data.pop("labels", None)

        return data


@register_dataset("seg_a")
class SegADataset(BasePointCloudDataset):
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

        return files

    def _get_class_info(
        self,
    ) -> tuple[Optional[dict[int, str]], Optional[dict[int, int]]]:
        return None, None

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = super().__getitem__(idx)

        if "labels" in data:
            building_mask = data["labels"] == self.building_class

            if building_mask.sum() > 100:  # Enough building points
                data["points"] = data["points"][building_mask]
                data["coords"] = data["coords"][building_mask]
                data.pop("labels", None)  # No longer needed

        data = self._apply_masking(data)

        return data

    def _apply_masking(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = data["coords"]
        points = data["points"]
        n = coords.shape[0]

        if n < 64:
            data["visible"] = points
            data["target"] = points
            data["mask"] = torch.zeros(n, dtype=torch.bool)
            return data

        if self.mask_type == "structured":
            mask = self._structured_mask(coords)
        else:
            mask = self._random_mask(n)

        visible_mask = ~mask

        data["visible"] = points[visible_mask]
        data["visible_coords"] = coords[visible_mask]
        data["target"] = points[mask]
        data["target_coords"] = coords[mask]
        data["mask"] = mask

        return data

    def _structured_mask(self, coords: torch.Tensor) -> torch.Tensor:
        n = coords.shape[0]
        z = coords[:, 2]

        strategy = torch.randint(0, 3, (1,)).item()

        # Horizontal slice removal (walls)
        if strategy == 0:
            z_range = z.max() - z.min()
            slice_z = z.min() + torch.rand(1).item() * z_range
            slice_thickness = z_range * self.mask_ratio * 0.5
            mask = torch.abs(z - slice_z) < slice_thickness

        # Top removal (roof)
        elif strategy == 1:
            threshold = torch.quantile(z, 1.0 - self.mask_ratio)
            mask = z >= threshold

        # Quadrant removal
        else:
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

        if mask.sum() > n * 0.95:
            keep_idx = torch.randperm(mask.sum())[: int(n * 0.05)]
            masked_indices = torch.where(mask)[0]
            mask[masked_indices[keep_idx]] = False

        return mask

    def _random_mask(self, n: int) -> torch.Tensor:
        num_mask = int(n * self.mask_ratio)
        mask = torch.zeros(n, dtype=torch.bool)
        mask_idx = torch.randperm(n)[:num_mask]
        mask[mask_idx] = True
        return mask


@register_dataset("fema")
@register_dataset("hazus")
class FEMADataset(Dataset):
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

        self.samples = self._load_samples()

        logger.info(
            f"Initialized FEMADataset with {len(self.samples)} building samples"
        )

    def _load_samples(self) -> list[dict[str, Any]]:
        samples = []

        feature_dir = self.root / "fema_features" / self.split
        if not feature_dir.exists():
            feature_dir = self.root / self.split

        for file_path in sorted(feature_dir.glob("*.pth")):
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=False)

                # Single building
                if "dlp" in data:
                    samples.append(data)
                # Multiple buildings per file
                elif "buildings" in data:
                    samples.extend(data["buildings"])

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        dlp = sample.get("dlp", sample.get("features"))
        if isinstance(dlp, np.ndarray):
            dlp = torch.from_numpy(dlp).float()
        elif not isinstance(dlp, torch.Tensor):
            dlp = torch.tensor(dlp, dtype=torch.float32)

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
            result["label"] = torch.tensor(sample["label"], dtype=torch.long)

        if "stories" in sample:
            result["stories"] = torch.tensor(sample["stories"], dtype=torch.long)
        if "basement" in sample:
            result["basement"] = torch.tensor(sample["basement"], dtype=torch.long)

        return result


def build_dataset(
    cfg: DictConfig,
    split: str = "train",
    task: Optional[str] = None,
    transform: Optional[Callable] = None,
) -> Dataset:
    if task is None:
        task = cfg.task.name if hasattr(cfg, "task") else "generic"

    data_cfg = cfg.data if hasattr(cfg, "data") else cfg
    task_cfg = cfg.task if hasattr(cfg, "task") else {}

    if hasattr(cfg, "paths") and hasattr(cfg.paths, "processed"):
        root = Path(cfg.paths.processed)
    else:
        data_name = data_cfg.get("name", "")
        root = Path("data/processed") / data_name
    if not root.is_absolute():
        import hydra.utils as _hu

        orig_cwd = Path(_hu.get_original_cwd()) if _has_hydra_cwd() else Path.cwd()
        root = orig_cwd / root

    kwargs = {
        "root": root,
        "split": split,
        "voxel_size": data_cfg.get("voxel_size", 0.04),
        "max_points": data_cfg.get("max_points", None),
        "transform": transform,
        "ignore_index": data_cfg.get("ignore_index", -1),
    }

    if task in ["mae"]:
        dataset_cls = get_dataset_class("mae")
        masking = task_cfg.get("masking", {})
        kwargs["mask_ratio"] = (
            masking.get("ratio", 0.75) if hasattr(masking, "get") else 0.75
        )

    elif task in ["seg_a"]:
        dataset_cls = get_dataset_class("seg_a")
        kwargs["dataset_type"] = data_cfg.get("name", "sensat")

    elif task in ["seg_b", "seg_b_geom", "seg_b_color"]:
        dataset_cls = get_dataset_class("seg_b")
        kwargs["building_class"] = data_cfg.get("filter_class", 2)
        masking = task_cfg.get("masking", {})
        kwargs["mask_ratio"] = (
            masking.get("ratio", 0.75) if hasattr(masking, "get") else 0.75
        )
        kwargs["include_color"] = "color" in task

    elif task in ["fema", "hazus"]:
        dataset_cls = get_dataset_class("fema")
        return dataset_cls(
            root=root,
            split=split,
        )

    else:
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
    dataset = build_dataset(cfg, split=split, task=task, transform=transform)

    data_cfg = cfg.data if hasattr(cfg, "data") else cfg

    is_train = split == "train"

    task_cfg = cfg.task if hasattr(cfg, "task") else {}
    batch_size = task_cfg.get("batch_size", data_cfg.get("batch_size", 8))
    num_workers = data_cfg.get("num_workers", 0)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": is_train,
        "num_workers": num_workers,
        "pin_memory": data_cfg.get("pin_memory", True),
        "drop_last": is_train and len(dataset) > batch_size,
        "persistent_workers": num_workers > 0,
    }

    if collate is None:
        task_name = task or (cfg.task.name if hasattr(cfg, "task") else "")
        if task_name not in ["fema", "hazus"]:
            collate = collate_fn

    if collate is not None:
        loader_kwargs["collate_fn"] = collate

    return DataLoader(dataset, **loader_kwargs)


def get_available_datasets() -> list[str]:
    return list(DATASET_REGISTRY.keys())


def get_num_classes(cfg: DictConfig) -> int:
    if hasattr(cfg, "task") and hasattr(cfg.task, "dataset"):
        return cfg.task.dataset.get("num_classes", 0)

    if hasattr(cfg, "data") and hasattr(cfg.data, "classes"):
        return cfg.data.classes.get("num_classes", 0)

    data_name = cfg.data.name if hasattr(cfg, "data") else "generic"

    if data_name == "sensat":
        return SensatUrbanDataset.NUM_CLASSES
    elif data_name == "whu":
        return WHUDataset.NUM_SIMPLIFIED_CLASSES

    return 0
