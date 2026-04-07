from collections.abc import Iterable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

CANONICAL_FEATURE_ORDER = (
    "x",
    "y",
    "z",
    "rel_z",
    "r",
    "g",
    "b",
    "intensity",
)

_INFERRED_LAYOUTS = {
    4: ["x", "y", "z", "rel_z"],
    5: ["x", "y", "z", "rel_z", "intensity"],
    7: ["x", "y", "z", "rel_z", "r", "g", "b"],
    8: ["x", "y", "z", "rel_z", "r", "g", "b", "intensity"],
}

_OPTIONAL_FLAGS = {"optional", "auto", "available"}
_FALSE_FLAGS = {"false", "no", "off", "0", "disabled"}


def infer_feature_names(num_channels: int) -> list[str]:
    if num_channels in _INFERRED_LAYOUTS:
        return list(_INFERRED_LAYOUTS[num_channels])
    return list(CANONICAL_FEATURE_ORDER[:num_channels])


def _is_enabled(flag: object, available: bool = True) -> bool:
    if isinstance(flag, str):
        lowered = flag.lower()
        if lowered in _OPTIONAL_FLAGS:
            return available
        if lowered in _FALSE_FLAGS:
            return False
        return True
    return bool(flag)


def resolve_input_feature_names(cfg: DictConfig) -> list[str]:
    features_cfg = cfg.data.features
    names: list[str] = []

    if _is_enabled(features_cfg.get("xyz", True)):
        names.extend(["x", "y", "z"])
    if _is_enabled(features_cfg.get("relz", True)):
        names.append("rel_z")
    if _is_enabled(features_cfg.get("rgb", False), available=False):
        names.extend(["r", "g", "b"])
    if _is_enabled(features_cfg.get("intensity", False), available=False):
        names.append("intensity")

    return names


def resolve_available_feature_names(
    cfg: DictConfig,
    available_keys: Iterable[str],
) -> list[str]:
    available = set(available_keys)
    features_cfg = cfg.data.features
    names: list[str] = []

    if _is_enabled(features_cfg.get("xyz", True)):
        names.extend(["x", "y", "z"])
    if _is_enabled(features_cfg.get("relz", True)):
        names.append("rel_z")

    if _is_enabled(features_cfg.get("rgb", False), available="rgb" in available):
        if "rgb" not in available:
            raise ValueError(
                "RGB was requested but the point cloud does not contain it"
            )
        names.extend(["r", "g", "b"])

    if _is_enabled(
        features_cfg.get("intensity", False), available="intensity" in available
    ):
        if "intensity" not in available:
            raise ValueError(
                "Intensity was requested but the point cloud does not contain it"
            )
        names.append("intensity")

    return names


def get_feature_indices(
    feature_names: Sequence[str],
    target_names: Sequence[str],
    strict: bool = True,
) -> list[int]:
    index_map = {name: idx for idx, name in enumerate(feature_names)}
    indices: list[int] = []

    for name in target_names:
        if name in index_map:
            indices.append(index_map[name])
        elif strict:
            raise KeyError(f"Feature '{name}' is not present in {list(feature_names)}")

    return indices


def select_feature_columns(
    features: torch.Tensor,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    strict: bool = True,
) -> torch.Tensor:
    indices = get_feature_indices(feature_names, target_names, strict=strict)
    if not indices:
        return features.new_zeros((features.shape[0], 0))
    return features[:, indices]


def select_feature_array(
    features: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    strict: bool = True,
) -> np.ndarray:
    indices = get_feature_indices(feature_names, target_names, strict=strict)
    if not indices:
        return np.zeros((features.shape[0], 0), dtype=features.dtype)
    return features[:, indices]


def resolve_mae_target_feature_names(
    cfg: DictConfig,
    input_feature_names: Sequence[str],
) -> list[str]:
    target_cfg = cfg.task.loss.get("target", "all")

    if isinstance(target_cfg, (list, tuple, ListConfig)):
        requested = list(target_cfg)
    elif isinstance(target_cfg, str):
        presets = {
            "all": list(input_feature_names),
            "geom": ["x", "y", "z", "rel_z"],
            "xyz_relz": ["x", "y", "z", "rel_z"],
            "rgb": ["r", "g", "b"],
            "appearance": ["r", "g", "b", "intensity"],
        }
        requested = presets.get(
            target_cfg.lower(),
            [item.strip() for item in target_cfg.split(",") if item.strip()],
        )
    else:
        requested = list(input_feature_names)

    target_names = [name for name in requested if name in input_feature_names]
    if not target_names:
        raise ValueError("No valid MAE target features were resolved from the config")

    return target_names


def resolve_geom_feature_names(input_feature_names: Sequence[str]) -> list[str]:
    return [name for name in ["x", "y", "z", "rel_z"] if name in input_feature_names]


def resolve_attr_feature_names(input_feature_names: Sequence[str]) -> list[str]:
    return [
        name for name in ["r", "g", "b", "intensity"] if name in input_feature_names
    ]


def normalize_feature_targets(
    target: torch.Tensor,
    method: str = "none",
    eps: float = 1e-6,
) -> torch.Tensor:
    lowered = (method or "none").lower()
    if lowered == "none":
        return target
    if lowered == "zscore":
        mean = target.mean(dim=0, keepdim=True)
        std = target.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (target - mean) / std
    if lowered == "minmax":
        lower = target.amin(dim=0, keepdim=True)
        upper = target.amax(dim=0, keepdim=True)
        scale = (upper - lower).clamp_min(eps)
        return (target - lower) / scale
    raise ValueError(f"Unsupported normalization method: {method}")
