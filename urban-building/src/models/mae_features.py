from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
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

_INFERRED_LAYOUTS: dict[int, list[str]] = {
    4: ["x", "y", "z", "rel_z"],
    5: ["x", "y", "z", "rel_z", "intensity"],
    7: ["x", "y", "z", "rel_z", "r", "g", "b"],
    8: ["x", "y", "z", "rel_z", "r", "g", "b", "intensity"],
}

_OPTIONAL_FLAGS = {"optional", "auto", "available"}
_DISABLED_FLAGS = {"false", "no", "off", "0", "disabled"}


def infer_feature_names(num_channels: int) -> list[str]:
    if num_channels in _INFERRED_LAYOUTS:
        return list(_INFERRED_LAYOUTS[num_channels])
    return list(CANONICAL_FEATURE_ORDER[:num_channels])


def _feature_cfg(cfg: DictConfig) -> DictConfig:
    task_features = cfg.task.get("features", None) if hasattr(cfg, "task") else None
    if task_features is not None:
        return task_features
    return cfg.data.features


def _flag_text(flag: object) -> str | None:
    if isinstance(flag, str):
        return flag.strip().lower()
    return None


def _is_disabled(flag: object) -> bool:
    text = _flag_text(flag)
    if text is not None:
        return text in _DISABLED_FLAGS
    return not bool(flag)


def _is_optional(flag: object) -> bool:
    text = _flag_text(flag)
    return text in _OPTIONAL_FLAGS if text is not None else False


def resolve_input_feature_names(cfg: DictConfig) -> list[str]:
    features_cfg = _feature_cfg(cfg)
    names: list[str] = []

    if not _is_disabled(features_cfg.get("xyz", True)):
        names.extend(["x", "y", "z"])
    if not _is_disabled(features_cfg.get("relz", True)):
        names.append("rel_z")
    if not _is_disabled(features_cfg.get("rgb", False)):
        names.extend(["r", "g", "b"])
    if not _is_disabled(features_cfg.get("intensity", False)):
        names.append("intensity")

    if not names:
        raise ValueError("MAE requires at least one input feature")

    return names


def resolve_optional_feature_names(cfg: DictConfig) -> set[str]:
    features_cfg = _feature_cfg(cfg)
    optional_names: set[str] = set()

    if _is_optional(features_cfg.get("rgb", False)):
        optional_names.update({"r", "g", "b"})
    if _is_optional(features_cfg.get("intensity", False)):
        optional_names.add("intensity")

    return optional_names


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


def resolve_target_feature_names(
    cfg: DictConfig,
    input_feature_names: Sequence[str],
) -> list[str]:
    target_cfg = cfg.task.loss.get("target", "all")

    if isinstance(target_cfg, (list, tuple, ListConfig)):
        requested = [str(item).strip().lower() for item in target_cfg]
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
            [item.strip().lower() for item in target_cfg.split(",") if item.strip()],
        )
    else:
        requested = list(input_feature_names)

    target_names = [name for name in requested if name in input_feature_names]
    if not target_names:
        raise ValueError("No valid MAE target features were resolved from the config")

    return target_names


def _normalize_feature_names(
    feature_names: Any,
    num_channels: int,
) -> list[str]:
    if feature_names is None:
        return infer_feature_names(num_channels)

    if isinstance(feature_names, np.ndarray):
        raw_names = feature_names.tolist()
    elif isinstance(feature_names, ListConfig) or isinstance(feature_names, Sequence) and not isinstance(
        feature_names, (str, bytes)
    ):
        raw_names = list(feature_names)
    else:
        raw_names = [feature_names]

    normalized: list[str] = []
    for item in raw_names[:num_channels]:
        if isinstance(item, bytes):
            text = item.decode("utf-8", errors="ignore")
        else:
            text = str(item)
        normalized.append(text.strip().lower())

    fallback = infer_feature_names(num_channels)
    while len(normalized) < num_channels:
        normalized.append(fallback[len(normalized)])

    return normalized


def build_feature_map(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    coords = np.asarray(data["coords"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"Expected coords with shape (N, 3), got {coords.shape}")

    n_points = int(coords.shape[0])
    feature_map: dict[str, np.ndarray] = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    }

    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != n_points:
        raise ValueError(
            f"Expected features with shape (N, C) matching coords, got {features.shape}"
        )

    feature_names = _normalize_feature_names(
        data.get("feature_names"), features.shape[1]
    )
    for index, name in enumerate(feature_names):
        feature_map[name] = features[:, index]

    rgb = data.get("rgb", None)
    if rgb is not None:
        rgb_array = np.asarray(rgb, dtype=np.float32)
        if rgb_array.shape[0] != n_points:
            raise ValueError(
                f"Expected rgb with {n_points} rows, got shape {rgb_array.shape}"
            )
        if rgb_array.ndim != 2 or rgb_array.shape[1] < 3:
            raise ValueError(f"Expected rgb with shape (N, 3), got {rgb_array.shape}")
        feature_map["r"] = rgb_array[:, 0]
        feature_map["g"] = rgb_array[:, 1]
        feature_map["b"] = rgb_array[:, 2]

    intensity = data.get("intensity", None)
    if intensity is not None:
        intensity_array = np.asarray(intensity, dtype=np.float32).reshape(-1)
        if intensity_array.shape[0] != n_points:
            raise ValueError(
                f"Expected intensity with {n_points} rows, got shape {intensity_array.shape}"
            )
        feature_map["intensity"] = intensity_array

    return feature_map


def assemble_requested_features(
    data: Mapping[str, np.ndarray],
    requested_feature_names: Sequence[str],
    optional_feature_names: Sequence[str] = (),
) -> tuple[np.ndarray, list[str]]:
    feature_map = build_feature_map(data)
    n_points = int(np.asarray(data["coords"]).shape[0])
    optional = set(optional_feature_names)
    columns: list[np.ndarray] = []

    for name in requested_feature_names:
        if name in feature_map:
            column = np.asarray(feature_map[name], dtype=np.float32).reshape(n_points)
        elif name in optional:
            column = np.zeros(n_points, dtype=np.float32)
        else:
            raise ValueError(
                f"Requested feature '{name}' is missing from the sample and is not optional"
            )
        columns.append(column)

    if not columns:
        return np.zeros((n_points, 0), dtype=np.float32), []

    return np.stack(columns, axis=1).astype(np.float32), list(requested_feature_names)
