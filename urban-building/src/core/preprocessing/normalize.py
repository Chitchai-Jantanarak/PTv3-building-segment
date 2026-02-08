# src/core/preprocessing/normalize.py
from typing import Optional

import numpy as np


def normalize_xyz(
    xyz: np.ndarray,
    center: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if center is None:
        center = xyz.mean(axis=0)

    xyz_centered = xyz - center

    if scale is None:
        scale = np.abs(xyz_centered).max()
        if scale == 0:
            scale = 1.0

    xyz_normalized = xyz_centered / scale

    return xyz_normalized, center, scale


def denormalize_xyz(
    xyz_normalized: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    return xyz_normalized * scale + center


def normalize_features(
    features: np.ndarray,
    method: str = "minmax",
) -> tuple[np.ndarray, dict]:
    if method == "minmax":
        fmin = features.min(axis=0)
        fmax = features.max(axis=0)
        denom = fmax - fmin
        denom[denom == 0] = 1.0
        normalized = (features - fmin) / denom
        params = {"min": fmin, "max": fmax}
    elif method == "zscore":
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        normalized = (features - mean) / std
        params = {"mean": mean, "std": std}
    else:
        normalized = features
        params = {}

    return normalized, params


def denormalize_features(
    features: np.ndarray,
    params: dict,
    method: str = "minmax",
) -> np.ndarray:
    if method == "minmax":
        return features * (params["max"] - params["min"]) + params["min"]
    elif method == "zscore":
        return features * params["std"] + params["mean"]
    return features
