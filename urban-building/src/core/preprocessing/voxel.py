# src/core/preprocessing/voxel.py
from typing import Dict, Optional, Tuple

import numpy as np


def voxelize(
    xyz: np.ndarray,
    voxel_size: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    mode: str = "random",
) -> Dict[str, np.ndarray]:
    voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)

    voxel_coords_min = voxel_coords.min(axis=0)
    voxel_coords = voxel_coords - voxel_coords_min

    dims = voxel_coords.max(axis=0) + 1
    voxel_ids = (
        voxel_coords[:, 0] * dims[1] * dims[2]
        + voxel_coords[:, 1] * dims[2]
        + voxel_coords[:, 2]
    )

    unique_ids, inverse, counts = np.unique(
        voxel_ids, return_inverse=True, return_counts=True
    )

    if mode == "random":
        sampled_indices = _sample_random(unique_ids, inverse, counts)
    elif mode == "center":
        sampled_indices = _sample_center(xyz, unique_ids, inverse, voxel_size)
    else:
        sampled_indices = _sample_random(unique_ids, inverse, counts)

    result = {
        "xyz": xyz[sampled_indices],
        "indices": sampled_indices,
        "voxel_coords": voxel_coords[sampled_indices],
    }

    if features is not None:
        result["features"] = features[sampled_indices]

    if labels is not None:
        result["labels"] = labels[sampled_indices]

    return result


def _sample_random(
    unique_ids: np.ndarray,
    inverse: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    n_voxels = len(unique_ids)
    sampled = np.zeros(n_voxels, dtype=np.int64)

    for i, uid in enumerate(unique_ids):
        mask = inverse == i
        indices = np.where(mask)[0]
        sampled[i] = np.random.choice(indices)

    return sampled


def _sample_center(
    xyz: np.ndarray,
    unique_ids: np.ndarray,
    inverse: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    n_voxels = len(unique_ids)
    sampled = np.zeros(n_voxels, dtype=np.int64)

    for i, uid in enumerate(unique_ids):
        mask = inverse == i
        indices = np.where(mask)[0]
        points = xyz[indices]
        center = points.mean(axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        sampled[i] = indices[np.argmin(dists)]

    return sampled


def compute_grid_coords(
    xyz: np.ndarray,
    grid_size: float,
) -> np.ndarray:
    return np.floor(xyz / grid_size).astype(np.int32)


def inverse_voxelize(
    voxel_coords: np.ndarray,
    voxel_size: float,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    xyz = voxel_coords.astype(np.float32) * voxel_size + voxel_size / 2
    if offset is not None:
        xyz = xyz + offset
    return xyz
