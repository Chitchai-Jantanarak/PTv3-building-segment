# src/core/preprocessing/voxel.py
from typing import Optional

import numpy as np


def voxelize(
    xyz: np.ndarray,
    voxel_size: float,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    mode: str = "random",
) -> dict[str, np.ndarray]:
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
    order = np.argsort(inverse, kind="mergesort")

    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    rand_offsets = (np.random.random(len(counts)) * counts).astype(np.int64)
    return order[offsets[:-1] + rand_offsets]


def _sample_center(
    xyz: np.ndarray,
    unique_ids: np.ndarray,
    inverse: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    n_voxels = len(unique_ids)

    centers = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(centers, inverse, xyz)
    counts = np.bincount(inverse, minlength=n_voxels)
    centers /= counts[:, None]

    dists = np.linalg.norm(xyz - centers[inverse], axis=1)

    order = np.lexsort((dists, inverse))
    sorted_inverse = inverse[order]

    first_mask = np.empty(len(sorted_inverse), dtype=bool)
    first_mask[0] = True
    first_mask[1:] = sorted_inverse[1:] != sorted_inverse[:-1]

    return order[first_mask]


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
