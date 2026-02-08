# src/models/encoders/ptv3/point.py
from typing import Optional

import torch
from torch import Tensor


def build_point_dict(
    feat: Tensor,
    coord: Tensor,
    grid_size: float,
    batch: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    point = {
        "feat": feat,
        "coord": coord,
        "grid_size": grid_size,
    }

    if batch is not None:
        point["batch"] = batch
    elif offset is not None:
        point["offset"] = offset
    else:
        point["batch"] = torch.zeros(
            feat.shape[0], dtype=torch.long, device=feat.device
        )

    return point


def extract_features(point: dict[str, Tensor]) -> Tensor:
    return point["feat"]


def compute_offset_from_batch(batch: Tensor) -> Tensor:
    unique, counts = torch.unique_consecutive(batch, return_counts=True)
    offset = torch.cumsum(counts, dim=0)
    return offset


def compute_batch_from_offset(offset: Tensor, n_points: int) -> Tensor:
    batch = torch.zeros(n_points, dtype=torch.long, device=offset.device)
    start = 0
    for i, end in enumerate(offset):
        batch[start:end] = i
        start = end
    return batch
