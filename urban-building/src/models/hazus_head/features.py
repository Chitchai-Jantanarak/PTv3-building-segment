# src/models/hazus_head/features.py
from typing import Optional

import torch
from torch import Tensor


def compute_geometry_stats(
    xyz: Tensor,
) -> dict[str, Tensor]:
    device = xyz.device

    height = xyz[:, 2].max() - xyz[:, 2].min()

    xy = xyz[:, :2]
    xy_min = xy.min(dim=0)[0]
    xy_max = xy.max(dim=0)[0]
    xy_range = xy_max - xy_min

    footprint_area = xy_range[0] * xy_range[1]

    length = xy_range.max()
    width = xy_range.min()
    aspect_ratio = length / (width + 1e-6)

    centroid = xyz.mean(dim=0)

    return {
        "height": height,
        "footprint_area": footprint_area,
        "length": length,
        "width": width,
        "aspect_ratio": aspect_ratio,
        "centroid": centroid,
        "n_points": torch.tensor(xyz.shape[0], device=device, dtype=torch.float32),
    }


def compute_roof_features(
    xyz: Tensor,
    roof_threshold: float = 0.8,
) -> dict[str, Tensor]:
    z = xyz[:, 2]
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min

    roof_height = z_min + roof_threshold * z_range
    roof_mask = z > roof_height
    roof_points = xyz[roof_mask]

    if roof_points.shape[0] < 10:
        return {
            "roof_type": torch.tensor(0, device=xyz.device),
            "roof_slope": torch.tensor(0.0, device=xyz.device),
        }

    roof_z = roof_points[:, 2]
    roof_z_std = roof_z.std()
    roof_z_range = roof_z.max() - roof_z.min()

    if roof_z_std < 0.1 * z_range:
        roof_type = 0
    elif roof_z_range < 0.2 * z_range:
        roof_type = 1
    else:
        roof_type = 2

    roof_slope = roof_z_range / (z_range + 1e-6)

    return {
        "roof_type": torch.tensor(roof_type, device=xyz.device),
        "roof_slope": roof_slope,
    }


def extract_building_features(
    xyz: Tensor,
    mae_error: Optional[Tensor] = None,
) -> Tensor:
    geom = compute_geometry_stats(xyz)
    roof = compute_roof_features(xyz)

    features = [
        geom["height"].unsqueeze(0),
        geom["footprint_area"].unsqueeze(0),
        geom["aspect_ratio"].unsqueeze(0),
        geom["n_points"].unsqueeze(0),
        roof["roof_type"].float().unsqueeze(0),
        roof["roof_slope"].unsqueeze(0),
    ]

    if mae_error is not None:
        features.append(mae_error.mean().unsqueeze(0))

    return torch.cat(features, dim=0)


class HazusFeatureExtractor:
    def __init__(self):
        pass

    def extract(
        self,
        xyz: Tensor,
        mae_error: Optional[Tensor] = None,
    ) -> Tensor:
        return extract_building_features(xyz, mae_error)

    def extract_batch(
        self,
        xyz: Tensor,
        batch: Tensor,
        mae_errors: Optional[Tensor] = None,
    ) -> Tensor:
        unique_batches = torch.unique(batch)
        features_list = []

        for b in unique_batches:
            mask = batch == b
            xyz_b = xyz[mask]
            mae_b = mae_errors[mask] if mae_errors is not None else None
            feat_b = self.extract(xyz_b, mae_b)
            features_list.append(feat_b)

        return torch.stack(features_list, dim=0)
