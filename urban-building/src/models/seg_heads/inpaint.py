# src/models/seg_heads/inpaint.py
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.models.encoders.ptv3 import PTv3Encoder


class GeomInpaintHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.head(features)


class ColorDisplacementHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        rgb_channels: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rgb_channels),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.head(features)


class SegBGeomModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.encoder = PTv3Encoder(cfg)

        self.geom_head = GeomInpaintHead(
            in_channels=self.encoder.latent_dim,
            output_dim=3,
            hidden_dim=256,
        )

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        encoded = self.encoder(feat, coord, batch)
        xyz_pred = self.geom_head(encoded)

        return {
            "xyz_pred": xyz_pred,
            "features": encoded,
        }


class SegBColorModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.encoder = PTv3Encoder(cfg)

        self.geom_head = GeomInpaintHead(
            in_channels=self.encoder.latent_dim,
            output_dim=3,
            hidden_dim=256,
        )

        self.color_head = ColorDisplacementHead(
            in_channels=self.encoder.latent_dim,
            rgb_channels=3,
            hidden_dim=128,
        )

        self.geom_weight = cfg.task.loss.weights.geom
        self.color_weight = cfg.task.loss.weights.color

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        encoded = self.encoder(feat, coord, batch)
        xyz_pred = self.geom_head(encoded)
        rgb_pred = self.color_head(encoded)

        return {
            "xyz_pred": xyz_pred,
            "rgb_pred": rgb_pred,
            "features": encoded,
        }


class SegBv2Model(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.encoder = PTv3Encoder(cfg)

        self.geom_head = GeomInpaintHead(
            in_channels=self.encoder.latent_dim,
            output_dim=4,
            hidden_dim=256,
        )

        self.color_head = ColorDisplacementHead(
            in_channels=self.encoder.latent_dim,
            rgb_channels=3,
            hidden_dim=128,
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.geom_weight = cfg.task.loss.weights.geom
        self.color_weight = cfg.task.loss.weights.color

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        encoded = self.encoder(feat, coord, batch)
        xyzr_pred = self.geom_head(encoded)
        rgb_pred = self.color_head(encoded)
        anomaly_score = self.anomaly_head(encoded)

        return {
            "xyzr_pred": xyzr_pred,
            "rgb_pred": rgb_pred,
            "anomaly_score": anomaly_score,
            "features": encoded,
        }


class StructuredMasking:
    def __init__(
        self,
        targets: list = None,
    ):
        self.targets = targets or ["wall", "roof"]

    def __call__(
        self,
        coord: Tensor,
        labels: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        n_points = coord.shape[0]
        device = coord.device

        z = coord[:, 2]
        z_range = z.max() - z.min()

        mask = torch.zeros(n_points, dtype=torch.bool, device=device)

        if "roof" in self.targets:
            roof_threshold = z.min() + 0.8 * z_range
            roof_mask = z > roof_threshold
            sample_ratio = torch.rand(n_points, device=device) < 0.5
            mask = mask | (roof_mask & sample_ratio)

        if "wall" in self.targets:
            mid_low = z.min() + 0.3 * z_range
            mid_high = z.min() + 0.7 * z_range
            wall_mask = (z > mid_low) & (z < mid_high)
            sample_ratio = torch.rand(n_points, device=device) < 0.3
            mask = mask | (wall_mask & sample_ratio)

        visible_indices = torch.where(~mask)[0]
        masked_indices = torch.where(mask)[0]

        return visible_indices, masked_indices


class AnomalyMasking:
    def __init__(
        self,
        ratio: float = 0.3,
        mode: str = "random",
        density_threshold: float = 0.5,
    ):
        self.ratio = ratio
        self.mode = mode
        self.density_threshold = density_threshold

    def __call__(
        self,
        coord: Tensor,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        n_points = coord.shape[0]
        device = coord.device

        if self.mode == "random":
            mask = self._random_mask(n_points, device)
        elif self.mode == "spatial":
            mask = self._spatial_mask(coord, device)
        elif self.mode == "density":
            mask = self._density_mask(coord, batch, device)
        elif self.mode == "hybrid":
            mask = self._hybrid_mask(coord, batch, device)
        else:
            mask = self._random_mask(n_points, device)

        visible_indices = torch.where(~mask)[0]
        masked_indices = torch.where(mask)[0]

        return visible_indices, masked_indices

    def _random_mask(self, n_points: int, device: torch.device) -> Tensor:
        return torch.rand(n_points, device=device) < self.ratio

    def _spatial_mask(self, coord: Tensor, device: torch.device) -> Tensor:
        center = coord.mean(dim=0)
        dist = torch.norm(coord - center, dim=1)
        dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)

        threshold = torch.quantile(dist_norm, 1 - self.ratio)
        mask = dist_norm > threshold

        return mask

    def _density_mask(
        self,
        coord: Tensor,
        batch: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        n_points = coord.shape[0]
        mask = torch.zeros(n_points, dtype=torch.bool, device=device)

        grid_size = 1.0
        grid_coord = torch.floor(coord / grid_size).long()
        grid_coord = grid_coord - grid_coord.min(dim=0)[0]

        dims = grid_coord.max(dim=0)[0] + 1
        cell_ids = (
            grid_coord[:, 0] * dims[1] * dims[2]
            + grid_coord[:, 1] * dims[2]
            + grid_coord[:, 2]
        )

        unique_cells, inverse, counts = torch.unique(
            cell_ids, return_inverse=True, return_counts=True
        )

        point_density = counts[inverse].float()
        density_norm = point_density / point_density.max()

        low_density = density_norm < self.density_threshold
        random_sample = torch.rand(n_points, device=device) < self.ratio
        mask = low_density | random_sample

        return mask

    def _hybrid_mask(
        self,
        coord: Tensor,
        batch: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        n_points = coord.shape[0]

        spatial_mask = self._spatial_mask(coord, device)
        density_mask = self._density_mask(coord, batch, device)
        random_mask = self._random_mask(n_points, device)

        weights = torch.rand(3, device=device)
        weights = weights / weights.sum()

        combined = (
            weights[0] * spatial_mask.float()
            + weights[1] * density_mask.float()
            + weights[2] * random_mask.float()
        )

        threshold = torch.quantile(combined, 1 - self.ratio)
        mask = combined > threshold

        return mask
