# src/models/seg_heads/inpaint.py

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
        batch: Tensor | None = None,
        mask: Tensor | None = None,
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
        batch: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        encoded = self.encoder(feat, coord, batch)
        xyz_pred = self.geom_head(encoded)
        rgb_pred = self.color_head(encoded)

        return {
            "xyz_pred": xyz_pred,
            "rgb_pred": rgb_pred,
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
        labels: Tensor | None = None,
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
