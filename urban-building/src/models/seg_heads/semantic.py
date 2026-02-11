# src/models/seg_heads/semantic.py
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.models.encoders.ptv3 import PTv3Encoder


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.head(features)


class SegAModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.encoder = PTv3Encoder(cfg)

        num_classes = cfg.data.get("num_classes", 13)
        self.head = SegmentationHead(
            in_channels=self.encoder.latent_dim,
            num_classes=num_classes,
            hidden_dim=256,
        )

        self.freeze_encoder = cfg.task.freeze_encoder

        if self.freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        encoded = self.encoder(feat, coord, batch)
        logits = self.head(encoded)

        return {
            "logits": logits,
            "features": encoded,
        }

    def predict(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.forward(feat, coord, batch)
        return torch.argmax(output["logits"], dim=-1)


def generate_pseudo_labels(
    coord: Tensor,
    rel_z: Tensor,
    height_threshold: float = 2.0,
    planarity_threshold: float = 0.8,
) -> Tensor:
    is_elevated = rel_z > height_threshold

    labels = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
    labels[is_elevated] = 1

    return labels
