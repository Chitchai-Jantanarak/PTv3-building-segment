# src/models/mae/encoder.py
from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.models.encoders.ptv3 import PTv3Encoder


class MAEEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.encoder = PTv3Encoder(cfg)
        self.latent_dim = self.encoder.latent_dim

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encoder(feat, coord, batch, offset)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
