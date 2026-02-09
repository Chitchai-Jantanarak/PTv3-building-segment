# src/models/encoders/ptv3/wrapper.py
import sys
from pathlib import Path
from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

PTv3_PARENT = Path(__file__).parent.parent.parent.parent.parent.parent
if str(PTv3_PARENT) not in sys.path:
    sys.path.insert(0, str(PTv3_PARENT))

from PointTransformerV3.model import PointTransformerV3  # noqa: E402

from src.models.encoders.ptv3.point import (  # noqa: E402
    build_point_dict,
    extract_features,
)


class PTv3Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        in_channels = cfg.model.in_channels
        if hasattr(cfg.model, "intensity_channel") and cfg.model.intensity_channel:
            in_channels += 1

        self.net = PointTransformerV3(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=list(cfg.model.enc_depths),
            enc_channels=list(cfg.model.enc_channels),
            enc_num_head=[c // 32 for c in cfg.model.enc_channels],
            enc_patch_size=[cfg.model.patch_size] * len(cfg.model.enc_depths),
            dec_depths=list(cfg.model.dec_depths),
            dec_channels=list(cfg.model.dec_channels),
            dec_num_head=[c // 32 for c in cfg.model.dec_channels],
            dec_patch_size=[cfg.model.patch_size] * len(cfg.model.dec_depths),
            mlp_ratio=4,
            enable_flash=cfg.model.enable_flash,
            cls_mode=False,
        )

        self.grid_size = cfg.model.grid_size
        self.latent_dim = cfg.model.dec_channels[0]

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        point = build_point_dict(
            feat=feat,
            coord=coord,
            grid_size=self.grid_size,
            batch=batch,
            offset=offset,
        )

        point = self.net(point)

        return extract_features(point)

    def forward_dict(self, point: dict[str, Tensor]) -> dict[str, Tensor]:
        if "grid_size" not in point:
            point["grid_size"] = self.grid_size
        return self.net(point)


class PTv3EncoderOnly(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        in_channels = cfg.model.in_channels
        if hasattr(cfg.model, "intensity_channel") and cfg.model.intensity_channel:
            in_channels += 1

        bottleneck_dim = cfg.model.enc_channels[-1]
        self.net = PointTransformerV3(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=list(cfg.model.enc_depths),
            enc_channels=list(cfg.model.enc_channels),
            enc_num_head=[c // 32 for c in cfg.model.enc_channels],
            enc_patch_size=[cfg.model.patch_size] * len(cfg.model.enc_depths),
            dec_depths=[1],
            dec_channels=[bottleneck_dim],
            dec_num_head=[bottleneck_dim // 32],
            dec_patch_size=[cfg.model.patch_size],
            mlp_ratio=4,
            enable_flash=cfg.model.enable_flash,
            cls_mode=False,
        )

        self.grid_size = cfg.model.grid_size
        self.latent_dim = bottleneck_dim

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
    ) -> Tensor:
        point = build_point_dict(
            feat=feat,
            coord=coord,
            grid_size=self.grid_size,
            batch=batch,
            offset=offset,
        )

        point = self.net(point)

        return extract_features(point)
