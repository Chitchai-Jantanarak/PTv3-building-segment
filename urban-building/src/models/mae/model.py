# src/models/mae/model.py
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.losses import masked_mse_loss
from src.models.mae.decoder import MAEDecoder, TransformerDecoder
from src.models.mae.encoder import MAEEncoder
from src.models.mae.masking import BlockMasking


class MAEModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.encoder = MAEEncoder(cfg)

        self.decoder = MAEDecoder(
            cfg=cfg,
            latent_dim=self.encoder.latent_dim,
            output_dim=4,
        )

        self.masking = BlockMasking(
            ratio=cfg.task.masking.ratio,
            block_size=cfg.task.masking.block_size,
        )

        self.cfg = cfg

    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if batch is None:
            batch = torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device)

        visible_idx, masked_idx, visible_mask = self.masking(coord, batch)

        visible_feat = feat[visible_idx]
        visible_coord = coord[visible_idx]
        visible_batch = batch[visible_idx]

        encoded = self.encoder(visible_feat, visible_coord, visible_batch)

        reconstructed = self.decoder(
            encoded=encoded,
            visible_indices=visible_idx,
            masked_indices=masked_idx,
            n_total=feat.shape[0],
        )

        return {
            "reconstructed": reconstructed,
            "visible_indices": visible_idx,
            "masked_indices": masked_idx,
            "visible_mask": visible_mask,
            "encoded": encoded,
        }

    def compute_loss(
        self,
        output: Dict[str, Tensor],
        target: Tensor,
    ) -> Tensor:
        reconstructed = output["reconstructed"]
        masked_idx = output["masked_indices"]

        mask = torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)
        mask[masked_idx] = True

        loss = masked_mse_loss(reconstructed, target, mask)

        return loss

    def encode(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encoder(feat, coord, batch)


class MAEForPretraining(MAEModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def training_step(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        target = feat[:, :4]

        output = self.forward(feat, coord, batch)
        loss = self.compute_loss(output, target)

        return {
            "loss": loss,
            "reconstructed": output["reconstructed"],
            "masked_indices": output["masked_indices"],
        }
