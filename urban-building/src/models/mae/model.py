# src/models/mae/model.py

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.losses import masked_mse_loss
from src.models.mae.decoder import MAEDecoder
from src.models.mae.encoder import MAEEncoder
from src.models.mae.masking import BlockMasking
from src.models.mae_features import (
    get_feature_indices,
    resolve_input_feature_names,
    resolve_target_feature_names,
)


class MAEModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.input_feature_names = resolve_input_feature_names(cfg)
        self.target_feature_names = resolve_target_feature_names(
            cfg, self.input_feature_names
        )
        self.target_feature_indices = get_feature_indices(
            self.input_feature_names,
            self.target_feature_names,
        )

        self.encoder = MAEEncoder(cfg)

        encoder_input_dim = int(cfg.model.in_channels)
        if bool(cfg.model.get("intensity_channel", False)):
            encoder_input_dim += 1

        if len(self.input_feature_names) == encoder_input_dim:
            self.input_adapter: nn.Module = nn.Identity()
        else:
            self.input_adapter = nn.Linear(
                len(self.input_feature_names),
                encoder_input_dim,
            )

        self.decoder = MAEDecoder(
            cfg=cfg,
            latent_dim=self.encoder.latent_dim,
            output_dim=len(self.target_feature_names),
        )

        self.masking = BlockMasking(
            ratio=cfg.task.masking.ratio,
            block_size=cfg.task.masking.block_size,
        )

        self.register_buffer(
            "feature_loss_weights",
            self._build_loss_weights()
        )

    def _build_loss_weights(self) -> torch.Tensor:
        weights = torch.ones(len(self.target_feature_names))
        for i, name in enumerate(self.target_feature_names):
            if name in ("z", "rel_z"):
                weights[i] = 3.0 # x3
        return weights

    @staticmethod
    def _per_sample_stats(
        target_feat: Tensor,
        batch: Tensor,
        min_std: float = 1e-3,
    ) -> tuple[Tensor, Tensor, Tensor]:
        n, f = target_feat.shape
        device = target_feat.device
        batch_max = int(batch.max().item()) + 1

        mean_b = torch.zeros(batch_max, f, device=device, dtype=target_feat.dtype)
        std_b = torch.ones(batch_max, f, device=device, dtype=target_feat.dtype)
        valid_b = torch.zeros(batch_max, f, device=device, dtype=torch.bool)

        for b in range(batch_max):
            mask = batch == b
            if mask.sum() <= 1:
                continue
            sample = target_feat[mask]
            m = sample.mean(dim=0)
            s = sample.std(dim=0, unbiased=False)
            mean_b[b] = m
            valid_b[b] = s > min_std
            std_b[b] = s.clamp(min=min_std)

        mean = mean_b[batch]
        std = std_b[batch]
        valid = valid_b[batch]
        return mean, std, valid


    def forward(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Tensor | None = None,
        offset: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if feat.ndim != 2:
            raise ValueError(
                f"Expected feat with shape (N, C), got {tuple(feat.shape)}"
            )
        if feat.shape[1] != len(self.input_feature_names):
            raise ValueError(
                f"Expected {len(self.input_feature_names)} MAE input channels "
                f"{self.input_feature_names}, got tensor with shape {tuple(feat.shape)}"
            )

        if batch is None:
            batch = torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device)

        encoder_feat = self.input_adapter(feat)
        visible_idx, masked_idx, visible_mask = self.masking(coord, batch)

        visible_feat = encoder_feat[visible_idx]
        visible_coord = coord[visible_idx]
        visible_batch = batch[visible_idx]

        encoded = self.encoder(visible_feat, visible_coord, visible_batch)

        target_feat = feat[:, self.target_feature_indices]
        visible_raw_color = target_feat[visible_idx, 4:]

        reconstructed_norm = self.decoder(
            encoded=encoded,
            visible_indices=visible_idx,
            masked_indices=masked_idx,
            n_total=feat.shape[0],
            coord=coord,
            batch=batch,
            visible_raw_feat=visible_raw_color,
        )

        target_feat = feat[:, self.target_feature_indices]
        mean, std, valid = self._per_sample_stats(target_feat, batch)
        reconstructed = reconstructed_norm * std + mean

        return {
            "reconstructed":      reconstructed,
            "reconstructed_norm": reconstructed_norm,
            "target_mean":        mean,
            "target_std":         std,
            "target_valid":       valid,
            "visible_indices":    visible_idx,
            "masked_indices":     masked_idx,
            "visible_mask":       visible_mask,
            "encoded":            encoded,
        }

    def compute_loss(
        self,
        output: dict[str, Tensor],
        target: Tensor,
    ) -> Tensor:
        reconstructed_norm = output["reconstructed_norm"]
        masked_idx = output["masked_indices"]
        mean = output["target_mean"]
        std = output["target_std"]
        valid = output["target_valid"]

        target_norm = (target - mean) / std

        weights = self.feature_loss_weights.to(reconstructed_norm.dtype)
        diff = (reconstructed_norm - target_norm) ** 2  
        diff = diff * weights.unsqueeze(0) * valid.to(diff.dtype)

        denom = (valid[masked_idx].to(diff.dtype) * weights.unsqueeze(0)).sum()
        loss = diff[masked_idx].sum() / denom.clamp(min=1.0)

        return loss

    def build_target(self, feat: Tensor) -> Tensor:
        return feat[:, self.target_feature_indices]

    def encode(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if feat.shape[1] != len(self.input_feature_names):
            raise ValueError(
                f"Expected {len(self.input_feature_names)} MAE input channels "
                f"{self.input_feature_names}, got tensor with shape {tuple(feat.shape)}"
            )
        return self.encoder(self.input_adapter(feat), coord, batch)


class MAEForPretraining(MAEModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def training_step(
        self,
        feat: Tensor,
        coord: Tensor,
        batch: Tensor | None = None,
    ) -> dict[str, Tensor]:
        target = self.build_target(feat)

        output = self.forward(feat, coord, batch)
        loss = self.compute_loss(output, target)

        return {
            "loss": loss,
            "reconstructed": output["reconstructed"],
            "masked_indices": output["masked_indices"],
        }
