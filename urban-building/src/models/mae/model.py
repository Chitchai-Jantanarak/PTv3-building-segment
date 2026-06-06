# src/models/mae/model.py

import os
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.core.utils import as_plain_dict, get_logger
from src.losses import masked_mse_loss
from src.models.mae.decoder import MAEDecoder
from src.models.mae.encoder import MAEEncoder
from src.models.mae.masking import BlockMasking
from src.models.mae.rgbi_head import RGBIHead
from src.models.mae_features import (
    get_feature_indices,
    resolve_input_feature_names,
    resolve_target_feature_names,
)

AUTO_WEIGHTS_PATH = Path("configs/task/_auto/mae_feature_weights.yaml")


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

        geom_set = {"x", "y", "z", "rel_z"}
        rgbi_set = {"r", "g", "b", "intensity"}
        self.geom_pos = [
            i for i, n in enumerate(self.target_feature_names) if n in geom_set
        ]
        self.rgbi_pos = [
            i for i, n in enumerate(self.target_feature_names) if n in rgbi_set
        ]
        self.rgbi_names = [self.target_feature_names[i] for i in self.rgbi_pos]
        self.rgbi_dim = len(self.rgbi_pos)

        default_mean = []
        default_std = []
        for n in self.rgbi_names:
            if n in {"r", "g", "b"}:
                default_mean.append(0.5)
                default_std.append(0.2)
            else:
                default_mean.append(0.5)
                default_std.append(0.5)
        self.register_buffer(
            "rgbi_pos_tensor", torch.tensor(self.rgbi_pos, dtype=torch.long)
        )
        self.register_buffer("rgbi_mean", torch.tensor(default_mean, dtype=torch.float32))
        self.register_buffer("rgbi_std", torch.tensor(default_std, dtype=torch.float32))

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
            output_dim=len(self.geom_pos),
        )
        self.rgbi_head = RGBIHead(
            cfg=cfg,
            latent_dim=self.encoder.latent_dim,
            rgbi_dim=self.rgbi_dim,
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
        feature_names = self.target_feature_names
        loss_cfg = self.cfg.task.get("loss", {})

        manual = as_plain_dict(loss_cfg.get("feature_weights", None))
        if manual:
            return self._log_and_build_weights(
                manual, feature_names, source=f"manual override ({self._cfg_loss_path()})"
            )

        if AUTO_WEIGHTS_PATH.exists():
            try:
                auto_cfg = OmegaConf.load(AUTO_WEIGHTS_PATH)
            except Exception as exc:
                get_logger("MAE").warning(
                    f"[loss-weights] failed to parse {AUTO_WEIGHTS_PATH}: {exc}; falling back"
                )
            else:
                auto = as_plain_dict(auto_cfg.get("feature_weights", None))
                if auto:
                    return self._log_and_build_weights(
                        auto, feature_names, source=f"auto-calibration ({AUTO_WEIGHTS_PATH})"
                    )

        return self._log_and_build_weights(
            {}, feature_names, source="default (all 1.0)"
        )

    @staticmethod
    def _cfg_loss_path() -> str:
        return "configs/task/mae.yaml :: loss.feature_weights"

    @staticmethod
    def _log_and_build_weights(
        mapping: dict,
        feature_names: list[str],
        source: str,
    ) -> torch.Tensor:
        weights = torch.tensor(
            [float(mapping.get(name, 1.0)) for name in feature_names],
            dtype=torch.float32,
        )
        logger = get_logger("MAE")
        pretty = ", ".join(
            f"{n}={w:.3f}" for n, w in zip(feature_names, weights.tolist())
        )
        logger.info(f"[loss-weights] source: {source}")
        logger.info(f"[loss-weights] values: {pretty}")
        missing = [n for n in feature_names if n not in mapping] if mapping else []
        if mapping and missing:
            logger.warning(
                f"[loss-weights] no entry for {missing} -- defaulted to 1.0"
            )
        return weights

    def per_feature_loss(
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
        diff = (reconstructed_norm - target_norm) ** 2
        diff = diff * valid.to(diff.dtype)

        numer = diff[masked_idx].sum(dim=0)
        denom = valid[masked_idx].to(diff.dtype).sum(dim=0).clamp(min=1.0)
        return numer / denom

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

    def set_intensity_norm(self, mean: float, std: float) -> None:
        if "intensity" not in self.rgbi_names:
            return
        i = self.rgbi_names.index("intensity")
        self.rgbi_mean[i] = float(mean)
        self.rgbi_std[i] = max(float(std), 1e-2)

    def _feature_stats(
        self,
        target_feat: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, std, valid = self._per_sample_stats(target_feat, batch)
        if self.rgbi_pos:
            pos = self.rgbi_pos_tensor
            mean[:, pos] = self.rgbi_mean.to(mean.dtype)
            std[:, pos] = self.rgbi_std.to(std.dtype)
            valid[:, pos] = True
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
        visible_batch_raw = batch[visible_idx]

        # Remap to contiguous 0..K-1.  After block masking some batch IDs may
        # have 0 visible points, leaving gaps (e.g. [0,0,2,2]).  PTv3 builds
        # spconv sparse tensors whose offset arithmetic assumes dense IDs; a
        # gap causes ScatterGather OOB in the pooling backward.
        _, visible_batch = torch.unique(visible_batch_raw, return_inverse=True)

        encoded = self.encoder(visible_feat, visible_coord, visible_batch)

        # MAE_DETACH_ENCODER=1 → skip encoder backward (diagnostic only; encoder
        # won't be updated).  Use to confirm whether the OOB is inside PTv3.
        if os.environ.get("MAE_DETACH_ENCODER", "0") == "1":
            encoded = encoded.detach()

        target_feat = feat[:, self.target_feature_indices]
        visible_raw_rgbi = target_feat[visible_idx][:, self.rgbi_pos_tensor]

        geom_norm = self.decoder(
            encoded=encoded,
            visible_indices=visible_idx,
            masked_indices=masked_idx,
            coord=coord,
            batch=batch,
        )
        rgbi_raw = self.rgbi_head(
            encoded=encoded,
            visible_indices=visible_idx,
            masked_indices=masked_idx,
            coord=coord,
            batch=batch,
            visible_raw_rgbi=visible_raw_rgbi,
        )

        mean, std, valid = self._feature_stats(target_feat, batch)

        n_total = feat.shape[0]
        n_target = len(self.target_feature_names)
        reconstructed_norm = torch.zeros(
            n_total, n_target, device=encoded.device, dtype=encoded.dtype
        )

        stitched = torch.zeros(
            masked_idx.shape[0], n_target, device=encoded.device, dtype=encoded.dtype
        )
        geom_pos = torch.tensor(self.geom_pos, device=encoded.device, dtype=torch.long)
        rgbi_pos = self.rgbi_pos_tensor.to(encoded.device)
        stitched[:, geom_pos] = geom_norm.to(stitched.dtype)
        rgbi_norm = (rgbi_raw - self.rgbi_mean.to(rgbi_raw.dtype)) / self.rgbi_std.to(
            rgbi_raw.dtype
        )
        stitched[:, rgbi_pos] = rgbi_norm.to(stitched.dtype)
        reconstructed_norm[masked_idx] = stitched

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
