from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.tools.knn import knn_neighbors


class RGBIHead(nn.Module):
    def __init__(self, cfg: DictConfig, latent_dim: int, rgbi_dim: int):
        super().__init__()
        self.rgbi_dim = rgbi_dim
        dec_cfg = cfg.task.get("color_decoder", {})
        self.enabled = bool(dec_cfg.get("enabled", True))
        self.k = int(dec_cfg.get("k", 8))
        self.block_size = float(dec_cfg.get("block_size", 0.05))
        self.chunk_size = int(dec_cfg.get("chunk_size", 8192))

        in_dim = latent_dim + rgbi_dim + 3
        hidden = max(latent_dim // 2, 32)
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, rgbi_dim)
        self.log_tau = nn.Parameter(torch.zeros(1))

    def _normalize_coord(self, coord: Tensor, batch: Tensor) -> Tensor:
        out = torch.empty_like(coord)
        for b in batch.unique():
            m = batch == b
            sub = coord[m]
            c_min = sub.min(dim=0, keepdim=True).values
            c_max = sub.max(dim=0, keepdim=True).values
            span = (c_max - c_min).clamp(min=1e-6)
            out[m] = (sub - c_min) / span
        return out

    def forward(
        self,
        encoded: Tensor,
        visible_indices: Tensor,
        masked_indices: Tensor,
        coord: Tensor,
        batch: Tensor,
        visible_raw_rgbi: Tensor,
    ) -> Tensor:
        n_masked = masked_indices.shape[0]
        device = encoded.device
        if n_masked == 0:
            return torch.zeros(0, self.rgbi_dim, device=device, dtype=encoded.dtype)

        if not self.enabled or visible_indices.numel() == 0:
            hidden = self.out.in_features
            zeros = torch.zeros(n_masked, hidden, device=device, dtype=self.out.weight.dtype)
            return self.out(zeros).to(encoded.dtype)

        coord_norm = self._normalize_coord(coord.float(), batch)
        q_coord = coord_norm[masked_indices]
        r_coord = coord_norm[visible_indices]
        q_batch = batch[masked_indices]
        r_batch = batch[visible_indices]

        idx, dist = knn_neighbors(
            q_coord, r_coord, q_batch, r_batch,
            k=self.k, block_size=self.block_size, chunk_size=self.chunk_size,
        )

        ref_feat = torch.cat([encoded, visible_raw_rgbi.to(encoded.dtype)], dim=1)
        nbr_feat = ref_feat[idx]
        nbr_coord = r_coord[idx].to(encoded.dtype)
        rel = nbr_coord - q_coord.unsqueeze(1).to(encoded.dtype)

        h = self.neighbor_mlp(torch.cat([nbr_feat, rel], dim=-1))

        finite = torch.isfinite(dist)
        tau = self.log_tau.exp().clamp(min=1e-4)
        dist_safe = dist.masked_fill(~finite, 0.0)
        scores = (-dist_safe / tau).masked_fill(~finite, float("-inf"))
        w = torch.softmax(scores, dim=1)
        w = torch.nan_to_num(w, nan=0.0).unsqueeze(-1).to(h.dtype)

        agg = (h * w).sum(dim=1)
        return self.out(agg)
