# src/models/mae/decoder.py

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

class MAEDecoder(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        latent_dim: int,
        output_dim: int = 4,
        coord_dim: int = 3,
    ):
        super().__init__()

        hidden_dim = latent_dim // 2
        geom_dim = 4
        color_dim = 4
        self.color_dim = color_dim

        self.register_buffer("coord_scale", torch.tensor(1.0))

        self.pos_embed = nn.Sequential(
            nn.Linear(coord_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        n_heads = cfg.task.get("n_heads", 8)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.attn_norm = nn.LayerNorm(latent_dim)

        self.color_in = nn.Linear(color_dim, latent_dim)
        self.color_cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.color_norm = nn.LayerNorm(latent_dim)

        self.geom_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, geom_dim),
        )

        self.color_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, color_dim),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def _normalize_coord(
        self,
        coord: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if batch is None:
            c_min = coord.min(dim=0, keepdim=True).values
            c_max = coord.max(dim=0, keepdim=True).values
            span = (c_max - c_min).clamp(min=1.0)
            return (coord - c_min) / span

        batch_max = int(batch.max().item()) + 1
        out = torch.empty_like(coord)
        for b in range(batch_max):
            mask = batch == b
            if not mask.any():
                continue
            sub = coord[mask]
            c_min = sub.min(dim=0, keepdim=True).values
            c_max = sub.max(dim=0, keepdim=True).values
            span = (c_max - c_min).clamp(min=1.0)
            out[mask] = (sub - c_min) / span
        return out

    def _cross_attend(
        self,
        attn: nn.MultiheadAttention,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        visible_indices: Tensor,
        masked_indices: Tensor,
        batch: Tensor | None,
    ) -> Tensor:
        if batch is None:
            out, _ = attn(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), need_weights=False
            )
            return out.squeeze(0)

        out = torch.zeros_like(q)
        vis_batch = batch[visible_indices]
        msk_batch = batch[masked_indices]
        batch_max = int(batch.max().item()) + 1
        for b in range(batch_max):
            vis_mask_b = vis_batch == b
            msk_mask_b = msk_batch == b
            if not msk_mask_b.any() or not vis_mask_b.any():
                continue  # leave zeros for masked points with no visible context
            q_b = q[msk_mask_b].unsqueeze(0)
            k_b = k[vis_mask_b].unsqueeze(0)
            v_b = v[vis_mask_b].unsqueeze(0)
            out_b, _ = attn(q_b, k_b, v_b, need_weights=False)
            out[msk_mask_b] = out_b.squeeze(0).to(out.dtype)
        return out

    def forward(
        self,
        encoded: Tensor,
        visible_indices: Tensor,
        masked_indices: Tensor,
        n_total: int,
        coord: Tensor | None = None,
        batch: Tensor | None = None,
        visible_raw_feat: Tensor | None = None,
    ) -> Tensor:
        n_msk = masked_indices.shape[0]

        coord_norm = self._normalize_coord(coord, batch)
        pos_all = self.pos_embed(coord_norm)
        pos_vis = pos_all[visible_indices]
        pos_msk = pos_all[masked_indices]

        q = self.mask_token.expand(n_msk, -1) + pos_msk

        # --- Geometry path: keys = values = encoded features + position ---
        kv_geom = encoded + pos_vis
        geom_attn = self._cross_attend(
            self.cross_attn, q, kv_geom, kv_geom,
            visible_indices, masked_indices, batch,
        )
        geom_features = self.attn_norm(q + geom_attn)

        # --- Color path: scores on geometry/position, values carry color ---
        # The query asks "which visible points are relevant to me?" (via keys)
        # and retrieves "a learned blend of their colors" (via values). This is
        # the path the old decoder lacked -- color was never in the values.
        if visible_raw_feat is not None:
            color_values = self.color_in(visible_raw_feat) + pos_vis
        else:
            # Encode-only / no raw color available: fall back to encoded values.
            color_values = kv_geom
        color_keys = encoded + pos_vis
        color_attn = self._cross_attend(
            self.color_cross_attn, q, color_keys, color_values,
            visible_indices, masked_indices, batch,
        )
        color_features = self.color_norm(q + color_attn)

        reconstructed = torch.zeros(
            n_total, 8, device=encoded.device, dtype=encoded.dtype
        )

        geom_msk = self.geom_head(geom_features)
        reconstructed[masked_indices, :4] = geom_msk.to(encoded.dtype)

        color_msk_pred = self.color_head(color_features)
        reconstructed[masked_indices, 4:] = color_msk_pred.to(encoded.dtype)

        return reconstructed


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        latent_dim: int,
        output_dim: int = 4,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        self.output_proj = nn.Linear(latent_dim, output_dim)

    def forward(
        self,
        encoded: Tensor,
        visible_indices: Tensor,
        masked_indices: Tensor,
        n_total: int,
    ) -> Tensor:
        n_encoded = encoded.shape[0]
        n_visible = visible_indices.shape[0]

        full_features = self.mask_token.expand(n_total, -1).clone()

        if n_encoded == n_visible:
            full_features[visible_indices] = encoded
        elif n_encoded < n_visible:
            full_features[visible_indices[:n_encoded]] = encoded
        else:
            full_features[visible_indices] = encoded[:n_visible]

        full_features = full_features.unsqueeze(0)
        decoded = self.transformer(full_features, full_features)
        decoded = decoded.squeeze(0)

        reconstructed = self.output_proj(decoded)
        return reconstructed
