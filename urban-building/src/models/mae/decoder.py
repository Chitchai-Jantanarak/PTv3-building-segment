# src/models/mae/decoder.py

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.tools.knn import block_local_knn


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

        self.color_k = cfg.task.color_decoder.get("k", 8)
        self.color_block_size = cfg.task.color_decoder.get("block_size", 0.05)

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

    def _color_context(
        self,
        encoded: Tensor,
        coord_norm: Tensor,
        query_indices: Tensor,
        ref_indices: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if ref_indices.shape[0] == 0:
            return torch.zeros(
                query_indices.shape[0], 4, device=encoded.device, dtype=encoded.dtype
            )

        if query_indices.shape[0] == 0:
            return torch.zeros(
                query_indices.shape[0], 4, device=encoded.device, dtype=encoded.dtype
            )

        max_coord_idx = coord_norm.shape[0] - 1

        if query_indices.max() > max_coord_idx or ref_indices.max() > max_coord_idx:
            return torch.zeros(
                query_indices.shape[0], 4, device=encoded.device, dtype=encoded.dtype
            )

        query_coord = coord_norm[query_indices]
        ref_coord = coord_norm[ref_indices]
        ref_features = encoded[ref_indices]

        try:
            if (
                torch.isnan(query_coord).any()
                or torch.isinf(query_coord).any()
                or torch.isnan(ref_coord).any()
                or torch.isinf(ref_coord).any()
            ):
                return torch.zeros(
                    query_indices.shape[0],
                    4,
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
        except Exception:
            return torch.zeros(
                query_indices.shape[0], 4, device=encoded.device, dtype=encoded.dtype
            )

        query_coord = torch.clamp(query_coord, 0.0, 1.0)
        ref_coord = torch.clamp(ref_coord, 0.0, 1.0)

        if batch is not None:
            query_batch = torch.clamp(batch[query_indices], min=0)
            ref_batch = torch.clamp(batch[ref_indices], min=0)
        else:
            query_batch = None
            ref_batch = None

        color_features = block_local_knn(
            query_coord=query_coord,
            ref_coord=ref_coord,
            ref_features=ref_features,
            query_batch=query_batch,
            ref_batch=ref_batch,
            k=self.color_k,
            block_size=self.color_block_size,
            chunk_size=2048,
        )

        return self.color_head(color_features)

    def forward(
        self,
        encoded: Tensor,
        visible_indices: Tensor,
        masked_indices: Tensor,
        n_total: int,
        coord: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        n_msk = masked_indices.shape[0]

        coord_norm = self._normalize_coord(coord, batch)

        pos_all = self.pos_embed(coord_norm)
        pos_vis = pos_all[visible_indices]
        pos_msk = pos_all[masked_indices]

        kv = encoded + pos_vis

        q = self.mask_token.expand(n_msk, -1) + pos_msk

        if batch is None:
            kv_seq = kv.unsqueeze(0)
            q_seq = q.unsqueeze(0)

            attn_out, _ = self.cross_attn(q_seq, kv_seq, kv_seq, need_weights=False)
            attn_out = attn_out.squeeze(0)
        else:
            attn_out = torch.zeros_like(q)
            vis_batch = batch[visible_indices]
            msk_batch = batch[masked_indices]

            batch_max = int(batch.max().item()) + 1
            for b in range(batch_max):
                vis_mask_b = vis_batch == b
                msk_mask_b = msk_batch == b

                if not msk_mask_b.any():
                    continue

                q_b = q[msk_mask_b].unsqueeze(0)

                if not vis_mask_b.any():
                    attn_out[msk_mask_b] = torch.zeros_like(
                        q_b.squeeze(0), dtype=attn_out.dtype
                    )
                    continue

                kv_b = kv[vis_mask_b].unsqueeze(0)
                out_b, _ = self.cross_attn(q_b, kv_b, kv_b, need_weights=False)
                attn_out[msk_mask_b] = out_b.squeeze(0).to(attn_out.dtype)

        masked_features = self.attn_norm(q + attn_out)

        reconstructed = torch.zeros(
            n_total, 8, device=encoded.device, dtype=encoded.dtype
        )

        geom_msk = self.geom_head(masked_features)
        reconstructed[masked_indices, :4] = geom_msk.to(encoded.dtype)

        color_msk = self._color_context(
            encoded, coord_norm, masked_indices, visible_indices, batch
        )
        reconstructed[masked_indices, 4:] = color_msk.to(encoded.dtype)

        vis_feat = encoded + pos_vis
        geom_vis = self.geom_head(vis_feat)
        reconstructed[visible_indices, :4] = geom_vis.to(encoded.dtype)

        color_vis = self._color_context(
            encoded, coord_norm, visible_indices, visible_indices, batch
        )
        reconstructed[visible_indices, 4:] = color_vis.to(encoded.dtype)

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
