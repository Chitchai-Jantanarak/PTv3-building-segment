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
        n_attn_heads: int = 4
    ):
        super().__init__()

        hidden_dim = latent_dim // 2

        self.register_buffer("coord_scale", torch.tensor(1.0)) # abs coords

        self.pos_embed = nn.Sequential(
            nn.Linear(coord_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_attn_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.attn_norm = nn.LayerNorm(latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def _normalize_coord(self, coord: Tensor) -> Tensor:
        c_min = coord.min(dim=0, keepdim=True).values
        c_max = coord.max(dim=0, keepdim=True).values
        span = (c_max - c_min).clamp(min=1.0)
        return (coord - c_min) / span # -> [0, 1]

    def forward(
        self,
        encoded: Tensor,
        visible_indices: Tensor,
        masked_indices: Tensor,
        n_total: int,
        coord: Tensor | None = None,
    ) -> Tensor:
        n_vis = visible_indices.shape[0]
        n_msk = masked_indices.shape[0]

        coord_norm = self._normalize_coord(coord)

        pos_all = self.pos_embed(coord_norm)                  
        pos_vis = pos_all[visible_indices]                    
        pos_msk = pos_all[masked_indices]                     

        kv = encoded + pos_vis                                

        q = self.mask_token.expand(n_msk, -1) + pos_msk      

        kv_seq = kv.unsqueeze(0)        
        q_seq  = q.unsqueeze(0)         

        attn_out, _ = self.cross_attn(q_seq, kv_seq, kv_seq)
        attn_out = attn_out.squeeze(0)                        

        masked_features = self.attn_norm(q + attn_out)        

        full_features = torch.zeros(n_total, masked_features.shape[-1],
                                    device=encoded.device, dtype=encoded.dtype)
        full_features[masked_indices] = masked_features

        reconstructed = torch.zeros(n_total, self.decoder[-1].out_features,
                                    device=encoded.device, dtype=encoded.dtype)
        reconstructed[masked_indices] = self.decoder(masked_features)

        vis_feat = encoded + pos_vis
        reconstructed[visible_indices] = self.decoder(vis_feat)
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
