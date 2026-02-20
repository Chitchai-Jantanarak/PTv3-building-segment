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

        # Positional encoding: project 3D coords into latent space
        self.pos_embed = nn.Sequential(
            nn.Linear(coord_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Context projection: compress global context for masked tokens
        self.context_proj = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        encoded: Tensor,
        visible_indices: Tensor,
        masked_indices: Tensor,
        n_total: int,
        coord: Tensor | None = None,
    ) -> Tensor:
        n_encoded = encoded.shape[0]
        n_visible = visible_indices.shape[0]

        # Pool encoder output into global context so masked positions
        # depend on visible features (gives encoder gradient)
        global_ctx = self.context_proj(encoded.mean(dim=0, keepdim=True))

        full_features = self.mask_token.expand(n_total, -1).clone()

        if n_encoded == n_visible:
            full_features[visible_indices] = encoded
        elif n_encoded < n_visible:
            full_features[visible_indices[:n_encoded]] = encoded
        else:
            full_features[visible_indices] = encoded[:n_visible]

        # Add global context (broadcast) — encoder gradient flows here
        full_features = full_features + global_ctx

        # Add coordinate positional encoding so masked tokens know
        # their spatial location (not all predicting the same value)
        if coord is not None:
            full_features = full_features + self.pos_embed(coord)

        reconstructed = self.decoder(full_features)
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
