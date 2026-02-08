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
    ):
        super().__init__()

        hidden_dim = latent_dim // 2

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
    ) -> Tensor:

        full_features = self.mask_token.expand(n_total, -1).clone()
        full_features[visible_indices] = encoded

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

        full_features = self.mask_token.expand(n_total, -1).clone()
        full_features[visible_indices] = encoded

        full_features = full_features.unsqueeze(0)
        decoded = self.transformer(full_features, full_features)
        decoded = decoded.squeeze(0)

        reconstructed = self.output_proj(decoded)

        return reconstructed
