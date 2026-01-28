import torch.nn as nn

from .decoder import MAEDecoder
from .encoder import MAEEncoder


class MAEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = MAEEncoder(cfg)
        self.decoder = MAEDecoder(cfg.model.latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        pred_xyz = self.decoder(z)
        return (pred_xyz - x[..., :3] ** 2).mean()  # r2
