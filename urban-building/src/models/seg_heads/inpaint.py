import torch.nn as nn
from src.models.encoders.ptv3 import PTv3Encoder


class GeomInpaintHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PTv3Encoder(cfg)
        self.decoder = nn.Linear(cfg.model.latent_dim, 3)  # rgb

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class GreyInpaintHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PTv3Encoder(cfg)
        self.decoder = nn.Linear(cfg.model.latent_dim, 1)  # grayscale

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
