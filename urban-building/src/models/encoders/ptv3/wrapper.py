import torch.nn as nn
from PointTransformerV3 import PointTransformerV3


class PTv3Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = PointTransformerV3(
            in_channels=len(cfg.model.input_channels), embed_dim=cfg.model.latent_dim
        )
        self.latent_dim = cfg.model.latent_dim

    def forward(self, x):
        return self.net(x)
