import torch.nn as nn
from src.models.encoders.ptv3 import PTV3Encoder


class MAEEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PTV3Encoder(cfg)

    def forward(self, x):
        return self.encoder(x)
