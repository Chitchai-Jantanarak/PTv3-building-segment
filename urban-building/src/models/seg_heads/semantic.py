import torch.nn as nn
from src.models.encoders.ptv3 import PTv3Encoder


class SegmentationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PTv3Encoder(cfg)
        self.classifier = nn.Linear(cfg.model.latent_dim, cfg.dataset.num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)
