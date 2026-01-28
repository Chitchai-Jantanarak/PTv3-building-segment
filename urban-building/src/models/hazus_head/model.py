import torch.nn as nn


class HazusHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.model.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.model.num_classes),
        )

    def forward(self, dlp):
        return self.net(dlp)
