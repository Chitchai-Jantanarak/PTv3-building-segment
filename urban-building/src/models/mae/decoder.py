import torch.nn as nn


class MAEDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 3)
        )

    def forward(self, z):
        return self.mlp(z)
