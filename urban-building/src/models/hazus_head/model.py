# src/models/hazus_head/model.py
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.models.hazus_head.codebook import NUM_HAZUS_CLASSES, HazusCodebook
from src.models.hazus_head.features import HazusFeatureExtractor
from src.models.hazus_head.rules import FEMAConstraints


class HazusHead(nn.Module):
    def __init__(
        self,
        in_features: int = 7,
        hidden_dim: int = 128,
        num_classes: int = NUM_HAZUS_CLASSES,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(features)


class HazusModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_extractor = HazusFeatureExtractor()
        self.codebook = HazusCodebook()
        self.constraints = FEMAConstraints()

        self.head = HazusHead(
            in_features=7,
            hidden_dim=128,
            num_classes=self.codebook.num_classes,
        )

    def forward(
        self,
        xyz: Tensor,
        batch: Tensor,
        mae_errors: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        features = self.feature_extractor.extract_batch(xyz, batch, mae_errors)

        logits = self.head(features)
        predictions = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "predictions": predictions,
            "features": features,
        }

    def predict_single(
        self,
        xyz: Tensor,
        mae_error: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        features = self.feature_extractor.extract(xyz, mae_error)
        features = features.unsqueeze(0)

        logits = self.head(features)
        prediction = torch.argmax(logits, dim=-1)

        return {
            "logits": logits.squeeze(0),
            "prediction": prediction.squeeze(0),
            "features": features.squeeze(0),
            "decoded": self.codebook.decode(prediction)[0],
        }
