# src/models/hazus_head/rules.py
from typing import Dict, List, Optional

import torch
from torch import Tensor


class FEMAConstraints:
    def __init__(self):
        self.height_limits = {
            "W1": (0, 10),
            "W2": (0, 15),
            "S1L": (0, 20),
            "S1M": (20, 50),
            "S1H": (50, 150),
            "C1L": (0, 20),
            "C1M": (20, 50),
            "C1H": (50, 150),
            "RM1L": (0, 20),
            "RM1M": (20, 50),
        }

        self.area_limits = {
            "W1": (0, 500),
            "W2": (0, 1000),
            "S1L": (0, 2000),
            "S1M": (500, 5000),
            "S1H": (1000, 10000),
            "C1L": (0, 2000),
            "C1M": (500, 5000),
            "C1H": (1000, 10000),
            "RM1L": (0, 1500),
            "RM1M": (500, 3000),
        }

    def validate(
        self,
        building_type: str,
        height: float,
        area: float,
    ) -> bool:
        if building_type not in self.height_limits:
            return False

        h_min, h_max = self.height_limits[building_type]
        a_min, a_max = self.area_limits[building_type]

        height_valid = h_min <= height <= h_max
        area_valid = a_min <= area <= a_max

        return height_valid and area_valid

    def get_valid_types(
        self,
        height: float,
        area: float,
    ) -> List[str]:
        valid = []

        for btype in self.height_limits.keys():
            if self.validate(btype, height, area):
                valid.append(btype)

        return valid


def apply_height_rules(
    features: Tensor,
    height_idx: int = 0,
) -> Tensor:
    height = features[:, height_idx]

    low_rise = (height >= 0) & (height < 20)
    mid_rise = (height >= 20) & (height < 50)
    high_rise = height >= 50

    height_class = torch.zeros(height.shape[0], dtype=torch.long, device=height.device)
    height_class[mid_rise] = 1
    height_class[high_rise] = 2

    return height_class


def compute_confidence(
    logits: Tensor,
    constraints_satisfied: Tensor,
) -> Tensor:
    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1)[0]

    confidence = max_prob * constraints_satisfied.float()

    return confidence
