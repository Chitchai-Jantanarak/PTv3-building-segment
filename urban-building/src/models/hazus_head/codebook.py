# src/models/hazus_head/codebook.py

import torch
from torch import Tensor

HAZUS_BUILDING_TYPES = {
    0: {"code": "W1", "name": "Wood Light Frame", "stories": "1-2"},
    1: {"code": "W2", "name": "Wood Commercial", "stories": "all"},
    2: {"code": "S1L", "name": "Steel Moment Frame Low", "stories": "1-3"},
    3: {"code": "S1M", "name": "Steel Moment Frame Mid", "stories": "4-7"},
    4: {"code": "S1H", "name": "Steel Moment Frame High", "stories": "8+"},
    5: {"code": "S2L", "name": "Steel Braced Frame Low", "stories": "1-3"},
    6: {"code": "S2M", "name": "Steel Braced Frame Mid", "stories": "4-7"},
    7: {"code": "S2H", "name": "Steel Braced Frame High", "stories": "8+"},
    8: {"code": "C1L", "name": "Concrete Moment Frame Low", "stories": "1-3"},
    9: {"code": "C1M", "name": "Concrete Moment Frame Mid", "stories": "4-7"},
    10: {"code": "C1H", "name": "Concrete Moment Frame High", "stories": "8+"},
    11: {"code": "C2L", "name": "Concrete Shear Wall Low", "stories": "1-3"},
    12: {"code": "C2M", "name": "Concrete Shear Wall Mid", "stories": "4-7"},
    13: {"code": "C2H", "name": "Concrete Shear Wall High", "stories": "8+"},
    14: {"code": "RM1L", "name": "Reinforced Masonry Low", "stories": "1-3"},
    15: {"code": "RM1M", "name": "Reinforced Masonry Mid", "stories": "4+"},
    16: {"code": "URM", "name": "Unreinforced Masonry", "stories": "all"},
}

NUM_HAZUS_CLASSES = len(HAZUS_BUILDING_TYPES)


def get_building_code(class_idx: int) -> str:
    if class_idx in HAZUS_BUILDING_TYPES:
        return HAZUS_BUILDING_TYPES[class_idx]["code"]
    return "UNK"


def get_building_name(class_idx: int) -> str:
    if class_idx in HAZUS_BUILDING_TYPES:
        return HAZUS_BUILDING_TYPES[class_idx]["name"]
    return "Unknown"


def decode_predictions(
    predictions: Tensor,
) -> list[dict[str, str]]:
    results = []

    for pred in predictions:
        idx = pred.item()
        results.append(
            {
                "class_idx": idx,
                "code": get_building_code(idx),
                "name": get_building_name(idx),
            }
        )

    return results


class HazusCodebook:
    def __init__(self):
        self.types = HAZUS_BUILDING_TYPES
        self.num_classes = NUM_HAZUS_CLASSES

    def get_class_weights(
        self,
        device: torch.device,
    ) -> Tensor:
        weights = torch.ones(self.num_classes, device=device)
        weights[0] = 0.5
        weights[1] = 0.5
        return weights

    def decode(self, predictions: Tensor) -> list[dict[str, str]]:
        return decode_predictions(predictions)
