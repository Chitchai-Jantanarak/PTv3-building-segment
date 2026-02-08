# src/infer/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.core.utils import get_logger, load_ckpt


class InferenceEngine(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device or cfg.run.device)
        self.logger = get_logger("INF")

        self.model = self._build_model()
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

    @abstractmethod
    def _build_model(self) -> nn.Module:
        pass

    def _load_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        load_ckpt(path, self.model, device=str(self.device))
        self.logger.info(f"Loaded checkpoint: {path}")

    @abstractmethod
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        pass

    @torch.no_grad()
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.predict(data)


def prepare_input(
    data: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result = {}

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif hasattr(value, "__array__"):
            result[key] = torch.from_numpy(value).to(device)
        else:
            result[key] = value

    return result
