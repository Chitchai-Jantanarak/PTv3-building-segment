# src/core/utils/checkpoint.py
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_ckpt(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: Union[str, Path],
    best: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if extra:
        state.update(extra)

    filepath = path / "best.pt" if best else path / f"checkpoint_ep{epoch:03d}.pt"

    torch.save(state, filepath)
    return str(filepath)


def load_ckpt(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    strict: bool = True,
    device: Optional[str] = None,
) -> dict[str, Any]:
    path = Path(path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=strict)

    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state


def get_latest_ckpt(path: Union[str, Path]) -> Optional[Path]:
    path = Path(path)
    if not path.exists():
        return None

    ckpts = list(path.glob("checkpoint_ep*.pt"))
    if not ckpts:
        return None

    return max(ckpts, key=lambda x: int(x.stem.split("ep")[1]))
