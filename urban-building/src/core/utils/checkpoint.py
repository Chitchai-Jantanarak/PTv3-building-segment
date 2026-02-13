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


def load_pretrained_encoder(
    model: nn.Module,
    mae_ckpt_path: Union[str, Path],
    device: Optional[str] = None,
) -> int:
    mae_ckpt_path = Path(mae_ckpt_path)
    if not mae_ckpt_path.exists():
        raise FileNotFoundError(f"MAE checkpoint not found: {mae_ckpt_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(mae_ckpt_path, map_location=device)
    mae_state = state["model_state_dict"]

    # Extract encoder keys: encoder.encoder.net.* -> encoder.net.*
    encoder_state = {}
    for key, value in mae_state.items():
        if key.startswith("encoder.encoder."):
            new_key = key[len("encoder."):]  # strip first "encoder."
            encoder_state[new_key] = value

    if not encoder_state:
        raise ValueError(
            f"No encoder weights found in MAE checkpoint. "
            f"Keys start with: {list(mae_state.keys())[:5]}"
        )

    missing, unexpected = model.load_state_dict(encoder_state, strict=False)

    return len(encoder_state)


def get_latest_ckpt(path: Union[str, Path]) -> Optional[Path]:
    path = Path(path)
    if not path.exists():
        return None

    ckpts = list(path.glob("checkpoint_ep*.pt"))
    if not ckpts:
        return None

    return max(ckpts, key=lambda x: int(x.stem.split("ep")[1]))
