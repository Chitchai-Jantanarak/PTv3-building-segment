# src/core/utils/checkpoint.py
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_ckpt(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: str | Path,
    best: bool = False,
    extra: dict[str, Any] | None = None,
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


from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


def load_ckpt(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    strict: bool = True,
    device: str | None = None,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    path = Path(path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(path, map_location=device)

    ckpt_sd = state["model_state_dict"]
    model_sd = model.state_dict()

    # strict=True keeps old behavior
    if strict:
        model.load_state_dict(ckpt_sd, strict=True)
    else:
        filtered_sd = {}
        skipped = []

        for key, value in ckpt_sd.items():
            if key not in model_sd:
                skipped.append(
                    f"{key}: not in current model"
                )
                continue

            if model_sd[key].shape != value.shape:
                skipped.append(
                    f"{key}: ckpt={tuple(value.shape)} "
                    f"model={tuple(model_sd[key].shape)}"
                )
                continue

            filtered_sd[key] = value

        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)

        print(f"Loaded {len(filtered_sd)}/{len(ckpt_sd)} checkpoint tensors")

        if skipped:
            print("Skipped incompatible checkpoint tensors:")
            for item in skipped:
                print(f"  {item}")

        if missing:
            print(f"Missing keys after partial load: {len(missing)}")
            for key in missing:
                print(f"  {key}")

        if unexpected:
            print(f"Unexpected keys after partial load: {len(unexpected)}")
            for key in unexpected:
                print(f"  {key}")

    # Important: avoid loading optimizer after architecture changed
    if (
        optimizer is not None
        and load_optimizer
        and strict
        and "optimizer_state_dict" in state
    ):
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state


def load_pretrained_encoder(
    model: nn.Module,
    mae_ckpt_path: str | Path,
    device: str | None = None,
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
            new_key = key[len("encoder.") :]  # strip first "encoder."
            encoder_state[new_key] = value

    if not encoder_state:
        raise ValueError(
            f"No encoder weights found in MAE checkpoint. "
            f"Keys start with: {list(mae_state.keys())[:5]}"
        )

    missing, unexpected = model.load_state_dict(encoder_state, strict=False)

    # Validate: encoder keys we extracted should have been accepted by the model.
    # `unexpected` = keys we supplied that the model doesn't have.
    n_loaded = len(encoder_state) - len(unexpected)
    if unexpected:
        import logging

        logging.getLogger("checkpoint").warning(
            f"Encoder loading: {len(unexpected)} unexpected keys "
            f"(not in model): {unexpected[:5]}"
        )
    if missing:
        import logging

        # missing keys are model params NOT in the checkpoint -- expected for
        # the seg head, but encoder.* misses indicate a real problem.
        encoder_missing = [k for k in missing if k.startswith("encoder.")]
        if encoder_missing:
            logging.getLogger("checkpoint").warning(
                f"Encoder loading: {len(encoder_missing)} encoder keys "
                f"MISSING from checkpoint (weights are random): "
                f"{encoder_missing[:5]}"
            )

    return n_loaded


def get_latest_ckpt(path: str | Path) -> Path | None:
    path = Path(path)
    if not path.exists():
        return None

    ckpts = list(path.glob("checkpoint_ep*.pt"))
    if not ckpts:
        return None

    return max(ckpts, key=lambda x: int(x.stem.split("ep")[1]))
