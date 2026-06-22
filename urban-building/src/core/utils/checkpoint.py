# src/core/utils/checkpoint.py
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


def encoder_fingerprint(cfg: Any) -> dict[str, Any]:
    m = cfg.model
    return {
        "in_channels": int(m.in_channels),
        "enc_channels": list(m.enc_channels),
        "dec_channels": list(m.dec_channels),
        "enc_depths": list(m.enc_depths),
        "dec_depths": list(m.dec_depths),
        "patch_size": int(m.patch_size),
        "grid_size": float(m.grid_size),
        "intensity_channel": bool(m.get("intensity_channel", False)),
    }


def save_ckpt(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: str | Path,
    best: bool = False,
    extra: dict[str, Any] | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if fingerprint is not None:
        state["encoder_fingerprint"] = fingerprint

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


def _extract_encoder_state(mae_state: dict[str, Any]) -> dict[str, Any]:
    encoder_state = {}
    for key, value in mae_state.items():
        if key.startswith("encoder.encoder."):
            encoder_state[key[len("encoder.") :]] = value
    return encoder_state


def _bridge_report(
    model: nn.Module, encoder_state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_sd = model.state_dict()
    filtered: dict[str, Any] = {}
    unexpected: list[str] = []
    shape_mismatches: list[tuple[str, tuple, tuple]] = []

    for key, value in encoder_state.items():
        if key not in model_sd:
            unexpected.append(key)
            continue
        if tuple(model_sd[key].shape) != tuple(value.shape):
            shape_mismatches.append(
                (key, tuple(value.shape), tuple(model_sd[key].shape))
            )
            continue
        filtered[key] = value

    model_encoder_keys = [k for k in model_sd if k.startswith("encoder.")]
    encoder_missing = [k for k in model_encoder_keys if k not in filtered]

    report: dict[str, Any] = {
        "n_offered": len(encoder_state),
        "n_loaded": len(filtered),
        "n_model_encoder_keys": len(model_encoder_keys),
        "encoder_missing": encoder_missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    report["ok"] = (
        report["n_loaded"] > 0
        and not encoder_missing
        and not unexpected
        and not shape_mismatches
    )
    return report, filtered


def _format_bridge_failure(report: dict[str, Any]) -> str:
    lines = [
        "MAE encoder bridge verification FAILED:",
        f"  offered={report['n_offered']} loaded={report['n_loaded']} "
        f"model_encoder_keys={report['n_model_encoder_keys']}",
    ]
    if report["shape_mismatches"]:
        lines.append(f"  shape mismatches ({len(report['shape_mismatches'])}):")
        for k, ck, mk in report["shape_mismatches"][:10]:
            lines.append(f"    {k}: ckpt={ck} model={mk}")
    if report["unexpected"]:
        lines.append(
            f"  unexpected ckpt keys not in model ({len(report['unexpected'])}): "
            f"{report['unexpected'][:10]}"
        )
    if report["encoder_missing"]:
        lines.append(
            f"  model encoder keys NOT loaded -- random weights "
            f"({len(report['encoder_missing'])}): {report['encoder_missing'][:10]}"
        )
    lines.append(
        "  -> cfg.model likely drifted from the MAE training config. "
        "Reconcile configs/model/ptv3.yaml (and data feature flags) and re-verify."
    )
    return "\n".join(lines)


def encoder_bridge_report(
    model: nn.Module,
    mae_ckpt_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    mae_ckpt_path = Path(mae_ckpt_path)
    if not mae_ckpt_path.exists():
        raise FileNotFoundError(f"MAE checkpoint not found: {mae_ckpt_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(mae_ckpt_path, map_location=device)
    encoder_state = _extract_encoder_state(state["model_state_dict"])
    if not encoder_state:
        raise ValueError(
            f"No encoder weights (encoder.encoder.*) found in MAE checkpoint. "
            f"Keys start with: {list(state['model_state_dict'].keys())[:5]}"
        )

    report, _ = _bridge_report(model, encoder_state)
    report["ckpt_fingerprint"] = state.get("encoder_fingerprint")
    return report


def load_pretrained_encoder(
    model: nn.Module,
    mae_ckpt_path: str | Path,
    device: str | None = None,
    strict_encoder: bool = True,
) -> int:
    mae_ckpt_path = Path(mae_ckpt_path)
    if not mae_ckpt_path.exists():
        raise FileNotFoundError(f"MAE checkpoint not found: {mae_ckpt_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(mae_ckpt_path, map_location=device)
    encoder_state = _extract_encoder_state(state["model_state_dict"])

    if not encoder_state:
        raise ValueError(
            f"No encoder weights found in MAE checkpoint. "
            f"Keys start with: {list(state['model_state_dict'].keys())[:5]}"
        )

    report, filtered = _bridge_report(model, encoder_state)

    if strict_encoder and not report["ok"]:
        raise RuntimeError(_format_bridge_failure(report))

    model.load_state_dict(filtered, strict=False)

    if not strict_encoder and not report["ok"]:
        import logging

        logging.getLogger("checkpoint").warning(_format_bridge_failure(report))

    return report["n_loaded"]


def get_latest_ckpt(path: str | Path) -> Path | None:
    path = Path(path)
    if not path.exists():
        return None

    ckpts = list(path.glob("checkpoint_ep*.pt"))
    if not ckpts:
        return None

    return max(ckpts, key=lambda x: int(x.stem.split("ep")[1]))
