# src/train/mae.py
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.core.utils import as_plain_dict, get_logger, set_seed
from src.core.utils.checkpoint import load_ckpt
from src.datasets import build_dataloader
from src.models.mae import MAEForPretraining
from src.models.mae.model import AUTO_WEIGHTS_PATH
from src.train._base import build_optimizer, build_scheduler, train_loop


def mae_criterion(model, batch, device):
    feat = batch["points"].to(device)
    coord = batch["coords"].to(device)
    batch_idx = batch["batch"].to(device)

    output = model.training_step(feat, coord, batch_idx)
    return output["loss"]


def train_mae(cfg: DictConfig) -> None:
    if cfg.task.get("loss", {}).get("calibrate", False):
        calibrate_mae(cfg)
        return

    logger = get_logger("MAE")
    logger.info("Starting MAE pretraining")

    set_seed(cfg.run.seed)

    model = MAEForPretraining(cfg)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.task.name
    best_ckpt = ckpt_dir / "best.pt"

    if best_ckpt.exists():
        logger.info(f"Resuming weights from: {best_ckpt}")
        load_ckpt(
            path=best_ckpt,
            model=model,
            optimizer=None,
            strict=False,
            device=cfg.run.device,
        )
    else:
        logger.info("No checkpoint found — training from scratch")

    if cfg.task.get("freeze_encoder", False):
        logger.info("Freezing encoder")
        for param in model.encoder.parameters():
            param.requires_grad = False

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    result = train_loop(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=mae_criterion,
        logger=logger,
    )

    # Post-training evaluation
    from src.eval import run_evaluation

    device = torch.device(cfg.run.device)
    out_dir = Path(cfg.paths.ckpt_root) / cfg.task.name
    run_evaluation(
        task="mae",
        model=result.model,
        val_loader=val_loader,
        device=device,
        out_dir=out_dir,
        train_losses=result.train_losses,
        val_losses=result.val_losses,
        cfg=cfg,
    )

    logger.info("MAE pretraining complete")


def calibrate_mae(cfg: DictConfig) -> None:
    logger = get_logger("MAE-CAL")
    logger.info("Calibrating per-feature loss weights")

    set_seed(cfg.run.seed)

    model = MAEForPretraining(cfg)

    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.task.name
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        logger.info(f"Loading encoder weights from: {best_ckpt}")
        load_ckpt(
            path=best_ckpt,
            model=model,
            optimizer=None,
            strict=False,
            device=cfg.run.device,
        )
    else:
        logger.info("No checkpoint -- calibrating against freshly-initialised model")

    device = torch.device(cfg.run.device)
    model = model.to(device)
    model.eval()

    train_loader = build_dataloader(cfg, split="train")

    feature_names = list(model.target_feature_names)
    n_features = len(feature_names)
    accum = torch.zeros(n_features, dtype=torch.float64, device=device)
    count = 0
    target_steps = int(cfg.task.loss.get("calibration_steps", 200))

    max_passes = max(1, target_steps)  # hard guard against infinite loops
    pass_idx = 0
    with torch.no_grad():
        while count < target_steps and pass_idx < max_passes:
            pass_idx += 1
            consumed_this_pass = 0
            for batch in train_loader:
                if count >= target_steps:
                    break
                feat = batch["points"].to(device)
                coord = batch["coords"].to(device)
                batch_t = batch["batch"].to(device)

                target = model.build_target(feat)
                output = model.forward(feat, coord, batch_t)
                per_feat = model.per_feature_loss(output, target).double()
                accum += per_feat
                count += 1
                consumed_this_pass += 1

                if count % 25 == 0 or count == target_steps:
                    logger.info(f"Calibration step {count}/{target_steps}")

            if consumed_this_pass == 0:
                break

    if count == 0:
        raise RuntimeError("Calibration produced 0 steps -- check dataloader")

    logger.info(f"Calibration finished: {count} steps over {pass_idx} loader pass(es)")

    mean_loss = (accum / count).cpu().tolist()
    multipliers = as_plain_dict(cfg.task.loss.get("priority_multipliers", None))

    MIN_MEANINGFUL_LOSS = 1e-3

    inv_raw: list[float | None] = []
    flagged: list[str] = []
    for i, loss in enumerate(mean_loss):
        if loss < MIN_MEANINGFUL_LOSS:
            inv_raw.append(None)
            flagged.append(feature_names[i])
        else:
            inv_raw.append(1.0 / loss)

    valid_inv = [r for r in inv_raw if r is not None]
    if not valid_inv:
        raise RuntimeError(
            "All features fell below MIN_MEANINGFUL_LOSS during calibration -- "
            "check per-sample normalization and the masking ratio"
        )
    inv_mean = sum(valid_inv) / len(valid_inv)
    normalised = [
        (r / inv_mean) if r is not None else 1.0
        for r in inv_raw
    ]
    final = {
        name: round(normalised[i] * float(multipliers.get(name, 1.0)), 4)
        for i, name in enumerate(feature_names)
    }

    if flagged:
        logger.warning(
            f"[calibration] features {flagged} had mean loss < {MIN_MEANINGFUL_LOSS} "
            "(likely a normalization / valid-mask bug) -- assigned neutral weight 1.0. "
            "Fix the underlying normalization before trusting these weights."
        )

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    timestamp = datetime.now().isoformat(timespec="seconds")
    ckpt_label = best_ckpt.name if best_ckpt.exists() else "from-init"
    mult_str = (
        " ".join(f"{k}={v}" for k, v in multipliers.items()) if multipliers else "(none)"
    )

    header_lines = [
        f"# generated {timestamp} -- git {git_sha} -- {count} steps -- ckpt {ckpt_label}",
        f"# priority_multipliers applied: {mult_str}",
        "# per-feature unweighted mean MSE (normalized target space):",
    ]
    for name, loss in zip(feature_names, mean_loss):
        marker = "  <- FLAGGED (loss too low, neutral weight 1.0)" if name in flagged else ""
        header_lines.append(f"#   {name:10s} {loss:.6f}{marker}")
    if flagged:
        header_lines.append(
            f"# flagged features: {flagged} -- fix normalization before trusting their weights"
        )
    header_lines.append(
        "# DO NOT hand-edit -- to override, set loss.feature_weights in mae.yaml"
    )

    body_lines = ["", "feature_weights:"]
    for name in feature_names:
        body_lines.append(f"  {name}: {final[name]}")
    body_lines.append("")  # trailing newline

    AUTO_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_WEIGHTS_PATH.write_text("\n".join(header_lines + body_lines))

    logger.info(f"Wrote calibrated weights to {AUTO_WEIGHTS_PATH}")
    for name in feature_names:
        logger.info(f"  {name:10s} loss={dict(zip(feature_names, mean_loss))[name]:.6f}  weight={final[name]}")
    logger.info("Calibration complete -- commit the generated file and re-run without task.loss.calibrate=true")


train = train_mae
