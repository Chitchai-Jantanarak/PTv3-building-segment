# src/train/hazus.py
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader
from src.models.hazus_head import HazusModel
from src.train._base import build_optimizer, build_scheduler, train_loop


def hazus_criterion(model, batch, device):
    xyz = batch["coords"].to(device)
    batch_idx = batch["batch"].to(device)
    labels = batch["labels"].to(device)
    mae_errors = batch.get("mae_errors")
    if mae_errors is not None:
        mae_errors = mae_errors.to(device)

    output = model(xyz, batch_idx, mae_errors)
    logits = output["logits"]

    loss = F.cross_entropy(logits, labels)
    return loss


def train_hazus(cfg: DictConfig) -> None:
    logger = get_logger("HAZ")
    logger.info("Starting HAZUS training")

    set_seed(cfg.run.seed)

    model = HazusModel(cfg)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        criterion=hazus_criterion,
        logger=logger,
    )

    from src.eval import run_evaluation

    device = torch.device(cfg.run.device)
    out_dir = Path(cfg.paths.ckpt_root) / cfg.task.name
    run_evaluation(
        task="hazus",
        model=result.model,
        val_loader=val_loader,
        device=device,
        out_dir=out_dir,
        train_losses=result.train_losses,
        val_losses=result.val_losses,
        cfg=cfg,
    )

    logger.info("HAZUS training complete")


train = train_hazus
