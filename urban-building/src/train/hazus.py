# src/train/hazus.py
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader, build_dataset
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

    train_dataset = build_dataset(cfg, split="train")
    train_loader = build_dataloader(cfg, train_dataset, shuffle=True)

    val_dataset = build_dataset(cfg, split="val")
    val_loader = (
        build_dataloader(cfg, val_dataset, shuffle=False) if val_dataset else None
    )

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    model = train_loop(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=hazus_criterion,
        logger=logger,
    )

    logger.info("HAZUS training complete")
