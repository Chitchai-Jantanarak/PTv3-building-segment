# src/train/seg_a.py
import torch
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader, build_dataset
from src.losses import focal_loss
from src.models.seg_heads import SegAModel, generate_pseudo_labels
from src.train._base import build_optimizer, build_scheduler, train_loop


def seg_a_criterion(model, batch, device):
    feat = batch["features"].to(device)
    coord = batch["coords"].to(device)
    batch_idx = batch["batch"].to(device)

    if "labels" in batch and batch["labels"] is not None:
        labels = batch["labels"].to(device)
    else:
        rel_z = feat[:, 3]
        labels = generate_pseudo_labels(coord, rel_z)

    output = model(feat, coord, batch_idx)
    logits = output["logits"]

    loss = focal_loss(logits, labels, gamma=2.0)
    return loss


def train_seg_a(cfg: DictConfig) -> None:
    logger = get_logger("SEGA")
    logger.info("Starting Seg-A training")

    set_seed(cfg.run.seed)

    model = SegAModel(cfg)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Encoder frozen: {cfg.task.freeze_encoder}")

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
        criterion=seg_a_criterion,
        logger=logger,
    )

    logger.info("Seg-A training complete")
