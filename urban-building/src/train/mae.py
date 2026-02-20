# src/train/mae.py
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader
from src.models.mae import MAEForPretraining
from src.train._base import build_optimizer, build_scheduler, train_loop


def mae_criterion(model, batch, device):
    feat = batch["points"].to(device)
    coord = batch["coords"].to(device)
    batch_idx = batch["batch"].to(device)

    output = model.training_step(feat, coord, batch_idx)
    return output["loss"]


def train_mae(cfg: DictConfig) -> None:
    logger = get_logger("MAE")
    logger.info("Starting MAE pretraining")

    set_seed(cfg.run.seed)

    model = MAEForPretraining(cfg)
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


train = train_mae
