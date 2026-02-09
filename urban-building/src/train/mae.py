# src/train/mae.py
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

    model = train_loop(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=mae_criterion,
        logger=logger,
    )

    logger.info("MAE pretraining complete")


train = train_mae
