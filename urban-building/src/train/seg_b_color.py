# src/train/seg_b_color.py
import torch
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader
from src.losses import chamfer_loss, point_mse_loss
from src.models.seg_heads import SegBColorModel, StructuredMasking
from src.train._base import build_optimizer, build_scheduler, train_loop


def seg_b_color_criterion(model, batch, device):
    feat = batch["points"].to(device)
    coord = batch["coords"].to(device)
    batch_idx = batch["batch"].to(device)
    rgb = batch["rgb"].to(device) if "rgb" in batch else None

    masking = StructuredMasking(targets=["wall", "roof"])
    visible_idx, masked_idx = masking(coord)

    visible_feat = feat[visible_idx]
    visible_coord = coord[visible_idx]
    visible_batch = batch_idx[visible_idx]

    output = model(visible_feat, visible_coord, visible_batch)
    xyz_pred = output["xyz_pred"]
    rgb_pred = output["rgb_pred"]

    target_xyz = coord[masked_idx]
    geom_loss = chamfer_loss(xyz_pred, target_xyz)

    if rgb is not None:
        target_rgb = rgb[masked_idx]
        n_pred = rgb_pred.shape[0]
        n_masked = target_rgb.shape[0]
        if n_pred > n_masked:
            color_loss = point_mse_loss(rgb_pred[:n_masked], target_rgb)
        else:
            color_loss = point_mse_loss(rgb_pred, target_rgb[:n_pred])
    else:
        color_loss = torch.tensor(0.0, device=device)

    total_loss = model.geom_weight * geom_loss + model.color_weight * color_loss
    return total_loss


def train_seg_b_color(cfg: DictConfig) -> None:
    logger = get_logger("SEGC")
    logger.info("Starting Seg-B color training")

    set_seed(cfg.run.seed)

    model = SegBColorModel(cfg)
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
        criterion=seg_b_color_criterion,
        logger=logger,
    )

    logger.info("Seg-B color training complete")


train = train_seg_b_color
