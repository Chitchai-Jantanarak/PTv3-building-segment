# src/train/seg_b_color.py
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader, build_dataset
from src.losses import chamfer_loss, point_mse_loss
from src.models.seg_heads import SegBColorModel, StructuredMasking
from src.train._base import build_optimizer, build_scheduler, train_loop


def seg_b_color_criterion(model, batch, device):
    feat = batch["features"].to(device)
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
    geom_loss = chamfer_loss(xyz_pred[masked_idx], target_xyz)

    if rgb is not None:
        target_rgb = rgb[masked_idx]
        color_loss = point_mse_loss(rgb_pred[masked_idx], target_rgb)
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
        criterion=seg_b_color_criterion,
        logger=logger,
    )

    logger.info("Seg-B color training complete")
