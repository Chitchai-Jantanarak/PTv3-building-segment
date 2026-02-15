# src/train/seg_b_color.py
import torch
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader
from src.losses import chamfer_loss, point_mse_loss
from src.models.seg_heads import SegBColorModel
from src.train._base import build_optimizer, build_scheduler, train_loop


def seg_b_color_criterion(model, batch, device):
    rgb = batch["rgb"].to(device) if "rgb" in batch else None

    if "visible" in batch and "target_coords" in batch:
        visible_feat = batch["visible"].to(device)
        visible_coord = batch["visible_coords"].to(device)
        visible_batch = batch["visible_batch"].to(device)
        target_xyz = batch["target_coords"].to(device)
        target_feat = batch["target"].to(device)
    else:
        from src.models.seg_heads import StructuredMasking

        feat = batch["points"].to(device)
        coord = batch["coords"].to(device)
        batch_idx = batch["batch"].to(device)

        masking = StructuredMasking(targets=["wall", "roof"])
        visible_idx, masked_idx = masking(coord)

        if len(visible_idx) == 0 or len(masked_idx) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        visible_feat = feat[visible_idx]
        visible_coord = coord[visible_idx]
        visible_batch = batch_idx[visible_idx]
        target_xyz = coord[masked_idx]
        target_feat = feat[masked_idx]

    output = model(visible_feat, visible_coord, visible_batch)
    xyz_pred = output["xyz_pred"]
    rgb_pred = output["rgb_pred"]

    geom_loss = chamfer_loss(xyz_pred, target_xyz)

    if rgb is not None:
        if "mask" in batch:
            mask = batch["mask"].to(device)
            target_rgb = rgb[mask]
        else:
            target_rgb = rgb

        # Use nearest-neighbor correspondence: for each predicted point,
        # find the closest target point and compare colors
        with torch.no_grad():
            dist = torch.cdist(xyz_pred, target_xyz)  # (N_pred, N_target)
            nn_idx = dist.argmin(dim=1)  # nearest target for each prediction

        matched_target_rgb = target_rgb[nn_idx]
        color_loss = point_mse_loss(rgb_pred, matched_target_rgb)
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
