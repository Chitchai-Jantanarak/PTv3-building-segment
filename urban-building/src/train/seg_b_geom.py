# src/train/seg_b_geom.py
import torch
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader
from src.losses import chamfer_loss
from src.models.seg_heads import SegBGeomModel
from src.train._base import build_optimizer, build_scheduler, train_loop


def seg_b_geom_criterion(model, batch, device):
    if "visible" in batch and "target_coords" in batch:
        visible_feat = batch["visible"].to(device)
        visible_coord = batch["visible_coords"].to(device)
        visible_batch = batch["visible_batch"].to(device)
        target_xyz = batch["target_coords"].to(device)
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

    output = model(visible_feat, visible_coord, visible_batch)
    xyz_pred = output["xyz_pred"]

    loss = chamfer_loss(xyz_pred, target_xyz)
    return loss


def train_seg_b_geom(cfg: DictConfig) -> None:
    logger = get_logger("SEGB")
    logger.info("Starting Seg-B geometry training")

    set_seed(cfg.run.seed)

    model = SegBGeomModel(cfg)
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
        criterion=seg_b_geom_criterion,
        logger=logger,
    )

    logger.info("Seg-B geometry training complete")


train = train_seg_b_geom
