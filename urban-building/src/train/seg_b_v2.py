# src/train/seg_b_v2.py
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed
from src.datasets import build_dataloader, build_dataset
from src.losses import point_mse_loss
from src.models.seg_heads import AnomalyMasking, SegBv2Model
from src.train._base import build_optimizer, build_scheduler, train_loop


def seg_b_v2_criterion(model, batch, device):
    feat = batch["features"].to(device)
    coord = batch["coords"].to(device)
    batch_idx = batch["batch"].to(device)
    rgb = batch.get("rgb")
    if rgb is not None:
        rgb = rgb.to(device)

    masking = AnomalyMasking(ratio=0.3, mode="hybrid")
    visible_idx, masked_idx = masking(coord, batch_idx)

    if len(visible_idx) == 0 or len(masked_idx) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    visible_feat = feat[visible_idx]
    visible_coord = coord[visible_idx]
    visible_batch = batch_idx[visible_idx]

    output = model(visible_feat, visible_coord, visible_batch)

    xyzr_pred = output["xyzr_pred"]
    rgb_pred = output["rgb_pred"]
    anomaly_score = output["anomaly_score"]

    target_coord = coord[masked_idx]
    target_relz = feat[masked_idx, 3:4]
    target_xyzr = torch.cat([target_coord, target_relz], dim=-1)

    n_visible = len(visible_idx)
    n_masked = len(masked_idx)

    if n_visible > n_masked:
        pred_xyzr = xyzr_pred[:n_masked]
    else:
        pred_xyzr = xyzr_pred
        target_xyzr = target_xyzr[:n_visible]

    geom_loss = point_mse_loss(pred_xyzr, target_xyzr)

    if rgb is not None:
        target_rgb = rgb[masked_idx]
        if n_visible > n_masked:
            pred_rgb = rgb_pred[:n_masked]
        else:
            pred_rgb = rgb_pred
            target_rgb = target_rgb[:n_visible]
        color_loss = point_mse_loss(pred_rgb, target_rgb)
    else:
        color_loss = torch.tensor(0.0, device=device)

    anomaly_target = torch.zeros(n_visible, 1, device=device)
    anomaly_loss = F.binary_cross_entropy(anomaly_score, anomaly_target)

    total_loss = (
        model.geom_weight * geom_loss
        + model.color_weight * color_loss
        + 0.1 * anomaly_loss
    )

    return total_loss


def train_seg_b_v2(cfg: DictConfig) -> None:
    logger = get_logger("SBV2")
    logger.info("Starting Seg-B v2 training")

    set_seed(cfg.run.seed)

    model = SegBv2Model(cfg)
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
        criterion=seg_b_v2_criterion,
        logger=logger,
    )

    logger.info("Seg-B v2 training complete")
