# src/eval/evaluate.py
"""Post-training evaluation: run inference on val set, compute metrics, plot."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.core.utils import get_logger
from src.eval.metrics import (
    boundary_iou,
    chamfer_stats,
    confusion_matrix,
    height_wise_error,
    per_class_iou,
    per_feature_mse,
    spatial_error_grid,
)
from src.eval.plots import plot_all

logger = get_logger("EVAL")


def _collect_seg_a_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, np.ndarray]:
    """Run Seg-A model on dataloader, collect predictions + labels + coords."""
    model.eval()
    all_preds = []
    all_labels = []
    all_coords = []

    with torch.no_grad():
        for batch in dataloader:
            feat = batch["points"].to(device)
            coord = batch["coords"].to(device)
            batch_idx = batch["batch"].to(device)

            output = model(feat, coord, batch_idx)
            preds = torch.argmax(output["logits"], dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_coords.append(coord.cpu().numpy())

            if "labels" in batch and batch["labels"] is not None:
                all_labels.append(batch["labels"].numpy())

    result = {
        "preds": np.concatenate(all_preds),
        "coords": np.concatenate(all_coords),
    }
    if all_labels:
        result["labels"] = np.concatenate(all_labels)
    return result


def _collect_seg_b_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run Seg-B model on dataloader, collect predictions + targets."""
    model.eval()
    all_pred_xyz = []
    all_target_xyz = []

    with torch.no_grad():
        for batch in dataloader:
            feat = batch["points"].to(device)
            coord = batch["coords"].to(device)
            batch_idx = batch["batch"].to(device)

            output = model(feat, coord, batch_idx)

            if "xyz_pred" in output:
                all_pred_xyz.append(output["xyz_pred"].cpu().numpy())
            if "target_coords" in batch:
                all_target_xyz.append(batch["target_coords"].numpy())
            elif "coords" in batch:
                all_target_xyz.append(batch["coords"].numpy())

    result = {}
    if all_pred_xyz:
        result["pred_xyz"] = np.concatenate(all_pred_xyz)
    if all_target_xyz:
        result["target_xyz"] = np.concatenate(all_target_xyz)
    return result


def _collect_mae_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run MAE model on dataloader, collect reconstructed + original features."""
    model.eval()
    all_recon = []
    all_target = []

    with torch.no_grad():
        for batch in dataloader:
            feat = batch["points"].to(device)
            coord = batch["coords"].to(device)
            batch_idx = batch["batch"].to(device)

            output = model(feat, coord, batch_idx)

            masked_idx = output["masked_indices"]
            recon = output["reconstructed"][masked_idx].cpu().numpy()
            target = feat[masked_idx, :4].cpu().numpy()

            all_recon.append(recon)
            all_target.append(target)

    return {
        "reconstructed": np.concatenate(all_recon),
        "target": np.concatenate(all_target),
    }


def evaluate_seg_a(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    out_dir: Path,
    compute_boundary: bool = True,
) -> dict:
    """Full Seg-A evaluation: confusion matrix, IoU, boundary IoU, plots."""
    logger.info("Evaluating Seg-A on validation set...")
    data = _collect_seg_a_predictions(model, val_loader, device, num_classes)

    if "labels" not in data:
        logger.warning("No labels in validation data — skipping Seg-A evaluation")
        return {}

    preds = data["preds"]
    labels = data["labels"]
    coords = data["coords"]

    cm = confusion_matrix(preds, labels, num_classes)
    iou = per_class_iou(cm)

    metrics = {
        "confusion_matrix": cm,
        "per_class_iou": iou,
        "mean_iou": float(iou.mean()),
    }

    logger.info(f"mIoU: {iou.mean():.4f}")
    for i, name in enumerate(class_names):
        logger.info(f"  {name:20s}: IoU={iou[i]:.4f}")

    if compute_boundary and len(coords) < 500_000:
        logger.info("Computing boundary IoU (may take a moment)...")
        biou = boundary_iou(preds, labels, coords, num_classes)
        metrics["boundary_iou"] = biou
        logger.info(f"Mean Boundary IoU: {biou.mean():.4f}")

    return metrics


def evaluate_seg_b(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    grid_res: float = 1.0,
) -> dict:
    """Full Seg-B evaluation: chamfer stats, height error, spatial heatmap."""
    logger.info("Evaluating Seg-B on validation set...")
    data = _collect_seg_b_predictions(model, val_loader, device)

    if "pred_xyz" not in data or "target_xyz" not in data:
        logger.warning("Missing predictions or targets — skipping Seg-B evaluation")
        return {}

    pred = data["pred_xyz"]
    target = data["target_xyz"]

    # Align lengths (in case of batch size mismatch)
    n = min(len(pred), len(target))
    pred, target = pred[:n], target[:n]

    ch = chamfer_stats(pred, target)
    he = height_wise_error(pred, target)
    eg = spatial_error_grid(pred, target, grid_res=grid_res)

    logger.info(f"Chamfer: mean={ch['mean']:.4f} median={ch['median']:.4f} "
                f"p90={ch['p90']:.4f} p99={ch['p99']:.4f}")

    return {
        "chamfer": ch,
        "height_error": he,
        "error_grid": eg,
    }


def evaluate_mae(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """MAE evaluation: per-feature MSE breakdown."""
    logger.info("Evaluating MAE on validation set...")
    data = _collect_mae_predictions(model, val_loader, device)

    feature_names = ["x", "y", "z", "rel_z"]
    fm = per_feature_mse(data["reconstructed"], data["target"], feature_names)

    logger.info("Per-feature MSE:")
    for name, val in fm.items():
        logger.info(f"  {name:10s}: {val:.6f}")

    return {"feature_mse": fm}


def run_evaluation(
    task: str,
    model: torch.nn.Module,
    val_loader: DataLoader | None,
    device: torch.device,
    out_dir: Path,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    cfg: DictConfig | None = None,
) -> dict:
    """Run full evaluation for any task and generate plots.

    Called at the end of training. Returns metrics dict.
    """
    if val_loader is None:
        logger.warning("No validation loader — skipping evaluation")
        return {}

    out_dir = Path(out_dir)
    metrics = {}

    if train_losses:
        metrics["train_losses"] = train_losses
    if val_losses:
        metrics["val_losses"] = val_losses

    class_names = None

    if task == "seg_a":
        num_classes = cfg.data.get("num_classes", 13) if cfg else 13
        # Get class names from dataset
        if hasattr(val_loader.dataset, "_class_names") and val_loader.dataset._class_names:
            class_names = [
                val_loader.dataset._class_names.get(i, f"cls_{i}")
                for i in range(num_classes)
            ]
        elif hasattr(val_loader.dataset, "CLASSES"):
            class_names = [
                val_loader.dataset.CLASSES.get(i, f"cls_{i}")
                for i in range(num_classes)
            ]
        else:
            class_names = [f"cls_{i}" for i in range(num_classes)]

        seg_metrics = evaluate_seg_a(
            model, val_loader, device, num_classes, class_names, out_dir,
        )
        metrics.update(seg_metrics)

    elif task in ("seg_b_geom", "seg_b_color"):
        seg_b_metrics = evaluate_seg_b(model, val_loader, device, out_dir)
        metrics.update(seg_b_metrics)

    elif task == "mae":
        mae_metrics = evaluate_mae(model, val_loader, device, out_dir)
        metrics.update(mae_metrics)

    # Generate plots
    plot_dir = out_dir / "plots"
    saved = plot_all(task, metrics, plot_dir, class_names=class_names)

    if saved:
        logger.info(f"Saved {len(saved)} evaluation plots to {plot_dir}/")
        for p in saved:
            logger.info(f"  {p.name}")

    return metrics
