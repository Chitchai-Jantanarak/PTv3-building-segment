# src/eval/evaluate.py

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.core.utils import get_logger
from src.eval.metrics import (
    boundary_iou,
    chamfer_stats_from_nn,
    color_stats,
    confusion_matrix,
    error_by_value_bins,
    freq_weighted_iou,
    fscore_at_tau,
    height_wise_error,
    mean_accuracy,
    nn_distances,
    normalize_confusion,
    overall_accuracy,
    per_class_iou,
    per_feature_bias,
    per_feature_mse,
    per_feature_r2,
    per_feature_rmse,
    precision_recall_f1,
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
    model.eval()
    all_preds = []
    all_labels = []
    all_coords = []

    with torch.no_grad():
        for batch in dataloader:
            feat = batch["points"].to(device)
            coord = batch["coords"].to(device)
            batch_idx = batch["batch"].to(device)

            rgb = batch.get("rgb")
            if rgb is not None:
                rgb = rgb.to(device)

            output = model(feat, coord, batch_idx, rgb=rgb)
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
    model.eval()
    scenes: list[tuple[np.ndarray, np.ndarray]] = []
    all_pred_rgb = []
    all_target_rgb = []

    with torch.no_grad():
        for batch in dataloader:
            if "visible" in batch and "target_coords" in batch:
                feat = batch["visible"].to(device)
                coord = batch["visible_coords"].to(device)
                batch_idx = batch["visible_batch"].to(device)
                target = batch["target_coords"]
            else:
                feat = batch["points"].to(device)
                coord = batch["coords"].to(device)
                batch_idx = batch["batch"].to(device)
                target = batch.get("target_coords", batch.get("coords"))

            output = model(feat, coord, batch_idx)

            if "xyz_pred" in output and target is not None:
                pred_np = output["xyz_pred"].cpu().numpy()
                target_np = target.cpu().numpy()
                if len(pred_np) > 0 and len(target_np) > 0:
                    scenes.append((pred_np, target_np))

            if (
                "rgb_pred" in output
                and "rgb" in batch
                and "mask" in batch
                and target is not None
            ):
                target_rgb = batch["rgb"][batch["mask"]].to(device)
                xyz_pred = output["xyz_pred"]
                if target_rgb.shape[0] > 0 and xyz_pred.shape[0] > 0:
                    nn_idx = torch.cdist(xyz_pred, target.to(device)).argmin(dim=1)
                    all_pred_rgb.append(output["rgb_pred"].cpu().numpy())
                    all_target_rgb.append(target_rgb[nn_idx].cpu().numpy())

    result: dict[str, object] = {"scenes": scenes}
    if all_pred_rgb:
        result["pred_rgb"] = np.concatenate(all_pred_rgb)
        result["target_rgb"] = np.concatenate(all_target_rgb)
    return result


def _collect_mae_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    all_recon = []
    all_target = []
    sample_payload: dict[str, np.ndarray] | None = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            feat = batch["points"].to(device)
            coord = batch["coords"].to(device)
            batch_idx = batch["batch"].to(device)

            output = model(feat, coord, batch_idx)

            masked_idx = output["masked_indices"]
            recon_full = output["reconstructed"]
            if hasattr(model, "build_target"):
                target_tensor = model.build_target(feat)
            else:
                target_tensor = feat

            recon = recon_full[masked_idx].cpu().numpy()
            target = target_tensor[masked_idx].cpu().numpy()
            all_recon.append(recon)
            all_target.append(target)

            if sample_payload is None:
                first_mask = batch_idx == 0
                sel = first_mask.nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() > 0:
                    sel_set = set(sel.tolist())
                    vis_full = output["visible_indices"]
                    msk_full = output["masked_indices"]
                    enc_full = output.get("encoded")

                    vis_local = torch.tensor(
                        [j for j, g in enumerate(vis_full.tolist()) if g in sel_set],
                        dtype=torch.long, device=vis_full.device,
                    )
                    sel_list = sel.tolist()
                    g_to_local = {g: k for k, g in enumerate(sel_list)}

                    vis_global = vis_full[vis_local].tolist()
                    msk_global = [g for g in msk_full.tolist() if g in sel_set]
                    vis_local_remap = np.array(
                        [g_to_local[g] for g in vis_global], dtype=np.int64
                    )
                    msk_local_remap = np.array(
                        [g_to_local[g] for g in msk_global], dtype=np.int64
                    )

                    sample_payload = {
                        "coord":            coord[sel].cpu().numpy(),
                        "target":           target_tensor[sel].cpu().numpy(),
                        "reconstructed":    recon_full[sel].cpu().numpy(),
                        "visible_indices":  vis_local_remap,
                        "masked_indices":   msk_local_remap,
                    }
                    if enc_full is not None and vis_local.numel() > 0:
                        sample_payload["encoded"] = (
                            enc_full[vis_local].cpu().numpy()
                        )

    result = {
        "reconstructed": np.concatenate(all_recon),
        "target": np.concatenate(all_target),
    }
    if sample_payload is not None:
        result["sample"] = sample_payload
    return result


def evaluate_seg_a(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    out_dir: Path,
    compute_boundary: bool = True,
) -> dict:
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
    prf = precision_recall_f1(cm)
    support = cm.sum(axis=1)

    metrics = {
        "confusion_matrix": cm,
        "confusion_matrix_normalized": normalize_confusion(cm),
        "per_class_iou": iou,
        "mean_iou": float(iou.mean()),
        "overall_accuracy": overall_accuracy(cm),
        "mean_accuracy": mean_accuracy(cm),
        "fw_iou": freq_weighted_iou(cm),
        "per_class_precision": prf["precision"],
        "per_class_recall": prf["recall"],
        "per_class_f1": prf["f1"],
        "macro_precision": prf["macro_precision"],
        "macro_recall": prf["macro_recall"],
        "macro_f1": prf["macro_f1"],
        "support": support,
    }

    logger.info(
        f"mIoU={iou.mean():.4f} OA={metrics['overall_accuracy']:.4f} "
        f"mAcc={metrics['mean_accuracy']:.4f} FW-IoU={metrics['fw_iou']:.4f} "
        f"macro-F1={prf['macro_f1']:.4f}"
    )
    for i, name in enumerate(class_names):
        logger.info(
            f"  {name:20s}: IoU={iou[i]:.4f} P={prf['precision'][i]:.4f} "
            f"R={prf['recall'][i]:.4f} F1={prf['f1'][i]:.4f} n={int(support[i])}"
        )

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
    tau: float = 0.2,
) -> dict:
    logger.info("Evaluating Seg-B on validation set...")
    data = _collect_seg_b_predictions(model, val_loader, device)

    scenes = data.get("scenes", [])
    if not scenes:
        logger.warning("No predictions/targets — skipping Seg-B evaluation")
        return {}

    fwd_all, bwd_all, bwd_coord_all = [], [], []
    prec, rec = [], []
    for pred, target in scenes:
        forward, backward = nn_distances(pred, target)
        fwd_all.append(forward)
        bwd_all.append(backward)
        bwd_coord_all.append(target)
        fs = fscore_at_tau(forward, backward, tau=tau)
        prec.append(fs["precision"])
        rec.append(fs["recall"])

    forward = np.concatenate(fwd_all)
    backward = np.concatenate(bwd_all)
    bwd_coord = np.concatenate(bwd_coord_all)

    ch = chamfer_stats_from_nn(forward, backward)
    he = height_wise_error(backward, bwd_coord)
    eg = spatial_error_grid(backward, bwd_coord, grid_res=grid_res)

    precision = float(np.mean(prec))
    recall = float(np.mean(rec))
    denom = precision + recall
    fscore = {
        "tau": float(tau),
        "precision": precision,
        "recall": recall,
        "f1": float(2 * precision * recall / denom) if denom > 0 else 0.0,
    }

    logger.info(
        f"Chamfer: sym_mean={ch['mean']:.4f} CD={ch['chamfer']:.4f} "
        f"fwd={ch['forward_mean']:.4f} bwd={ch['backward_mean']:.4f} "
        f"hausdorff_p95={ch['hausdorff_p95']:.4f}"
    )
    logger.info(
        f"F-score@{tau}: P={precision:.4f} R={recall:.4f} F1={fscore['f1']:.4f} "
        f"(over {len(scenes)} scenes)"
    )

    result = {
        "chamfer": ch,
        "fscore": fscore,
        "height_error": he,
        "error_grid": eg,
    }

    if "pred_rgb" in data and "target_rgb" in data:
        cs = color_stats(data["pred_rgb"], data["target_rgb"])
        result["color"] = cs
        pc = cs["per_channel_mse"]
        logger.info(
            f"Color: mse={cs['mse']:.4f} mae={cs['mae']:.4f} psnr={cs['psnr']:.2f}dB "
            f"(r={pc.get('r', 0):.4f} g={pc.get('g', 0):.4f} b={pc.get('b', 0):.4f})"
        )

    return result


def evaluate_mae(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> dict:
    logger.info("Evaluating MAE on validation set...")
    data = _collect_mae_predictions(model, val_loader, device)

    pred   = data["reconstructed"]
    target = data["target"]
    feature_names = getattr(model, "target_feature_names", None)

    fm   = per_feature_mse(pred, target, feature_names)
    frm  = per_feature_rmse(pred, target, feature_names)
    fb   = per_feature_bias(pred, target, feature_names)
    fr2  = per_feature_r2(pred, target, feature_names)

    logger.info("Per-feature MSE / RMSE / Bias / R²:")
    names = feature_names or [f"feat_{i}" for i in range(pred.shape[1])]
    for name in names:
        logger.info(
            f"  {name:10s}: MSE={fm[name]:.4f}  RMSE={frm[name]:.4f}"
            f"  Bias={fb[name]:+.4f}  R²={fr2[name]:.4f}"
        )

    bins_data = {}
    for feat in ("z", "rel_z"):
        if feature_names and feat in feature_names:
            idx = feature_names.index(feat)
            bins_data[feat] = error_by_value_bins(pred, target, idx)

    groups = {
        "geom": ["x", "y", "z", "rel_z"],
        "rgb": ["r", "g", "b"],
        "intensity": ["intensity"],
    }
    group_mse = {}
    for g, members in groups.items():
        idxs = [names.index(n) for n in members if n in names]
        if idxs:
            group_mse[g] = float(((pred[:, idxs] - target[:, idxs]) ** 2).mean())
    for g, v in group_mse.items():
        logger.info(f"  group {g:9s}: MSE={v:.4f}")

    recon_error = np.linalg.norm(pred - target, axis=1)   # per-point L2

    latent_pca = None   # 2D PCA of stashed sample encodings
    sample = data.get("sample")
    if sample is not None and "encoded" in sample:
        enc = np.asarray(sample["encoded"], dtype=np.float64)
        if enc.shape[0] >= 3 and enc.shape[1] >= 2:
            centered = enc - enc.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            latent_pca = centered @ vt[:2].T

    result = {
        "feature_mse":  fm,
        "feature_rmse": frm,
        "feature_bias": fb,
        "feature_r2":   fr2,
        "group_mse":    group_mse,
        "recon_error":  recon_error,
        "bins_data":    bins_data,
        "pred":         pred,
        "target":       target,
        "feature_names": names,
    }
    if latent_pca is not None:
        result["latent_pca"] = latent_pca
    if "sample" in data:
        sample = data["sample"]
        sample["feature_names"] = names
        result["sample_3d"] = sample
    return result


def evaluate_hazus(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    logger.info("Evaluating HAZUS on validation set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if "labels" not in batch or batch["labels"] is None:
                continue
            xyz = batch["coords"].to(device)
            batch_idx = batch["batch"].to(device)
            mae_errors = batch.get("mae_errors")
            if mae_errors is not None:
                mae_errors = mae_errors.to(device)

            output = model(xyz, batch_idx, mae_errors)
            preds = output.get("predictions")
            if preds is None:
                preds = torch.argmax(output["logits"], dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    if not all_preds:
        logger.warning("No labels in HAZUS validation data — skipping evaluation")
        return {}

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    cm = confusion_matrix(preds, labels, num_classes)
    prf = precision_recall_f1(cm)
    metrics = {
        "confusion_matrix": cm,
        "confusion_matrix_normalized": normalize_confusion(cm),
        "overall_accuracy": overall_accuracy(cm),
        "mean_accuracy": mean_accuracy(cm),
        "macro_precision": prf["macro_precision"],
        "macro_recall": prf["macro_recall"],
        "macro_f1": prf["macro_f1"],
        "per_class_f1": prf["f1"],
        "support": cm.sum(axis=1),
    }
    logger.info(
        f"HAZUS: OA={metrics['overall_accuracy']:.4f} "
        f"mAcc={metrics['mean_accuracy']:.4f} macro-F1={prf['macro_f1']:.4f}"
    )
    return metrics

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
        if (
            hasattr(val_loader.dataset, "_class_names")
            and val_loader.dataset._class_names
        ):
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
            model,
            val_loader,
            device,
            num_classes,
            class_names,
            out_dir,
        )
        metrics.update(seg_metrics)

    elif task in ("seg_b_geom", "seg_b_color"):
        tau = float(cfg.task.get("eval_fscore_tau", 0.2)) if cfg else 0.2
        seg_b_metrics = evaluate_seg_b(model, val_loader, device, out_dir, tau=tau)
        metrics.update(seg_b_metrics)

    elif task == "mae":
        mae_metrics = evaluate_mae(model, val_loader, device, out_dir)
        metrics.update(mae_metrics)

    elif task == "hazus":
        num_classes = getattr(getattr(model, "codebook", None), "num_classes", 0)
        if num_classes:
            metrics.update(evaluate_hazus(model, val_loader, device, num_classes))

    plot_dir = out_dir / "plots"
    saved = plot_all(task, metrics, plot_dir, class_names=class_names)

    if saved:
        logger.info(f"Saved {len(saved)} evaluation plots to {plot_dir}/")
        for p in saved:
            logger.info(f"  {p.name}")

    _save_metrics_json(metrics, out_dir / "metrics.json")

    return metrics


def _json_safe(value: object, key: str = "") -> object:
    import numpy as _np

    skip = {"pred", "target", "coords", "distances", "recon_error",
            "pred_rgb", "target_rgb", "latent_pca", "sample", "sample_3d"}
    if key in skip:
        return None

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            sv = _json_safe(v, k)
            if sv is not None:
                out[k] = sv
        return out
    if isinstance(value, (_np.floating, _np.integer)):
        return value.item()
    if isinstance(value, _np.ndarray):
        if value.size > 1024:
            return {"_shape": list(value.shape), "_dtype": str(value.dtype)}
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save_metrics_json(metrics: dict, path: Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {k: _json_safe(v, k) for k, v in metrics.items()}
    safe = {k: v for k, v in safe.items() if v is not None}
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)
    logger.info(f"Saved metrics to {path}")
