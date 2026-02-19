# src/eval/plots.py
"""Matplotlib plotting functions for evaluation metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Loss curves ──────────────────────────────────────────────────────────


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float] | None,
    out_path: Path,
    title: str = "Training Loss",
) -> None:
    """Plot train (and optionally val) loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train", linewidth=1.5)
    if val_losses:
        ax.plot(epochs, val_losses, label="Val", linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale if loss range is large
    if train_losses and max(train_losses) / max(min(train_losses), 1e-10) > 100:
        ax.set_yscale("log")

    _save_fig(fig, out_path)


# ── Seg-A plots ──────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot confusion matrix heatmap."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
    else:
        cm_plot = cm.astype(float)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)))

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # Annotate cells
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_per_class_iou(
    iou: np.ndarray,
    class_names: list[str],
    out_path: Path,
    boundary_iou: np.ndarray | None = None,
    title: str = "Per-Class IoU",
) -> None:
    """Bar chart of per-class IoU, optionally with boundary IoU overlay."""
    n = len(class_names)
    x = np.arange(n)
    width = 0.35 if boundary_iou is not None else 0.6

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), 5))

    ax.bar(
        x - width / 2 if boundary_iou is not None else x,
        iou,
        width,
        label=f"IoU (mIoU={iou.mean():.3f})",
        color="#4C72B0",
    )

    if boundary_iou is not None:
        ax.bar(
            x + width / 2,
            boundary_iou,
            width,
            label=f"Boundary IoU (mean={boundary_iou.mean():.3f})",
            color="#DD8452",
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_fig(fig, out_path)


# ── Seg-B / inpainting plots ────────────────────────────────────────────


def plot_chamfer_histogram(
    distances: np.ndarray,
    out_path: Path,
    title: str = "Chamfer Distance Distribution",
) -> None:
    """Histogram of per-point chamfer distances with summary stats."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Clip outliers for better visualization
    p99 = np.percentile(distances, 99)
    plot_dists = distances[distances <= p99]

    ax.hist(plot_dists, bins=100, color="#4C72B0", alpha=0.8, edgecolor="none")

    # Stats lines
    mean_d = distances.mean()
    median_d = np.median(distances)
    ax.axvline(mean_d, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_d:.4f}")
    ax.axvline(median_d, color="orange", linestyle="--", linewidth=1.5, label=f"Median: {median_d:.4f}")

    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add text box with stats
    stats_text = (
        f"N={len(distances):,}\n"
        f"Mean={mean_d:.4f}\n"
        f"Median={median_d:.4f}\n"
        f"P90={np.percentile(distances, 90):.4f}\n"
        f"P99={p99:.4f}\n"
        f"Max={distances.max():.4f}"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    _save_fig(fig, out_path)


def plot_height_wise_error(
    bin_centers: np.ndarray,
    mean_error: np.ndarray,
    std_error: np.ndarray,
    counts: np.ndarray,
    out_path: Path,
    title: str = "Height-wise Reconstruction Error",
) -> None:
    """Error vs height with std band and point count overlay."""
    valid = counts > 0

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Error line with std fill
    ax1.plot(bin_centers[valid], mean_error[valid], "o-", color="#4C72B0", linewidth=1.5, markersize=4)
    ax1.fill_between(
        bin_centers[valid],
        np.maximum(mean_error[valid] - std_error[valid], 0),
        mean_error[valid] + std_error[valid],
        alpha=0.2, color="#4C72B0",
    )
    ax1.set_xlabel("Height (Z)")
    ax1.set_ylabel("Mean Error", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    # Point count on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(bin_centers[valid], counts[valid], width=(bin_centers[1] - bin_centers[0]) * 0.6,
            alpha=0.2, color="gray", label="Point Count")
    ax2.set_ylabel("Point Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_error_heatmap(
    grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    out_path: Path,
    title: str = "Spatial Error Heatmap (XY)",
) -> None:
    """2D heatmap of mean reconstruction error projected onto XY plane."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask NaN cells (no points)
    masked_grid = np.ma.masked_invalid(grid)

    im = ax.pcolormesh(
        x_edges, y_edges, masked_grid,
        cmap="hot_r", shading="flat",
    )
    fig.colorbar(im, ax=ax, label="Mean Error", shrink=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, out_path)


# ── MAE plots ────────────────────────────────────────────────────────────


def plot_per_feature_mse(
    feature_mse: dict[str, float],
    out_path: Path,
    title: str = "Per-Feature Reconstruction MSE",
) -> None:
    """Bar chart of per-feature MSE values."""
    names = [k for k in feature_mse if k != "total"]
    values = [feature_mse[k] for k in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color="#4C72B0", alpha=0.8)

    # Annotate bars with values
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom", fontsize=9,
        )

    total = feature_mse.get("total", 0)
    ax.axhline(total, color="red", linestyle="--", linewidth=1.5,
               label=f"Total MSE: {total:.4f}")

    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_path)


# ── Convenience ──────────────────────────────────────────────────────────


def plot_all(
    task: str,
    metrics: dict,
    out_dir: Path,
    class_names: list[str] | None = None,
) -> list[Path]:
    """Dispatch to appropriate plotting functions based on task.

    Returns list of saved plot paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Loss curves (common to all tasks)
    if "train_losses" in metrics:
        p = out_dir / "loss_curves.png"
        plot_loss_curves(
            metrics["train_losses"],
            metrics.get("val_losses"),
            p,
            title=f"{task} — Loss Curves",
        )
        saved.append(p)

    if task in ("seg_a",):
        if "confusion_matrix" in metrics and class_names:
            p = out_dir / "confusion_matrix.png"
            plot_confusion_matrix(metrics["confusion_matrix"], class_names, p)
            saved.append(p)

        if "per_class_iou" in metrics and class_names:
            p = out_dir / "per_class_iou.png"
            plot_per_class_iou(
                metrics["per_class_iou"],
                class_names,
                p,
                boundary_iou=metrics.get("boundary_iou"),
            )
            saved.append(p)

    if task in ("seg_b_geom", "seg_b_color"):
        if "chamfer" in metrics:
            p = out_dir / "chamfer_histogram.png"
            plot_chamfer_histogram(metrics["chamfer"]["distances"], p)
            saved.append(p)

        if "height_error" in metrics:
            he = metrics["height_error"]
            p = out_dir / "height_wise_error.png"
            plot_height_wise_error(
                he["bin_centers"], he["mean_error"],
                he["std_error"], he["counts"], p,
            )
            saved.append(p)

        if "error_grid" in metrics:
            eg = metrics["error_grid"]
            p = out_dir / "error_heatmap.png"
            plot_error_heatmap(eg["grid"], eg["x_edges"], eg["y_edges"], p)
            saved.append(p)

    if task == "mae" and "feature_mse" in metrics:
        p = out_dir / "per_feature_mse.png"
        plot_per_feature_mse(metrics["feature_mse"], p)
        saved.append(p)

    return saved
