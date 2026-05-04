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
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7
            )

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
    ax.axvline(
        mean_d, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_d:.4f}"
    )
    ax.axvline(
        median_d,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_d:.4f}",
    )

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
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
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
    ax1.plot(
        bin_centers[valid],
        mean_error[valid],
        "o-",
        color="#4C72B0",
        linewidth=1.5,
        markersize=4,
    )
    ax1.fill_between(
        bin_centers[valid],
        np.maximum(mean_error[valid] - std_error[valid], 0),
        mean_error[valid] + std_error[valid],
        alpha=0.2,
        color="#4C72B0",
    )
    ax1.set_xlabel("Height (Z)")
    ax1.set_ylabel("Mean Error", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    # Point count on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(
        bin_centers[valid],
        counts[valid],
        width=(bin_centers[1] - bin_centers[0]) * 0.6,
        alpha=0.2,
        color="gray",
        label="Point Count",
    )
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
        x_edges,
        y_edges,
        masked_grid,
        cmap="hot_r",
        shading="flat",
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    total = feature_mse.get("total", 0)
    ax.axhline(
        total,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Total MSE: {total:.4f}",
    )

    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_pred_vs_actual(
    pred: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    highlight: list[str] | None = None,
    max_points: int = 5000,
) -> None:
    """Scatter plot of predicted vs actual value for each feature.
    
    The key diagnostic plot — if points lie on y=x line, reconstruction is good.
    If points cluster horizontally at mean(target), model has mean-collapsed.
    """
    highlight = highlight or ["z", "rel_z"]
    n_feat = len(feature_names)
    ncols = min(4, n_feat)
    nrows = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten() if n_feat > 1 else [axes]

    rng = np.random.default_rng(42)

    for i, name in enumerate(feature_names):
        ax = axes[i]
        t = target[:, i]
        p = pred[:, i]

        # Subsample for speed
        idx = rng.choice(len(t), min(max_points, len(t)), replace=False)
        t_s, p_s = t[idx], p[idx]

        color = "crimson" if name in highlight else "#4C72B0"
        ax.scatter(t_s, p_s, s=2, alpha=0.3, color=color, rasterized=True)

        # Perfect prediction line
        lims = [min(t_s.min(), p_s.min()), max(t_s.max(), p_s.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="y=x")

        # R² in title
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        ax.set_title(f"{name}  R²={r2:.3f}",
                     color=color if name in highlight else "black",
                     fontweight="bold" if name in highlight else "normal")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Predicted vs Actual — MAE Reconstruction", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=120)


def plot_error_distribution(
    pred: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    highlight: list[str] | None = None,
) -> None:
    """Error distribution histogram per feature.
    
    Narrow centered distribution → good reconstruction.
    Wide or off-center → high variance or bias.
    """
    highlight = highlight or ["z", "rel_z"]
    n_feat = len(feature_names)
    ncols = min(4, n_feat)
    nrows = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten() if n_feat > 1 else [axes]

    for i, name in enumerate(feature_names):
        ax = axes[i]
        errors = pred[:, i] - target[:, i]   # signed error

        color = "crimson" if name in highlight else "#4C72B0"
        ax.hist(errors, bins=60, color=color, alpha=0.7, edgecolor="none")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
        ax.axvline(errors.mean(), color="orange", linewidth=1.5,
                   linestyle="-", label=f"bias={errors.mean():.3f}")

        rmse = float(np.sqrt((errors**2).mean()))
        ax.set_title(f"{name}  RMSE={rmse:.3f}")
        ax.set_xlabel("Pred − Actual")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Reconstruction Error Distribution", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi=120)


def plot_r2_bars(
    r2_scores: dict[str, float],
    out_path: Path,
    title: str = "R² Score per Feature",
) -> None:
    """Bar chart of R² scores.
    
    R²=1 is perfect. R²=0 means predicting mean. R²<0 is worse than mean.
    Most useful single plot for diagnosing mean-collapse on z/rel_z.
    """
    names = list(r2_scores.keys())
    values = list(r2_scores.values())

    colors = []
    for n, v in zip(names, values):
        if n in ("z", "rel_z"):
            colors.append("crimson" if v < 0.3 else "darkorange")
        else:
            colors.append("#4C72B0" if v >= 0.3 else "gray")

    fig, ax = plt.subplots(figsize=(max(8, len(names)), 5))
    bars = ax.bar(names, values, color=colors, alpha=0.85)

    for bar, val in zip(bars, values):
        ypos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-")
    ax.axhline(1, color="green", linewidth=1.0, linestyle="--",
               alpha=0.5, label="Perfect (R²=1)")
    ax.axhline(0, color="red", linewidth=1.0, linestyle="--",
               alpha=0.3, label="Mean baseline (R²=0)")

    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.set_ylim(min(-0.2, min(values) - 0.1), 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_error_by_value(
    bins_data: dict[str, dict],
    out_path: Path,
    title: str = "MSE by Actual Value (z and rel_z)",
) -> None:
    n = len(bins_data)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 4))
    if n == 1:
        axes = [axes]

    for ax, (feat_name, data) in zip(axes, bins_data.items()):
        centers = data["bin_centers"]
        mse = data["mean_mse"]
        counts = data["counts"]
        valid = counts > 0

        color = "crimson" if feat_name in ("z", "rel_z") else "#4C72B0"
        ax.plot(centers[valid], mse[valid], "o-", color=color,
                linewidth=1.5, markersize=4)
        ax.fill_between(centers[valid], 0, mse[valid], alpha=0.15, color=color)

        ax2 = ax.twinx()
        ax2.bar(centers[valid], counts[valid],
                width=(centers[1] - centers[0]) * 0.6 if len(centers) > 1 else 1,
                alpha=0.15, color="gray")
        ax2.set_ylabel("Point count", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray")

        ax.set_xlabel(f"Actual {feat_name} value")
        ax.set_ylabel("MSE")
        ax.set_title(f"{feat_name}: MSE vs actual value")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
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

    # Seg-A Plots
    if task == "seg_a":
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

    # Seg-B Plots
    elif task in ("seg_b_geom", "seg_b_color"):
        if "chamfer" in metrics:
            p = out_dir / "chamfer_histogram.png"
            plot_chamfer_histogram(metrics["chamfer"]["distances"], p)
            saved.append(p)
            
        if "height_error" in metrics:
            p = out_dir / "height_wise_error.png"
            he = metrics["height_error"]
            plot_height_wise_error(
                he["bin_centers"], he["mean_error"], he["std_error"], he["counts"], p
            )
            saved.append(p)
            
        if "error_grid" in metrics:
            p = out_dir / "spatial_error_heatmap.png"
            eg = metrics["error_grid"]
            plot_error_heatmap(eg["grid"], eg["x_edges"], eg["y_edges"], p)
            saved.append(p)

    # MAE Plots
    elif task == "mae":
        if "feature_mse" in metrics:
            p = out_dir / "per_feature_mse.png"
            plot_per_feature_mse(metrics["feature_mse"], p)
            saved.append(p)
            
        if "feature_r2" in metrics:
            p = out_dir / "per_feature_r2.png"
            plot_r2_bars(metrics["feature_r2"], p)
            saved.append(p)
            
        if all(k in metrics for k in ("pred", "target", "feature_names")):
            p_scatter = out_dir / "pred_vs_actual.png"
            plot_pred_vs_actual(
                metrics["pred"], metrics["target"], metrics["feature_names"], p_scatter
            )
            saved.append(p_scatter)

            p_dist = out_dir / "error_distribution.png"
            plot_error_distribution(
                metrics["pred"], metrics["target"], metrics["feature_names"], p_dist
            )
            saved.append(p_dist)

        if "bins_data" in metrics:
            p = out_dir / "error_by_value.png"
            plot_error_by_value(metrics["bins_data"], p)
            saved.append(p)

    return saved