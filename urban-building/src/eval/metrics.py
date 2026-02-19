# src/eval/metrics.py
"""Evaluation metrics for all pipeline stages."""

import numpy as np

# ── Seg-A metrics ────────────────────────────────────────────────────────


def confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int = -100,
) -> np.ndarray:
    """Compute NxN confusion matrix. Rows=true, Cols=predicted."""
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    p = preds[valid]
    t = targets[valid]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls, pred_cls in zip(t, p, strict=True):
        cm[true_cls, pred_cls] += 1
    return cm


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU from confusion matrix. Returns array of shape (num_classes,)."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, 0.0)
    return iou


def boundary_iou(
    preds: np.ndarray,
    targets: np.ndarray,
    coords: np.ndarray,
    num_classes: int,
    boundary_dist: float = 0.5,
    ignore_index: int = -100,
) -> np.ndarray:
    """IoU computed only on points near class boundaries.

    A point is on a boundary if any of its k-nearest neighbors has a
    different ground-truth label. Uses a distance threshold instead of
    exact kNN for speed on large point clouds.

    Args:
        preds: predicted labels (N,)
        targets: ground truth labels (N,)
        coords: point coordinates (N, 3)
        num_classes: number of classes
        boundary_dist: distance threshold to define boundary region
        ignore_index: label to ignore

    Returns:
        Per-class boundary IoU array of shape (num_classes,).
    """
    from scipy.spatial import cKDTree

    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    coords_v = coords[valid]
    targets_v = targets[valid]
    preds_v = preds[valid]

    tree = cKDTree(coords_v)
    # For each point, find neighbors within boundary_dist
    neighbor_lists = tree.query_ball_point(coords_v, r=boundary_dist)

    # A point is on the boundary if any neighbor has a different label
    boundary_mask = np.zeros(len(coords_v), dtype=bool)
    for i, neighbors in enumerate(neighbor_lists):
        labels_in_neighborhood = targets_v[neighbors]
        if len(np.unique(labels_in_neighborhood)) > 1:
            boundary_mask[i] = True

    if boundary_mask.sum() == 0:
        return np.zeros(num_classes)

    cm = confusion_matrix(
        preds_v[boundary_mask], targets_v[boundary_mask], num_classes
    )
    return per_class_iou(cm)


# ── Seg-B / inpainting metrics ──────────────────────────────────────────


def chamfer_stats(
    pred_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Per-point chamfer distances and summary stats.

    Returns dict with keys: distances, mean, median, p90, p99, max.
    """
    diff = pred_xyz - target_xyz
    dists = np.linalg.norm(diff, axis=-1)
    return {
        "distances": dists,
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "p90": float(np.percentile(dists, 90)),
        "p99": float(np.percentile(dists, 99)),
        "max": float(dists.max()),
    }


def height_wise_error(
    pred_xyz: np.ndarray,
    target_xyz: np.ndarray,
    n_bins: int = 20,
) -> dict[str, np.ndarray]:
    """Reconstruction error binned by Z coordinate.

    Returns dict with keys: bin_centers, mean_error, std_error, counts.
    """
    dists = np.linalg.norm(pred_xyz - target_xyz, axis=-1)
    z = target_xyz[:, 2]

    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-6:
        return {
            "bin_centers": np.array([z_min]),
            "mean_error": np.array([dists.mean()]),
            "std_error": np.array([dists.std()]),
            "counts": np.array([len(dists)]),
        }

    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(z, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mean_err = np.zeros(n_bins)
    std_err = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            mean_err[b] = dists[mask].mean()
            std_err[b] = dists[mask].std()

    return {
        "bin_centers": bin_centers,
        "mean_error": mean_err,
        "std_error": std_err,
        "counts": counts,
    }


def spatial_error_grid(
    pred_xyz: np.ndarray,
    target_xyz: np.ndarray,
    grid_res: float = 1.0,
) -> dict[str, np.ndarray]:
    """2D XY error heatmap — mean reconstruction error per grid cell.

    Returns dict with keys: grid, x_edges, y_edges, x_centers, y_centers.
    """
    dists = np.linalg.norm(pred_xyz - target_xyz, axis=-1)
    x, y = target_xyz[:, 0], target_xyz[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    nx = max(1, int(np.ceil((x_max - x_min) / grid_res)))
    ny = max(1, int(np.ceil((y_max - y_min) / grid_res)))

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    xi = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)

    grid_sum = np.zeros((ny, nx))
    grid_count = np.zeros((ny, nx), dtype=np.int64)

    for i in range(len(dists)):
        grid_sum[yi[i], xi[i]] += dists[i]
        grid_count[yi[i], xi[i]] += 1

    grid = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

    return {
        "grid": grid,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": (x_edges[:-1] + x_edges[1:]) / 2,
        "y_centers": (y_edges[:-1] + y_edges[1:]) / 2,
    }


# ── MAE metrics ─────────────────────────────────────────────────────────


def per_feature_mse(
    pred: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Per-feature MSE breakdown.

    Args:
        pred: predicted features (N, D)
        target: target features (N, D)
        feature_names: optional names for each feature dim

    Returns:
        Dict mapping feature name to MSE value.
    """
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(pred.shape[1])]

    result = {}
    for i, name in enumerate(feature_names):
        result[name] = float(((pred[:, i] - target[:, i]) ** 2).mean())
    result["total"] = float(((pred - target) ** 2).mean())
    return result
