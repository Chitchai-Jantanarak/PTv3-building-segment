# src/eval/metrics.py
import numpy as np

# ── Seg-A metrics ────────────────────────────────────────────────────────


def confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int = -100,
) -> np.ndarray:   # rows=true, cols=predicted
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    p = preds[valid]
    t = targets[valid]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls, pred_cls in zip(t, p, strict=True):
        cm[true_cls, pred_cls] += 1
    return cm


def per_class_iou(cm: np.ndarray) -> np.ndarray:   # -> (num_classes,)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, 0.0)
    return iou


def overall_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total > 0 else 0.0


def mean_accuracy(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    row = cm.sum(axis=1)
    acc = np.where(row > 0, tp / np.maximum(row, 1), 0.0)   # mean per-class recall
    valid = row > 0
    return float(acc[valid].mean()) if valid.any() else 0.0


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast(num, den).shape, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)   # 0 where den==0
    return out


def precision_recall_f1(cm: np.ndarray) -> dict[str, np.ndarray | float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def freq_weighted_iou(cm: np.ndarray) -> float:
    iou = per_class_iou(cm)
    total = cm.sum()
    freq = cm.sum(axis=1) / max(total, 1)
    return float((freq * iou).sum())          # sum_c freq_c * IoU_c


def normalize_confusion(cm: np.ndarray) -> np.ndarray:
    row = cm.sum(axis=1, keepdims=True)
    return np.where(row > 0, cm / np.maximum(row, 1), 0.0)   # rows sum to 1


def boundary_iou(
    preds: np.ndarray,
    targets: np.ndarray,
    coords: np.ndarray,
    num_classes: int,
    boundary_dist: float = 0.5,
    ignore_index: int = -100,
) -> np.ndarray:   # -> per-class boundary IoU (num_classes,)
    from scipy.spatial import cKDTree

    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    coords_v = coords[valid]
    targets_v = targets[valid]
    preds_v = preds[valid]

    tree = cKDTree(coords_v)
    neighbor_lists = tree.query_ball_point(coords_v, r=boundary_dist)

    boundary_mask = np.zeros(len(coords_v), dtype=bool)
    for i, neighbors in enumerate(neighbor_lists):
        labels_in_neighborhood = targets_v[neighbors]
        if len(np.unique(labels_in_neighborhood)) > 1:
            boundary_mask[i] = True

    if boundary_mask.sum() == 0:
        return np.zeros(num_classes)

    cm = confusion_matrix(preds_v[boundary_mask], targets_v[boundary_mask], num_classes)
    return per_class_iou(cm)


# ── Seg-B / inpainting metrics ──────────────────────────────────────────


def nn_distances(
    pred_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    pred = np.asarray(pred_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    forward, _ = cKDTree(target).query(pred)    # pred[i]   -> nearest target
    backward, _ = cKDTree(pred).query(target)   # target[j] -> nearest pred
    return forward, backward


def chamfer_stats(
    pred_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> dict[str, float | np.ndarray]:
    forward, backward = nn_distances(pred_xyz, target_xyz)
    return chamfer_stats_from_nn(forward, backward)


def chamfer_stats_from_nn(
    forward: np.ndarray,
    backward: np.ndarray,
) -> dict[str, float | np.ndarray]:
    distances = np.concatenate([forward, backward])
    return {
        "distances": distances,
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(distances.max()),
        "forward_mean": float(forward.mean()),
        "backward_mean": float(backward.mean()),
        "chamfer": float(forward.mean() + backward.mean()),
        "hausdorff": float(max(forward.max(), backward.max())),
        "hausdorff_p95": float(
            max(np.percentile(forward, 95), np.percentile(backward, 95))
        ),
    }


def fscore_at_tau(
    forward: np.ndarray,
    backward: np.ndarray,
    tau: float = 0.2,
) -> dict[str, float]:
    precision = float((forward < tau).mean())   # frac pred within tau of a target
    recall = float((backward < tau).mean())     # frac target within tau of a pred
    denom = precision + recall
    f1 = float(2 * precision * recall / denom) if denom > 0 else 0.0
    return {"tau": float(tau), "precision": precision, "recall": recall, "f1": f1}


def height_wise_error(
    dists: np.ndarray,
    coords: np.ndarray,
    n_bins: int = 20,
) -> dict[str, np.ndarray]:
    z = coords[:, 2]

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
    dists: np.ndarray,
    coords: np.ndarray,
    grid_res: float = 1.0,
) -> dict[str, np.ndarray]:
    x, y = coords[:, 0], coords[:, 1]

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


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:   # sRGB[0,1] -> CIE Lab (D65)
    rgb = np.clip(rgb, 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def color_stats(
    pred_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> dict[str, float | dict[str, float]]:
    pred = np.asarray(pred_rgb, dtype=np.float64)
    target = np.asarray(target_rgb, dtype=np.float64)

    if pred.max() > 1.5 or target.max() > 1.5:
        pred = pred / 255.0
        target = target / 255.0

    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    se = (pred - target) ** 2
    mse = float(se.mean())
    mae = float(np.abs(pred - target).mean())
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 1e-12 else 99.0

    channels = ["r", "g", "b"][: pred.shape[1]]
    per_channel = {c: float(se[:, i].mean()) for i, c in enumerate(channels)}

    delta_e = float("nan")
    if pred.shape[1] >= 3:
        de = np.linalg.norm(_srgb_to_lab(pred[:, :3]) - _srgb_to_lab(target[:, :3]), axis=1)
        delta_e = float(de.mean())

    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "delta_e": delta_e,
        "per_channel_mse": per_channel,
    }


# ── MAE metrics ─────────────────────────────────────────────────────────

def per_feature_mse(
    pred: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(pred.shape[1])]

    result = {}
    for i, name in enumerate(feature_names):
        result[name] = float(((pred[:, i] - target[:, i]) ** 2).mean())

    result["total"] = float(((pred - target) ** 2).mean())
    return result
    

def per_feature_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:   # RMSE: same units as original data
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(pred.shape[1])]
    result = {}
    for i, name in enumerate(feature_names):
        mse = float(((pred[:, i] - target[:, i]) ** 2).mean())
        result[name] = float(np.sqrt(mse))
    result["total"] = float(np.sqrt(((pred - target) ** 2).mean()))
    return result


def per_feature_bias(
    pred: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(pred.shape[1])]
    result = {}
    for i, name in enumerate(feature_names):
        result[name] = float((pred[:, i] - target[:, i]).mean())
    return result


def per_feature_r2(
    pred: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(pred.shape[1])]
    result = {}
    for i, name in enumerate(feature_names):
        t = target[:, i]
        p = pred[:, i]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        result[name] = float(r2)
    return result


def error_by_value_bins(
    pred: np.ndarray,
    target: np.ndarray,
    feature_idx: int,
    n_bins: int = 20,
) -> dict[str, np.ndarray]:
    t = target[:, feature_idx]
    p = pred[:, feature_idx]
    errors = (t - p) ** 2

    t_min, t_max = t.min(), t.max()
    if t_max - t_min < 1e-6:
        return {
            "bin_centers": np.array([t_min]),
            "mean_mse": np.array([errors.mean()]),
            "counts": np.array([len(errors)]),
        }

    bin_edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(t, bin_edges) - 1, 0, n_bins - 1)

    mean_mse = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            mean_mse[b] = errors[mask].mean()

    return {
        "bin_centers": bin_centers,
        "mean_mse": mean_mse,
        "counts": counts,
    }