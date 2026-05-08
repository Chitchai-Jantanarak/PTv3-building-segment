from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _subsample(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, max_points, replace=False)


def _equal_aspect_3d(ax, coords: np.ndarray) -> None:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = maxs - mins
    span = float(spans.max())
    if span <= 0:
        return
    centers = (mins + maxs) / 2
    for set_lim, c in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim), centers, strict=True
    ):
        set_lim(c - span / 2, c + span / 2)


def _morton_order(coord_norm_01: np.ndarray, bits: int = 10) -> np.ndarray:
    q = np.clip((coord_norm_01 * (1 << bits)).astype(np.uint64), 0, (1 << bits) - 1)

    def spread(v):
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        v = (v | (v << 2)) & 0x1249249249249249
        return v

    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)
    return np.argsort(code, kind="stable")


def plot_masked_vs_recon_3d(
    coord: np.ndarray,
    target_color: np.ndarray,
    recon_color: np.ndarray,
    masked_indices: np.ndarray,
    out_path: Path,
    max_points: int = 20_000,
    title: str = "Masked vs Reconstructed (RGB)",
) -> None:
    rng = np.random.default_rng(0)
    n = coord.shape[0]
    keep = _subsample(n, max_points, rng)
    coord_s = coord[keep]
    is_masked = np.isin(keep, masked_indices)

    fig = plt.figure(figsize=(14, 6))

    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    colors_l = np.where(is_masked[:, None], 0.7, target_color[keep])
    ax_l.scatter(
        coord_s[:, 0],
        coord_s[:, 1],
        coord_s[:, 2],
        c=np.clip(colors_l, 0, 1),
        s=2,
        depthshade=False,
    )
    ax_l.set_title("Ground truth\n(masked points greyed)")
    _equal_aspect_3d(ax_l, coord_s)

    ax_r = fig.add_subplot(1, 2, 2, projection="3d")
    colors_r = np.where(is_masked[:, None], recon_color[keep], target_color[keep])
    ax_r.scatter(
        coord_s[:, 0],
        coord_s[:, 1],
        coord_s[:, 2],
        c=np.clip(colors_r, 0, 1),
        s=2,
        depthshade=False,
    )
    ax_r.set_title("Reconstruction\n(masked = predicted RGB)")
    _equal_aspect_3d(ax_r, coord_s)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_density_heatmap(
    coord: np.ndarray,
    out_path: Path,
    error: np.ndarray | None = None,
    grid_res: float = 1.0,
    title: str = "Density / Error Heatmap (XY)",
) -> None:
    x, y = coord[:, 0], coord[:, 1]
    x_edges = np.arange(x.min(), x.max() + grid_res, grid_res)
    y_edges = np.arange(y.min(), y.max() + grid_res, grid_res)

    counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    counts = counts.T # (ny, nx)

    n_panels = 1 if error is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    axes = axes[0]

    log_counts = np.log1p(counts)
    im0 = axes[0].pcolormesh(
        x_edges, y_edges, log_counts, cmap="viridis", shading="flat"
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title(f"log(1 + count)   max={int(counts.max())}")
    fig.colorbar(im0, ax=axes[0], shrink=0.7)

    if error is not None:
        err_sum, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=error)
        with np.errstate(invalid="ignore", divide="ignore"):
            err_mean = np.where(counts > 0, err_sum.T / counts, np.nan)

        alpha = np.clip(log_counts / max(log_counts.max(), 1e-6), 0.15, 1.0)

        im1 = axes[1].pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(err_mean),
            cmap="hot_r",
            shading="flat",
            alpha=alpha,
        )
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[1].set_title("Mean error per cell (alpha = density)")
        fig.colorbar(im1, ax=axes[1], shrink=0.7, label="Mean error")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_pca_feature_embedding(
    coord: np.ndarray,
    features: np.ndarray,
    out_path: Path,
    max_points: int = 20_000,
    title: str = "Encoder Features — PCA(3) → RGB",
) -> None:
    rng = np.random.default_rng(0)
    keep = _subsample(coord.shape[0], max_points, rng)
    c = coord[keep]
    f = features[keep].astype(np.float32, copy=False)

    f = f - f.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(f, full_matrices=False)
    proj = f @ vt[:3].T # (M, 3)

    p_min = np.percentile(proj, 2, axis=0)
    p_max = np.percentile(proj, 98, axis=0)
    rgb = np.clip((proj - p_min) / np.maximum(p_max - p_min, 1e-6), 0, 1)

    fig = plt.figure(figsize=(13, 6))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_a.scatter(c[:, 0], c[:, 1], c[:, 2], c=rgb, s=2, depthshade=False)
    ax_a.set_title("Iso view")
    _equal_aspect_3d(ax_a, c)

    ax_b = fig.add_subplot(1, 2, 2)
    ax_b.scatter(c[:, 0], c[:, 1], c=rgb, s=2)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("X")
    ax_b.set_ylabel("Y")
    ax_b.set_title("Top-down view")

    var_ratio = (s[:3] ** 2) / max((s**2).sum(), 1e-12)
    fig.suptitle(
        f"{title}\nexplained variance: "
        f"{var_ratio[0]:.2f} / {var_ratio[1]:.2f} / {var_ratio[2]:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_patch_centers(
    coord: np.ndarray,
    out_path: Path,
    voxel_size: float = 2.0,
    max_points: int = 30_000,
    title: str = "Patch / Voxel Centers",
) -> None:
    rng = np.random.default_rng(0)
    keep = _subsample(coord.shape[0], max_points, rng)
    cloud = coord[keep]

    grid_idx = np.floor(coord / voxel_size).astype(np.int64)
    keys, inverse = np.unique(grid_idx, axis=0, return_inverse=True)
    n_buckets = keys.shape[0]

    sums = np.zeros((n_buckets, 3), dtype=np.float64)
    np.add.at(sums, inverse, coord)
    counts = np.bincount(inverse, minlength=n_buckets).astype(np.float64)
    centers = sums / counts[:, None]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        cloud[:, 2],
        c="lightgray",
        s=1,
        alpha=0.25,
        depthshade=False,
    )
    sizes = 6 + 30 * (counts / counts.max())
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=counts,
        cmap="plasma",
        s=sizes,
        edgecolors="black",
        linewidths=0.3,
        depthshade=False,
    )
    ax.set_title(
        f"{title}   voxel={voxel_size}   "
        f"#patches={n_buckets}   median_pts/patch={int(np.median(counts))}"
    )
    _equal_aspect_3d(ax, cloud)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_token_energy(
    coord: np.ndarray,
    features: np.ndarray,
    out_path: Path,
    max_points: int = 30_000,
    title: str = "Token Energy ‖f‖₂",
) -> None:
    rng = np.random.default_rng(0)
    keep = _subsample(coord.shape[0], max_points, rng)
    c = coord[keep]
    energy = np.linalg.norm(features[keep], axis=1)

    vmin, vmax = np.percentile(energy, (2, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.get_cmap("inferno")
    colors = cmap(norm(energy))

    fig = plt.figure(figsize=(13, 6))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_a.scatter(c[:, 0], c[:, 1], c[:, 2], c=colors, s=2, depthshade=False)
    ax_a.set_title("3D — colored by ‖f‖₂")
    _equal_aspect_3d(ax_a, c)
    fig.colorbar(
        mpl_cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_a,
        shrink=0.6,
        label="‖f‖₂",
    )

    ax_b = fig.add_subplot(1, 2, 2)
    ax_b.hist(energy, bins=80, color="#4C72B0", alpha=0.85, edgecolor="none")
    ax_b.axvline(
        np.median(energy),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"median={np.median(energy):.3f}",
    )
    ax_b.axvline(
        energy.mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={energy.mean():.3f}",
    )
    ax_b.set_xlabel("‖f‖₂")
    ax_b.set_ylabel("Count")
    ax_b.set_title("Energy distribution")
    ax_b.legend(fontsize=9)
    ax_b.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_space_filling_curve(
    coord: np.ndarray,
    out_path: Path,
    order: np.ndarray | None = None,
    max_points: int = 8_000,
    title: str = "Serialization (Space-Filling) Curve",
) -> None:
    if order is None:
        cmin = coord.min(axis=0, keepdims=True)
        cmax = coord.max(axis=0, keepdims=True)
        c01 = (coord - cmin) / np.maximum(cmax - cmin, 1e-6)
        order = _morton_order(c01)

    seq = coord[order]
    if seq.shape[0] > max_points:
        step = seq.shape[0] // max_points
        seq = seq[::step]

    t = np.linspace(0, 1, seq.shape[0])

    fig = plt.figure(figsize=(13, 6))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    rng = np.random.default_rng(0)
    bg_keep = _subsample(coord.shape[0], min(coord.shape[0], 15_000), rng)
    ax_a.scatter(
        coord[bg_keep, 0],
        coord[bg_keep, 1],
        coord[bg_keep, 2],
        c="lightgray",
        s=1,
        alpha=0.2,
        depthshade=False,
    )

    cmap = mpl_cm.get_cmap("turbo")
    for i in range(seq.shape[0] - 1):
        ax_a.plot(
            seq[i : i + 2, 0],
            seq[i : i + 2, 1],
            seq[i : i + 2, 2],
            color=cmap(t[i]),
            linewidth=0.7,
        )
    ax_a.set_title(f"3D curve ({seq.shape[0]} samples)")
    _equal_aspect_3d(ax_a, seq)

    ax_b = fig.add_subplot(1, 2, 2)
    ax_b.plot(seq[:, 0], seq[:, 1], "-", linewidth=0.5, color="black", alpha=0.4)
    ax_b.scatter(seq[:, 0], seq[:, 1], c=t, cmap="turbo", s=4)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("X")
    ax_b.set_ylabel("Y")
    ax_b.set_title("Top-down (color = serialization progress)")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out_path)


def plot_all_3d(
    sample: dict,
    out_dir: Path,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    coord = sample["coord"]
    target = sample.get("target")
    recon = sample.get("reconstructed")
    feat_names = sample.get("feature_names") or []
    encoded = sample.get("encoded")
    vis_idx = sample.get("visible_indices")
    msk_idx = sample.get("masked_indices")

    rgb_cols = [feat_names.index(n) for n in ("r", "g", "b") if n in feat_names]
    if (
        target is not None
        and recon is not None
        and msk_idx is not None
        and len(rgb_cols) == 3
    ):
        p = out_dir / "mae_masked_vs_recon_3d.png"
        plot_masked_vs_recon_3d(
            coord=coord,
            target_color=target[:, rgb_cols],
            recon_color=recon[:, rgb_cols],
            masked_indices=msk_idx,
            out_path=p,
        )
        saved.append(p)

    err = None
    if target is not None and recon is not None:
        err = np.linalg.norm(target - recon, axis=1)
    p = out_dir / "mae_density_heatmap.png"
    plot_density_heatmap(coord=coord, out_path=p, error=err)
    saved.append(p)

    if encoded is not None and vis_idx is not None:
        vis_coord = coord[vis_idx]

        p = out_dir / "mae_pca_features.png"
        plot_pca_feature_embedding(coord=vis_coord, features=encoded, out_path=p)
        saved.append(p)

        p = out_dir / "mae_token_energy.png"
        plot_token_energy(coord=vis_coord, features=encoded, out_path=p)
        saved.append(p)

    p = out_dir / "mae_patch_centers.png"
    plot_patch_centers(coord=coord, out_path=p)
    saved.append(p)

    p = out_dir / "mae_space_filling.png"
    plot_space_filling_curve(coord=coord, out_path=p, order=sample.get("order"))
    saved.append(p)

    return saved
