# src/infer/pipeline.py
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from src.core.utils import get_logger
from src.infer.export import export_csv, export_las
from src.models.hazus_head.codebook import HAZUS_BUILDING_TYPES
from src.models.hazus_head.features import compute_geometry_stats
from src.models.hazus_head.rules import FEMAConstraints

logger = get_logger("PIPE")

BUILDING_CLASS = 2

# FEMA occupancy main classes (sub-classes of building)
FEMA_MAIN_CLASSES = {
    0: "RES",  # Residential
    1: "COM",  # Commercial
    2: "IND",  # Industrial
    3: "GOV",  # Government
    4: "EDU",  # Education
    5: "AGR",  # Agricultural
    6: "REL",  # Religious
}

# Priority order for selecting among multiple valid HAZUS types
_HAZUS_PRIORITY = [
    "W1",
    "W2",
    "S1L",
    "S2L",
    "C1L",
    "C2L",
    "RM1L",
    "S1M",
    "S2M",
    "C1M",
    "C2M",
    "RM1M",
    "S1H",
    "S2H",
    "C1H",
    "C2H",
    "URM",
]


def _classify_occupancy(height: float, area: float, aspect_ratio: float) -> tuple[int, str]:
    """Rule-based FEMA occupancy classification from geometry."""
    # AGR: very large footprint, low height, elongated
    if area > 3000 and height < 8 and aspect_ratio > 3.0:
        return 5, "AGR"
    # IND: large footprint, low-mid height
    if area > 2000 and height < 20:
        return 2, "IND"
    # RES: small footprint, low height
    if area < 500 and height < 15:
        return 0, "RES"
    # COM: default for mid-size / taller buildings
    return 1, "COM"


def _classify_building_rule_based(
    xyz: np.ndarray,
    constraints: FEMAConstraints,
) -> dict:
    xyz_t = torch.from_numpy(xyz).float()
    geom = compute_geometry_stats(xyz_t)

    height = geom["height"].item()
    area = geom["footprint_area"].item()
    aspect_ratio = geom["aspect_ratio"].item()
    n_points = int(geom["n_points"].item())

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)

    # ── Structural type (HAZUS) ──
    valid_types = constraints.get_valid_types(height, area)

    if len(valid_types) == 1:
        hazus_code = valid_types[0]
        confidence = 1.0
    elif len(valid_types) > 1:
        for code in _HAZUS_PRIORITY:
            if code in valid_types:
                hazus_code = code
                break
        else:
            hazus_code = valid_types[0]
        confidence = 0.8
    else:
        if height < 20:
            hazus_code = "W1"
        elif height < 50:
            hazus_code = "C1M"
        else:
            hazus_code = "S1H"
        confidence = 0.5

    hazus_name = "Unknown"
    for _idx, info in HAZUS_BUILDING_TYPES.items():
        if info["code"] == hazus_code:
            hazus_name = info["name"]
            break

    # ── Occupancy class (FEMA main) ──
    occ_id, occ_class = _classify_occupancy(height, area, aspect_ratio)

    return {
        "hazus_code": hazus_code,
        "hazus_name": hazus_name,
        "occupancy_id": occ_id,
        "occupancy_class": occ_class,
        "height": round(height, 2),
        "footprint_area": round(area, 2),
        "aspect_ratio": round(aspect_ratio, 2),
        "n_points": n_points,
        "bbox_min_x": round(float(bbox_min[0]), 2),
        "bbox_min_y": round(float(bbox_min[1]), 2),
        "bbox_min_z": round(float(bbox_min[2]), 2),
        "bbox_max_x": round(float(bbox_max[0]), 2),
        "bbox_max_y": round(float(bbox_max[1]), 2),
        "bbox_max_z": round(float(bbox_max[2]), 2),
        "confidence": confidence,
    }


def _center_and_infer(engine, features, coords, centroid):
    """Run one chunk: center coords, infer, de-center xyz outputs."""
    centered = coords - centroid
    result = engine({"features": features, "coords": centered})
    # De-center any xyz predictions back to world coordinates
    centroid_t = torch.from_numpy(centroid).float()
    for k in list(result.keys()):
        if "xyz_pred" in k and isinstance(result[k], torch.Tensor):
            result[k] = result[k] + centroid_t
    return result


def _chunked_inference(engine, features, coords, chunk_size):
    """Run inference in chunks with per-chunk coordinate centering.

    Training centers coords per sample (subtract centroid) in _to_tensors.
    This replicates that behavior per chunk so the model sees the same
    coordinate scale it was trained on. xyz_pred outputs are de-centered
    back to world coordinates.
    """
    n = len(coords)

    # Ensure numpy arrays
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    if n <= chunk_size:
        centroid = coords.mean(axis=0).astype(np.float32)
        return _center_and_infer(engine, features, coords, centroid)

    all_results = {}
    result_keys = None
    n_chunks = (n + chunk_size - 1) // chunk_size
    logger.info(f"Chunked inference: {n:,} pts -> {n_chunks} chunks of {chunk_size:,}")

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk_coords = coords[i:end]
        chunk_feats = features[i:end]

        centroid = chunk_coords.mean(axis=0).astype(np.float32)
        chunk_result = _center_and_infer(engine, chunk_feats, chunk_coords, centroid)

        if result_keys is None:
            result_keys = list(chunk_result.keys())
            all_results = {k: [chunk_result[k]] for k in result_keys}
        else:
            for k in result_keys:
                all_results[k].append(chunk_result[k])

        torch.cuda.empty_cache()

    merged = {}
    for k in result_keys:
        vals = all_results[k]
        if isinstance(vals[0], torch.Tensor):
            merged[k] = torch.cat(vals, dim=0)
        else:
            merged[k] = vals
    return merged


def run_full_inference(cfg: DictConfig) -> None:
    from sklearn.cluster import DBSCAN

    from src.core.io import read_las, read_ply
    from src.core.io.las import write_las
    from src.core.preprocessing import Preprocessor
    from src.infer.seg_a import SegAInference
    from src.infer.seg_b import SegBColorInference, SegBGeomInference

    input_path = Path(cfg.task.input_path)
    output_dir = Path(cfg.task.get("output_dir", "outputs/pipeline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cfg = cfg.task.get("checkpoints", {})
    seg_a_ckpt = Path(ckpt_cfg.get("seg_a", "checkpoints/seg_a/best.pt"))
    seg_b_geom_ckpt = Path(
        ckpt_cfg.get("seg_b_geom", "checkpoints/seg_b_geom/best.pt")
    )
    seg_b_color_ckpt = Path(
        ckpt_cfg.get("seg_b_color", "checkpoints/seg_b_color/best.pt")
    )

    hazus_cfg = cfg.task.get("hazus", {})
    dbscan_eps = hazus_cfg.get("dbscan_eps", 1.5)
    dbscan_min_pts = hazus_cfg.get("dbscan_min_pts", 200)

    chunk_size = cfg.data.get("max_points", 100000)

    # ── Step 1: Load & Preprocess ────────────────────────────────────────
    logger.info(f"Loading input: {input_path}")
    if input_path.suffix.lower() in [".las", ".laz"]:
        raw_data = read_las(input_path)
    else:
        raw_data = read_ply(input_path)

    original_xyz = raw_data["xyz"].copy()
    n_total = len(original_xyz)
    logger.info(f"Loaded {n_total:,} points")

    preprocessor = Preprocessor(cfg)
    data = preprocessor.process(raw_data, voxelize_data=False)

    # ── Step 2: Seg-A ────────────────────────────────────────────────────
    if not seg_a_ckpt.exists():
        raise FileNotFoundError(f"Seg-A checkpoint required: {seg_a_ckpt}")

    logger.info("Running Seg-A inference...")
    seg_a_engine = SegAInference(cfg, seg_a_ckpt)

    seg_a_result = _chunked_inference(
        seg_a_engine, data["features"], data["xyz"], chunk_size
    )
    seg_a_labels = seg_a_result["predictions"].numpy()

    del seg_a_engine
    torch.cuda.empty_cache()

    seg_a_path = output_dir / "segmented.las"
    export_las(seg_a_path, xyz=original_xyz, labels=seg_a_labels)
    logger.info(f"Saved Seg-A output: {seg_a_path}")

    # ── Step 3: Filter buildings ─────────────────────────────────────────
    building_mask = seg_a_labels == BUILDING_CLASS
    n_building = building_mask.sum()
    logger.info(f"Building points: {n_building:,} / {n_total:,}")

    if n_building == 0:
        logger.warning("No building points found. Skipping Seg-B and HAZUS stages.")
        # Still produce merged output (same as segmented)
        write_las(output_dir / "merged.las", xyz=original_xyz, labels=seg_a_labels)
        logger.info("Pipeline complete (no buildings detected)")
        return

    building_xyz = original_xyz[building_mask]
    building_features = data["features"][building_mask]

    # ── Step 4: Seg-B-geom (optional) ────────────────────────────────────
    geom_xyz = None
    if seg_b_geom_ckpt.exists():
        logger.info("Running Seg-B-geom inference...")
        seg_b_geom_engine = SegBGeomInference(cfg, seg_b_geom_ckpt)
        seg_b_geom_result = _chunked_inference(
            seg_b_geom_engine, building_features, building_xyz, chunk_size
        )
        geom_xyz = seg_b_geom_result["xyz_pred"].numpy()

        geom_path = output_dir / "geom.las"
        export_las(geom_path, xyz=geom_xyz)
        logger.info(f"Saved Seg-B-geom output: {geom_path}")

        del seg_b_geom_engine
        torch.cuda.empty_cache()
    else:
        logger.info(f"Skipping Seg-B-geom (no checkpoint: {seg_b_geom_ckpt})")

    # ── Step 5: Seg-B-color (optional) ───────────────────────────────────
    color_rgb = None
    color_xyz = None
    if seg_b_color_ckpt.exists():
        logger.info("Running Seg-B-color inference...")
        seg_b_color_engine = SegBColorInference(cfg, seg_b_color_ckpt)
        seg_b_color_result = _chunked_inference(
            seg_b_color_engine, building_features, building_xyz, chunk_size
        )
        color_xyz = seg_b_color_result["xyz_pred"].numpy()
        color_rgb = seg_b_color_result["rgb_pred"].numpy()

        painted_path = output_dir / "painted.las"
        export_las(painted_path, xyz=color_xyz, rgb=color_rgb)
        logger.info(f"Saved Seg-B-color output: {painted_path}")

        del seg_b_color_engine
        torch.cuda.empty_cache()
    else:
        logger.info(f"Skipping Seg-B-color (no checkpoint: {seg_b_color_ckpt})")

    # ── Step 6: Rule-based HAZUS ─────────────────────────────────────────
    logger.info("Running HAZUS rule-based classification...")
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_pts)
    cluster_labels = clustering.fit_predict(building_xyz[:, :2])

    unique_clusters = sorted(set(cluster_labels) - {-1})
    logger.info(f"DBSCAN found {len(unique_clusters)} building clusters")

    constraints = FEMAConstraints()
    hazus_rows = []

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_xyz = building_xyz[mask]

        result = _classify_building_rule_based(cluster_xyz, constraints)
        result["cluster_id"] = int(cluster_id)
        hazus_rows.append(result)

    # Save CSV
    csv_path = output_dir / "hazus.csv"
    export_csv(csv_path, hazus_rows)
    logger.info(f"Saved HAZUS CSV: {csv_path} ({len(hazus_rows)} buildings)")

    # Build per-point arrays for building points
    hazus_point_cluster = cluster_labels.copy().astype(np.int32)
    hazus_point_cluster[cluster_labels == -1] = 0
    hazus_point_cluster[cluster_labels >= 0] += 1  # shift so 0 = noise

    hazus_point_occ = np.zeros(len(building_xyz), dtype=np.uint8)
    # Map cluster_id → occupancy_id from classification results
    cluster_to_occ = {r["cluster_id"]: r["occupancy_id"] for r in hazus_rows}
    for cid, occ_id in cluster_to_occ.items():
        hazus_point_occ[cluster_labels == cid] = occ_id

    # Save HAZUS LAS — use extra_dims (cluster IDs can exceed LAS 5-bit limit)
    hazus_las_path = output_dir / "hazus.las"
    write_las(
        hazus_las_path,
        xyz=building_xyz,
        labels=np.full(len(building_xyz), BUILDING_CLASS, dtype=np.int32),
        extra_dims={
            "cluster_id": hazus_point_cluster,
            "occupancy": hazus_point_occ,
        },
    )
    logger.info(f"Saved HAZUS LAS: {hazus_las_path}")

    # ── Step 7: Merged output ────────────────────────────────────────────
    logger.info("Building merged output...")
    merged_xyz = original_xyz.copy()
    merged_labels = seg_a_labels.copy()
    merged_rgb = raw_data.get("rgb")

    # Overwrite building geometry if available
    if geom_xyz is not None and len(geom_xyz) == n_building:
        merged_xyz[building_mask] = geom_xyz

    # Add color from Seg-B-color for building points
    if color_rgb is not None and len(color_rgb) == n_building:
        if merged_rgb is None:
            merged_rgb = np.zeros((n_total, 3), dtype=np.float32)
        merged_rgb[building_mask] = color_rgb

    # Build HAZUS extra dims for full point cloud
    hazus_cluster_full = np.zeros(n_total, dtype=np.int32)
    hazus_occ_full = np.zeros(n_total, dtype=np.uint8)
    building_indices = np.where(building_mask)[0]
    for i, idx in enumerate(building_indices):
        hazus_cluster_full[idx] = hazus_point_cluster[i]
        hazus_occ_full[idx] = hazus_point_occ[i]

    merged_path = output_dir / "merged.las"
    write_las(
        merged_path,
        xyz=merged_xyz,
        rgb=merged_rgb,
        labels=merged_labels,
        extra_dims={
            "cluster_id": hazus_cluster_full,
            "occupancy": hazus_occ_full,
        },
    )
    logger.info(f"Saved merged output: {merged_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Pipeline complete!")
    logger.info(
        f"  segmented.las : {n_total:,} points, {len(np.unique(seg_a_labels))} classes"
    )
    logger.info(f"  geom.las      : {'OK' if geom_xyz is not None else 'SKIPPED'}")
    logger.info(f"  painted.las   : {'OK' if color_rgb is not None else 'SKIPPED'}")
    logger.info(f"  hazus.csv     : {len(hazus_rows)} buildings classified")
    logger.info(f"  hazus.las     : {n_building:,} building points")
    logger.info(f"  merged.las    : {n_total:,} points (combined)")
    logger.info("=" * 50)
