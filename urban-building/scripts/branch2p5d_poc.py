"""2.5D branch PoC: provincial satellite+DEM Potree cloud -> schema v1 building records.
CPU only. No model. Reads the whole octree.bin as a flat 37-byte record array.

usage: uv run scripts/branch2p5d_poc.py --province nonthaburi [--bbox xmin ymin xmax ymax] [--min-cells 4]
"""
import argparse, json, os, sys, time, math
import numpy as np
import requests
from scipy import ndimage
from scipy.spatial import cKDTree

BASE = os.environ.get("POINTCLOUD_BASE_URL", "").rstrip("/") + "/{p}/"   # export POINTCLOUD_BASE_URL=https://<host>/pointcloud/provinces (never commit the host)
CELL = 5.0          # native grid, m (3857 units)
CTX = 25.0          # context / DTM grid, m
STORY_H = 3.2       # m, TH residential default
CLS = dict(unc=1, ground=2, lowveg=3, veg=5, bld=6, water=9, rail=10, road=11)

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def fetch(url, path):
    if os.path.exists(path):
        want = int(requests.head(url, timeout=30).headers.get("content-length", -1))
        if want == os.path.getsize(path): log("cached", path); return
    log("download", url)
    with requests.get(url, stream=True, timeout=60) as r, open(path, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 22): f.write(chunk)

def load_points(raw_dir, meta):
    dt = np.dtype([("pos", "<i4", (3,)), ("inten", "<u2"), ("ret", "u1"), ("nret", "u1"), ("cflag", "u1"),
                   ("cls", "u1"), ("ud", "u1"), ("sa", "<i2"), ("psid", "<u2"), ("gps", "<f8"), ("rgb", "<u2", (3,))])
    assert dt.itemsize == sum(a["size"] for a in meta["attributes"]) == 37, "attribute layout changed"
    r = np.fromfile(os.path.join(raw_dir, "octree.bin"), dtype=dt)
    assert len(r) == meta["points"], (len(r), meta["points"])
    xyz = r["pos"].astype(np.float64) * np.array(meta["scale"]) + np.array(meta["offset"])
    return xyz, r["cls"].copy(), r["rgb"].astype(np.float32) / 65535.0

def cell_percentile(cell_id, z, n_cells, q):
    # per-cell q-quantile via sort; returns (values, counts) with nan for empty cells
    order = np.lexsort((z, cell_id)); c = cell_id[order]; zs = z[order]
    uniq, start, cnt = np.unique(c, return_index=True, return_counts=True)
    idx = start + np.floor(q * (cnt - 1)).astype(np.int64)
    out = np.full(n_cells, np.nan); out[uniq] = zs[idx]
    cntv = np.zeros(n_cells, np.int32); cntv[uniq] = cnt
    return out, cntv

def fill_nearest(a):
    m = np.isnan(a)
    if not m.any(): return a
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--province", default="nonthaburi")
    ap.add_argument("--bbox", nargs=4, type=float, default=None, help="xmin ymin xmax ymax in 3857")
    ap.add_argument("--min-cells", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cell", default="auto", help="grid size in 3857 units, or 'auto' = median NN spacing of a 200k sample")
    a = ap.parse_args()
    global CELL
    if not os.environ.get("POINTCLOUD_BASE_URL"):
        sys.exit("set POINTCLOUD_BASE_URL=https://<host>/pointcloud/provinces (kept out of git)")
    p = a.province; base = BASE.format(p=p)
    raw = f"data/raw/branch2p5d/{p}"; out = a.out or f"output/branch2p5d/{p}"
    os.makedirs(raw, exist_ok=True); os.makedirs(out, exist_ok=True)
    fetch(base + "metadata.json", f"{raw}/metadata.json"); meta = json.load(open(f"{raw}/metadata.json"))
    fetch(base + "octree.bin", f"{raw}/octree.bin")
    assert meta.get("encoding", "DEFAULT") == "DEFAULT"

    t0 = time.time(); xyz, cls, rgb = load_points(raw, meta); log(f"points {len(xyz):,} in {time.time()-t0:.1f}s")
    if a.bbox:
        x0, y0, x1, y1 = a.bbox; keep = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
        xyz, cls, rgb = xyz[keep], cls[keep], rgb[keep]; log(f"bbox keep {len(xyz):,}")
    xmin, ymin = xyz[:, :2].min(0); xmax, ymax = xyz[:, :2].max(0)
    lat = math.degrees(math.atan(math.sinh(((ymin + ymax) / 2) / 6378137.0)))
    m_per_unit = math.cos(math.radians(lat))          # 3857 -> ground metres
    log(f"extent {xmax-xmin:.0f} x {ymax-ymin:.0f} units, lat {lat:.2f}, m/unit {m_per_unit:.4f}")
    hist = {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))}; log("classes", hist)
    # ---- spacing audit on a sample: NN distance + lattice check
    rng = np.random.default_rng(0); s = rng.choice(len(xyz), min(200_000, len(xyz)), replace=False)
    nnd = cKDTree(xyz[:, :2]).query(xyz[s, :2], k=2)[0][:, 1]
    sp_pct = {q: float(np.percentile(nnd, q)) for q in (5, 25, 50, 75, 95)}
    log("NN spacing units p5/25/50/75/95:", {k: round(v, 2) for k, v in sp_pct.items()})
    if a.cell == "auto":
        CELL = float(max(1.0, round(sp_pct[50])))
    else:
        CELL = float(a.cell)
    log(f"CELL = {CELL} units ({CELL*m_per_unit:.1f} m)")
    fx = np.round(xyz[s, 0] % CELL, 1); log("x mod CELL distinct values (lattice hint):", len(np.unique(fx)))

    # ---- 5 m grid indices
    W = int((xmax - xmin) // CELL) + 1; H = int((ymax - ymin) // CELL) + 1
    ci = ((xyz[:, 0] - xmin) // CELL).astype(np.int64); cj = ((xyz[:, 1] - ymin) // CELL).astype(np.int64)
    cid = cj * W + ci; n5 = H * W; log(f"grid5 {W}x{H} = {n5:,} cells")
    # ---- 25 m grid
    Wc = int((xmax - xmin) // CTX) + 1; Hc = int((ymax - ymin) // CTX) + 1
    ki = ((xyz[:, 0] - xmin) // CTX).astype(np.int64); kj = ((xyz[:, 1] - ymin) // CTX).astype(np.int64); kid = kj * Wc + ki; n25 = Hc * Wc

    # ---- DTM: class 1+2 low points, 20th pct per 25 m cell, nearest-fill, 3x3 median
    gm = (cls == CLS["unc"]) | (cls == CLS["ground"])
    dtm, dtm_n = cell_percentile(kid[gm], xyz[gm, 2], n25, 0.20)
    dtm = ndimage.median_filter(fill_nearest(dtm.reshape(Hc, Wc)), size=3)
    log(f"dtm cells with data {int((dtm_n>0).sum()):,}/{n25:,}")

    # ---- DSM at 5 m: max z per cell (all classes)
    dsm = np.full(n5, -np.inf); np.maximum.at(dsm, cid, xyz[:, 2]); dsm[np.isinf(dsm)] = np.nan; dsm = dsm.reshape(H, W)

    # ---- building mask + components
    bm = np.zeros(n5, bool); bm[cid[cls == CLS["bld"]]] = True; bm = bm.reshape(H, W)
    lab, nlab = ndimage.label(bm, structure=np.ones((3, 3), int)); log(f"components raw {nlab:,}")
    sizes = np.bincount(lab.ravel()); small = sizes < a.min_cells; small[0] = True
    lab[small[lab]] = 0
    ids, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W); n = len(ids) - 1
    log(f"components >= {a.min_cells} cells: {n:,}")
    if n == 0:
        json.dump({"province": p, "points": int(len(xyz)), "class_hist": hist, "cell": CELL, "spacing_pct": sp_pct, "components_raw": int(nlab), "components_kept": 0},
                  open(f"{out}/summary.json", "w"), indent=1)
        log("no components at this cell size; see summary.json"); return

    # ---- per-component geometry
    L = lab.ravel(); area_cells = np.bincount(L, minlength=n + 1)[1:]
    # exposed edges: neighbour differs (incl. background)
    edges = np.zeros(n + 1, np.int64)
    for dy, dx in ((0, 1), (1, 0)):
        A = lab[:H - dy, :W - dx]; B = lab[dy:, dx:]
        diff = A != B
        edges += np.bincount(A[diff], minlength=n + 1) + np.bincount(B[diff], minlength=n + 1)
    # border cells of the raster count as exposed
    edges += np.bincount(lab[0, :], minlength=n + 1) + np.bincount(lab[-1, :], minlength=n + 1) + np.bincount(lab[:, 0], minlength=n + 1) + np.bincount(lab[:, -1], minlength=n + 1)
    perim_u = edges[1:] * CELL
    jj, ii = np.nonzero(lab); l = lab[jj, ii]
    cx = np.bincount(l, weights=ii, minlength=n + 1)[1:] / area_cells; cy = np.bincount(l, weights=jj, minlength=n + 1)[1:] / area_cells
    # elongation from 2x2 covariance of cell coords
    sxx = np.bincount(l, weights=ii * ii, minlength=n + 1)[1:] / area_cells - cx ** 2
    syy = np.bincount(l, weights=jj * jj, minlength=n + 1)[1:] / area_cells - cy ** 2
    sxy = np.bincount(l, weights=ii * jj, minlength=n + 1)[1:] / area_cells - cx * cy
    tr = sxx + syy; det = sxx * syy - sxy ** 2; disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    e1 = tr / 2 + disc; e2 = np.maximum(tr / 2 - disc, 1e-9); elong = np.sqrt(e2 / e1)
    area_m2 = area_cells * CELL ** 2 * m_per_unit ** 2; perim_m = perim_u * m_per_unit
    compact = 4 * np.pi * area_m2 / np.maximum(perim_m ** 2, 1e-9)
    cxu = xmin + (cx + 0.5) * CELL; cyu = ymin + (cy + 0.5) * CELL     # centroid, 3857 units

    # ---- height: p90 of DSM inside component minus DTM at centroid
    dsm_flat = dsm.ravel(); valid = ~np.isnan(dsm_flat) & (L > 0)
    p90 = cell_percentile(L[valid], dsm_flat[valid], n + 1, 0.90)[0][1:]
    dtm_at = dtm[np.clip((cy * CELL // CTX).astype(int), 0, Hc - 1), np.clip((cx * CELL // CTX).astype(int), 0, Wc - 1)]
    height = np.maximum(p90 - dtm_at, 0.0)
    # relief gate: if the "DSM" carries no building relief province-wide, height is not observed
    relief_p95 = float(np.nanpercentile(height, 95)); has_relief = relief_p95 >= 3.0
    log(f"height p95 = {relief_p95:.2f} m -> {'relief present' if has_relief else 'NO relief: height nulled, conf 0'}")
    h_conf_val = 0.5 if has_relief else 0.0
    if not has_relief: height = np.full(n, np.nan)
    stories = np.maximum(1, np.round(np.nan_to_num(height) / STORY_H))
    band = np.where(~np.isfinite(height), "unknown", np.where(stories <= 1, "1", np.where(stories <= 3, "2-3", np.where(stories <= 7, "4-7", "8+"))))

    # ---- context rasters at 25 m
    def presence(c):
        g = np.zeros(n25, bool); g[kid[cls == c]] = True; return g.reshape(Hc, Wc)
    road, water, veg = presence(CLS["road"]), presence(CLS["water"]), presence(CLS["veg"])
    def dist_at(g):
        d = ndimage.distance_transform_edt(~g) * CTX * m_per_unit if g.any() else np.full(g.shape, np.nan)
        return d[np.clip((cy * CELL // CTX).astype(int), 0, Hc - 1), np.clip((cx * CELL // CTX).astype(int), 0, Wc - 1)]
    road_d, water_d = dist_at(road), dist_at(water)
    veg_frac = ndimage.uniform_filter(veg.astype(float), size=5)[np.clip((cy * CELL // CTX).astype(int), 0, Hc - 1), np.clip((cx * CELL // CTX).astype(int), 0, Wc - 1)]
    bld25 = ndimage.uniform_filter((lab > 0).astype(float), size=int(round(200 / CELL)))  # 200 m window on 5 m grid
    built_frac = bld25[np.clip(cy.astype(int), 0, H - 1), np.clip(cx.astype(int), 0, W - 1)]
    tree = cKDTree(np.c_[cxu, cyu]); nn = tree.query(np.c_[cxu, cyu], k=2)[0][:, 1] * m_per_unit
    ncount = np.array([len(x) - 1 for x in tree.query_ball_point(np.c_[cxu, cyu], r=100 / m_per_unit)])

    # ---- rgb per component from class-6 points
    pb = cls == CLS["bld"]; pl = lab.ravel()[cid[pb]]; keep = pl > 0
    rs = np.zeros((n + 1, 3)); np.add.at(rs, pl[keep], rgb[pb][keep]); rc = np.bincount(pl[keep], minlength=n + 1)
    rgb_mean = rs[1:] / np.maximum(rc[1:, None], 1)
    r2 = np.zeros((n + 1, 3)); np.add.at(r2, pl[keep], rgb[pb][keep] ** 2)
    rgb_std = np.sqrt(np.maximum(r2[1:] / np.maximum(rc[1:, None], 1) - rgb_mean ** 2, 0))

    # ---- confidence per schema v1
    fp_conf = np.clip(1 - 2.5 * perim_m / np.maximum(area_m2, 1), 0, 1)
    h_conf = np.full(n, h_conf_val)
    fp_conf = np.where(area_cells == 1, np.minimum(fp_conf, 0.25), fp_conf)   # singletons: one pixel, keep but flag

    # ---- outputs
    plan = np.where(compact > 0.75, "PLFSQ", np.where(elong < 0.35, "PLFR", np.where(compact < 0.4, "PLFI", "PLFR")))
    import csv
    cols = ["building_id", "cx_3857", "cy_3857", "area_m2", "perim_m", "compactness", "elongation", "plan_shape_gem", "height_m", "height_conf",
            "story_band", "footprint_conf", "road_distance_m", "water_distance_m", "vegetation_fraction_50m", "built_fraction_100m",
            "neighbor_count_100m", "nearest_building_m", "rgb_r", "rgb_g", "rgb_b", "rgb_std_mean", "n_cells", "n_points"]
    bid = [f"{p}-{int(cxu[i])}-{int(cyu[i])}" for i in range(n)]
    rows = zip(bid, cxu, cyu, area_m2, perim_m, compact, elong, plan, height, h_conf, band, fp_conf, road_d, water_d, veg_frac, built_frac,
               ncount, nn, rgb_mean[:, 0], rgb_mean[:, 1], rgb_mean[:, 2], rgb_std.mean(1), area_cells, rc[1:])
    with open(f"{out}/buildings.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([x if isinstance(x, str) else (round(float(x), 4) if np.isfinite(x) else "") for x in r])
    cell = lambda v, c, m: {"value": (None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v), "conf": float(c), "source": "branch2p5d", "method": m}
    with open(f"{out}/records.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "identity": {"building_id": bid[i], "province": p, "source_dataset": meta.get("name"), "source_kind": "satellite_dem_raster", "crs": "EPSG:3857", "branch": "branch2p5d"},
                "geometry": {"footprint_area_m2": cell(float(area_m2[i]), fp_conf[i], "class6_cells_x25_cos2lat"), "perimeter_m": cell(float(perim_m[i]), fp_conf[i], "exposed_edges"),
                             "compactness": cell(float(compact[i]), fp_conf[i], "4piA/P2"), "elongation": cell(float(elong[i]), fp_conf[i], "cov_eig"), "plan_shape_gem": cell(str(plan[i]), 0.4, "compact+elong_rule"),
                             "centroid": cell([float(cxu[i]), float(cyu[i])], 1.0, "cell_mean")},
                "height": {"height_m": cell(float(height[i]), h_conf[i], "p90_dsm_minus_dtm20" if has_relief else "not_observed_flat_dsm"), "story_band": cell(str(band[i]), 0.4 if has_relief else 0.0, f"height/{STORY_H}")},
                "context": {"neighbor_count_100m": cell(int(ncount[i]), 0.8, "centroid_kdtree"), "nearest_building_m": cell(float(nn[i]), 0.8, "centroid_kdtree"),
                            "built_fraction_100m": cell(float(built_frac[i]), 0.8, "mask_window"), "road_distance_m": cell(float(road_d[i]), 0.6, "class11_edt"),
                            "water_distance_m": cell(float(water_d[i]), 0.6, "class9_edt"), "vegetation_fraction_50m": cell(float(veg_frac[i]), 0.6, "class5_window")},
                "appearance": {"rgb_mean": cell([float(x) for x in rgb_mean[i]], 0.9, "class6_points"), "rgb_std": cell([float(x) for x in rgb_std[i]], 0.9, "class6_points")},
            }) + "\n")
    # footprints as GeoJSON via rasterio.features.shapes
    try:
        from rasterio import features
        from rasterio.transform import from_origin
        tf = from_origin(xmin, ymin + H * CELL, CELL, CELL)  # north-up: row 0 = top
        shp = features.shapes(lab[::-1].astype(np.int32), mask=(lab[::-1] > 0), transform=tf)
        feats = [{"type": "Feature", "properties": {"building_id": bid[int(v) - 1], "area_m2": round(float(area_m2[int(v) - 1]), 1), "height_m": round(float(height[int(v) - 1]), 2), "story_band": str(band[int(v) - 1])}, "geometry": g} for g, v in shp]
        json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}}, "features": feats}, open(f"{out}/footprints.geojson", "w"))
        log(f"geojson features {len(feats):,}")
    except Exception as e: log("geojson skipped:", e)

    summary = {"province": p, "points": int(len(xyz)), "class_hist": hist, "extent_units": [float(xmin), float(ymin), float(xmax), float(ymax)], "lat": lat, "m_per_unit": m_per_unit,
               "grid5": [W, H], "components_raw": int(nlab), "components_kept": int(n), "min_cells": a.min_cells,
               "cell_units": CELL, "spacing_pct": sp_pct, "has_relief": bool(has_relief), "relief_p95_m": relief_p95, "singletons": int((area_cells == 1).sum()),
               "area_m2_pct": {q: float(np.percentile(area_m2, q)) for q in (5, 25, 50, 75, 95)}, "height_m_pct": {q: float(np.nanpercentile(height, q)) if has_relief else None for q in (5, 25, 50, 75, 95)},
               "story_band_hist": {k: int(v) for k, v in zip(*np.unique(band, return_counts=True))},
               "dtm_z_pct": {q: float(np.nanpercentile(dtm, q)) for q in (5, 50, 95)}, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1); log(json.dumps(summary, indent=1))

    # ---- plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].hist(area_m2, bins=np.logspace(2, 5, 60)); ax[0].set_xscale("log"); ax[0].set_title("footprint area m2 (log)")
    ax[1].hist(height, bins=np.linspace(0, 40, 80)); ax[1].set_title("height m (p90 dsm - dtm20)")
    ax[2].hist(compact, bins=50); ax[2].set_title("compactness")
    fig.tight_layout(); fig.savefig(f"{out}/hist.png", dpi=110); plt.close(fig)
    step = max(1, W // 2000)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow((lab[::step, ::step] > 0), origin="lower", cmap="gray_r"); ax[0].set_title("building mask (5 m)")
    im = ax[1].imshow(dtm, origin="lower", cmap="terrain"); ax[1].set_title("DTM 25 m (class-1 p20)"); fig.colorbar(im, ax=ax[1], shrink=0.7)
    fig.tight_layout(); fig.savefig(f"{out}/overview.png", dpi=90); plt.close(fig)
    log("done ->", out)

if __name__ == "__main__":
    main()
