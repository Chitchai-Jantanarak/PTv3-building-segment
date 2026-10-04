"""Region-growing over-segmentation + weak-label rule precision on a real SensatUrban tile.
CPU only. usage: python scripts/overseg_sensat_check.py --tile data/processed/sensat/val/cambridge_block_32.npz [--voxel 0.2]
"""
import argparse, json, os, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SENSAT = ["Ground", "High Vegetation", "Buildings", "Walls", "Bridge", "Parking", "Rail", "Traffic Roads", "Street Furniture", "Cars", "Footpath", "Bikes", "Water"]
COLLAPSE = {0: 0, 5: 0, 7: 0, 10: 0, 2: 1, 3: 1}   # ground=0 building=1 other=2
C3 = ["ground", "building", "other"]

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True); ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--theta", type=float, default=12.0)
    ap.add_argument("--cmax", type=float, default=0.03); ap.add_argument("--min-pts", type=int, default=30)
    ap.add_argument("--dz", type=float, default=0.5, help="max |dz| per edge, m"); ap.add_argument("--radius", type=float, default=1.0, help="max edge length, m")
    ap.add_argument("--out", default="output/overseg_check"); ap.add_argument("--rules", default="v1", choices=["v0", "v1"]); ap.add_argument("--tag", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); name = os.path.basename(a.tile).split(".")[0]
    d = np.load(a.tile); xyz = d["xyz"].astype(np.float64); lab13 = d["labels"].astype(np.int64); relz = d["features"][:, 3].astype(np.float64)
    rgb = d["rgb"][d["indices"]].astype(np.float64) if "rgb" in d.files and "indices" in d.files and len(d["rgb"]) != len(xyz) else (d["rgb"].astype(np.float64) if "rgb" in d.files else None)
    if rgb is not None and rgb.max() > 1.5: rgb = rgb / 255.0
    log(f"{name}: {len(xyz):,} pts, extent {np.ptp(xyz[:,0]):.0f} x {np.ptp(xyz[:,1]):.0f} m")
    # voxel downsample: first point per voxel
    key = np.floor((xyz - xyz.min(0)) / a.voxel).astype(np.int64); _, first = np.unique(key, axis=0, return_index=True)
    xyz, lab13, relz = xyz[first], lab13[first], relz[first]; N = len(xyz); log(f"voxel {a.voxel} -> {N:,} pts")
    if rgb is not None: rgb = rgb[first]
    lab3 = np.full(N, 2);
    for k, v in COLLAPSE.items(): lab3[lab13 == k] = v

    # features from kNN covariance
    t = cKDTree(xyz); nd, nn = t.query(xyz, k=a.k + 1); nd, nn = nd[:, 1:], nn[:, 1:]
    nbr = xyz[nn]; c = nbr - nbr.mean(1, keepdims=True); cov = np.einsum("nki,nkj->nij", c, c) / a.k
    ev, evec = np.linalg.eigh(cov); l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
    nrm = evec[:, :, 0]; nrm *= np.sign(nrm[:, 2:3] + 1e-9)
    planarity = (l2 - l3) / (l1 + 1e-9); linearity = (l1 - l2) / (l1 + 1e-9); scatter = l3 / (l1 + 1e-9)
    curv = l3 / (l1 + l2 + l3 + 1e-9); vert = 1 - np.abs(nrm[:, 2]); valid = nd < 2.5
    log("features done")

    # region growing as graph: keep kNN edges whose endpoints agree in normal and are both low-curvature
    ei = np.repeat(np.arange(N), a.k); ej = nn.ravel(); ok = valid.ravel() & (ei < ej)
    ei, ej = ei[ok], ej[ok]
    cos_t = np.cos(np.radians(a.theta))
    # edge criteria: normal agreement, low curvature both ends, height continuity (kills roof->ground leaks at eaves), short edge
    dz_ok = np.abs(xyz[ei, 2] - xyz[ej, 2]) < a.dz
    keep = (np.abs((nrm[ei] * nrm[ej]).sum(1)) > cos_t) & (curv[ei] < a.cmax * 3) & (curv[ej] < a.cmax * 3) & dz_ok & (nd.ravel()[ok] < a.radius)
    # per-class feature distributions -> v1 threshold derivation
    dist = {}
    for c in range(3):
        mc = lab3 == c
        dist[C3[c]] = {f: [round(float(np.percentile(v[mc], q)), 3) for q in (10, 25, 50, 75, 90)] for f, v in
                       [("planarity", planarity), ("scattering", scatter), ("linearity", linearity), ("verticality", vert), ("curvature", curv), ("rel_z", relz)]}
    log("per-class feature p10/25/50/75/90:", json.dumps(dist))
    g = coo_matrix((np.ones(keep.sum()), (ei[keep], ej[keep])), shape=(N, N))
    _, seg = connected_components(g, directed=False)
    # seeds: a segment must contain at least one low-curvature point; else unassigned
    has_seed = np.zeros(seg.max() + 1, bool); has_seed[seg[curv <= a.cmax]] = True
    seg = np.where(has_seed[seg], seg, -1)
    ids, cnt = np.unique(seg[seg >= 0], return_counts=True); small = ids[cnt < a.min_pts]; seg[np.isin(seg, small)] = -1
    ids, inv = np.unique(seg, return_inverse=True); seg = inv - (1 if ids[0] == -1 else 0)  # -1 stays -1 if present
    if ids[0] != -1: seg = inv
    S = seg.max() + 1; m = seg >= 0; log(f"segments {S:,}, coverage {m.mean():.3f}")

    # purity per segment (3-class and 13-class)
    def purity(lab, ncls):
        cm = np.zeros((S, ncls), np.int64); np.add.at(cm, (seg[m], lab[m]), 1)
        return cm.max(1) / cm.sum(1), cm.argmax(1), cm.sum(1)
    pur3, maj3, sz = purity(lab3, 3); pur13, maj13, _ = purity(lab13, 13)
    w = sz / sz.sum()
    res = {"tile": name, "n_pts": int(N), "segments": int(S), "coverage": float(m.mean()),
           "purity3_weighted": float((pur3 * w).sum()), "purity13_weighted": float((pur13 * w).sum()),
           "segs_below_0.9_frac_pts": float(w[pur3 < 0.9].sum()), "purity3_p10": float(np.percentile(pur3, 10))}
    # per true class: fraction of its points inside segments whose majority is that class (segment-level recall) and coverage
    rec = {}
    for c in range(3):
        pts = lab3 == c; rec[C3[c]] = {"pts": int(pts.sum()), "covered": float(m[pts].mean()), "in_majority_seg": float((m[pts] & (maj3[np.maximum(seg, 0)] == c)[pts]).mean())}
    res["per_class"] = rec
    # over-seg: segments per class-connected object unknown (no instances); report segments per 1000 m2 of building
    log(json.dumps(res, indent=1))

    # ---- weak-label rules v0 at segment level
    def segmed(x, q=0.5):
        out = np.zeros(S);
        for s in range(S): pass
        order = np.argsort(seg[m]); ss = seg[m][order]; xs = x[m][order]
        u, st, ct = np.unique(ss, return_index=True, return_counts=True); idx = st + np.floor(q * (ct - 1)).astype(int)
        xs_sorted = np.concatenate([np.sort(xs[i:i + n]) for i, n in zip(st, ct)])
        out[u] = xs_sorted[idx]; return out
    f_plan, f_lin, f_sca, f_vert = segmed(planarity), segmed(linearity), segmed(scatter), segmed(vert)
    f_rzmed, f_rzmax = segmed(relz), segmed(relz, 0.95)
    # 2D area via bbox of segment (cheap proxy for hull)
    xmin = np.full(S, np.inf); xmax = np.full(S, -np.inf); ymin = np.full(S, np.inf); ymax = np.full(S, -np.inf)
    np.minimum.at(xmin, seg[m], xyz[m, 0]); np.maximum.at(xmax, seg[m], xyz[m, 0]); np.minimum.at(ymin, seg[m], xyz[m, 1]); np.maximum.at(ymax, seg[m], xyz[m, 1])
    f_area = np.maximum(xmax - xmin, 0.1) * np.maximum(ymax - ymin, 0.1)
    # adjacency: segment pairs sharing kNN edges
    adj_i, adj_j = seg[ei], seg[ej]; ok2 = (adj_i >= 0) & (adj_j >= 0) & (adj_i != adj_j)
    pairs = np.unique(np.c_[np.minimum(adj_i[ok2], adj_j[ok2]), np.maximum(adj_i[ok2], adj_j[ok2])], axis=0)
    is_roof = (f_plan >= 0.6) & (f_vert <= 0.4) & (f_rzmed >= 2.0) & (f_area >= 6)
    touches_roof = np.zeros(S, bool); touches_roof[pairs[:, 0][is_roof[pairs[:, 1]]]] = True; touches_roof[pairs[:, 1][is_roof[pairs[:, 0]]]] = True
    f_curv = segmed(curv)
    V = a.rules  # v0 = contract as written; v1 = thresholds derived from tile-32 per-class percentiles (see feature_dist)
    if V == "v0":
        rules = {
            "ground":      (0, (f_plan >= 0.70) & (f_vert <= 0.20) & (f_rzmed <= 0.30)),
            "roof":        (1, is_roof),
            "wall":        (1, (f_plan >= 0.50) & (f_vert >= 0.80) & (f_rzmax >= 2.0) & touches_roof),
            "vegetation":  (2, (f_sca >= 0.15) & (f_plan <= 0.35) & (f_rzmed >= 0.5)),
            "vehicle":     (2, (f_plan >= 0.40) & (f_rzmed >= 0.4) & (f_rzmax <= 3.0) & (f_area <= 15)),
            "bridge_deck": (2, (f_plan >= 0.70) & (f_vert <= 0.20) & (f_rzmed >= 2.0) & (f_area >= 200)),
            "pole_line":   (2, (f_lin >= 0.80) & (f_rzmax >= 2.0)),
        }
    else:
        # edge drop: over ALL kNN edges leaving a segment (to other segments or unassigned points), share whose far end is >= 1.5 m lower
        # boundary points = segment points with at least one kNN edge leaving the segment; drop measured in a 3 m ball (eave scale)
        si, sj = seg[ei], seg[ej]; bnd = np.zeros(N, bool); bnd[ei[(si >= 0) & (si != sj)]] = True; bnd[ej[(sj >= 0) & (si != sj)]] = True
        rng_b = np.random.default_rng(1); bidx = np.nonzero(bnd)[0]
        # cap 150 boundary points per segment
        order = rng_b.permutation(bidx); ssub = seg[order]; keep_b = np.zeros(len(order), bool); seen = np.zeros(S, int)
        for q, s_ in enumerate(ssub):
            if seen[s_] < 150: keep_b[q] = True; seen[s_] += 1
        bidx = order[keep_b]
        xy_tree = cKDTree(xyz[:, :2]); balls = xy_tree.query_ball_point(xyz[bidx, :2], r=3.0)
        drop_pt = np.array([(relz[b] < relz[i] - 1.5).mean() if len(b) else 0.0 for i, b in zip(bidx, balls)])
        f_drop = np.zeros(S); np.add.at(f_drop, seg[bidx], drop_pt); f_drop /= np.maximum(np.bincount(seg[bidx], minlength=S), 1)
        is_roof = (f_plan >= 0.45) & (f_vert <= 0.40) & (f_curv <= 0.03) & (f_rzmed >= 2.0) & (f_area >= 6) & (f_drop >= 0.05)
        touches_roof = np.zeros(S, bool); touches_roof[pairs[:, 0][is_roof[pairs[:, 1]]]] = True; touches_roof[pairs[:, 1][is_roof[pairs[:, 0]]]] = True
        rules = {
            "ground":      (0, (f_plan >= 0.45) & (f_vert <= 0.15) & (f_curv <= 0.02) & (f_rzmed <= 0.30)),
            "roof":        (1, is_roof),
            "wall":        (1, (f_vert >= 0.75) & (f_rzmax >= 2.0) & touches_roof),
            "vegetation":  (2, (f_sca >= 0.06) & (f_curv >= 0.03) & (f_plan <= 0.55) & (f_rzmed >= 0.5)),
            "elevated_ground": (0, (f_plan >= 0.45) & (f_vert <= 0.15) & (f_curv <= 0.02) & (f_rzmed >= 2.0) & (f_area >= 50) & (f_drop < 0.02)),
        }
        log("roof candidates by drop test: with drop %d, without %d" % (int(is_roof.sum()), int(((f_plan >= 0.45) & (f_vert <= 0.40) & (f_curv <= 0.03) & (f_rzmed >= 2.0) & (f_area >= 6) & (f_drop < 0.05)).sum())))
    table = {}
    for r, (cls, fire) in rules.items():
        n_f = int(fire.sum()); prec = float((maj3[fire] == cls).mean()) if n_f else None
        pts_f = int(sz[fire].sum()); prec_pts = float((sz[fire] * (maj3[fire] == cls)).sum() / max(pts_f, 1)) if n_f else None
        table[r] = {"class": C3[cls], "fires": n_f, "precision_seg": prec, "precision_pts": prec_pts, "pts": pts_f,
                    "decision": None if prec is None else ("keep" if prec >= 0.9 else "adjust" if prec >= 0.75 else "drop")}
    # decide policy -> label per segment
    fires_cls = np.full((S, 3), False)
    for r, (cls, fire) in rules.items(): fires_cls[:, cls] |= fire
    n_cls = fires_cls.sum(1); teach = np.where(n_cls == 1, fires_cls.argmax(1), -1)
    lbl = teach[np.maximum(seg, 0)]; lbl[~m] = -1
    if V != "v0":
        # residual rule, point level: unassigned + scattered + elevated -> other (vegetation, clutter). Never touches segment labels.
        # diagnostic only (NOT applied): can any local-geometry residual rule isolate vegetation from building clutter?
        for tag_, cond in [("loose", (curv >= 0.03) & (scatter >= 0.06)), ("tight", (curv >= 0.08) & (scatter >= 0.15)), ("very_tight", (curv >= 0.12) & (scatter >= 0.25))]:
            resid_ = (~m) & cond & (relz >= 0.5)
            log(f"residual diag {tag_}: {int(resid_.sum()):,} pts, precision(other) {float((lab3[resid_] == 2).mean()) if resid_.any() else None}")
        if rgb is not None:
            exg = 2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]   # excess green
            for thr in (0.05, 0.10, 0.15, 0.20):
                for tag_, cond in [("unassigned", ~m), ("unassigned+curv", (~m) & (curv >= 0.03))]:
                    rv = cond & (relz >= 0.5) & (exg >= thr)
                    log(f"greenness diag {tag_} exg>={thr}: {int(rv.sum()):,} pts, precision(other) {float((lab3[rv] == 2).mean()) if rv.any() else None}, recall(other pts) {float(rv[lab3 == 2].mean()):.3f}")
        resid = (~m) & (curv >= 0.03) & (scatter >= 0.06) & (relz >= 0.5); res_prec = float((lab3[resid] == 2).mean()) if resid.any() else None
    cov = float((lbl >= 0).mean()); acc = float((lbl[lbl >= 0] == lab3[lbl >= 0]).mean())
    cm = np.zeros((3, 3), np.int64); np.add.at(cm, (lab3[lbl >= 0], lbl[lbl >= 0]), 1)
    teacher = {"coverage_pts": cov, "accuracy_on_covered": acc, "confusion_true_x_teach": cm.tolist(),
               "building_precision": float(cm[1, 1] / max(cm[:, 1].sum(), 1)), "building_recall_of_covered": float(cm[1, 1] / max(cm[1].sum(), 1))}
    if V != "v0": table["residual_other_DIAG"] = {"class": "other", "fires": int(resid.sum()), "precision_pts": res_prec, "pts": int(resid.sum()), "decision": "diagnostic, not applied"}
    res["rules_v0"] = table; res["teacher_v0"] = teacher; res["feature_dist"] = dist; res["params"] = vars(a)
    log(json.dumps({"rules_v0": table, "teacher_v0": teacher}, indent=1))
    json.dump(res, open(f"{a.out}/{name}{a.tag}.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0); s = rng.choice(N, min(N, 150_000), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    cols = np.array(["#2a78d6", "#eb6834", "#1baf7a"]); grey = "#bbbbbb"
    ax[0].scatter(xyz[s, 0], xyz[s, 1], c=cols[lab3[s]], s=0.3); ax[0].set_title("true 3-class (blue ground, orange building, green other)")
    segc = np.where(seg[s] >= 0, (seg[s] * 7919) % 12, -1); pal = plt.get_cmap("tab20")(np.arange(12) / 12)
    ax[1].scatter(xyz[s, 0], xyz[s, 1], c=[pal[i] if i >= 0 else grey for i in segc], s=0.3); ax[1].set_title(f"segments ({S:,}), grey = unassigned")
    tc = np.where(lbl[s] >= 0, lbl[s], -1); ax[2].scatter(xyz[s, 0], xyz[s, 1], c=[cols[i] if i >= 0 else grey for i in tc], s=0.3); ax[2].set_title(f"teacher v0 (cov {cov:.2f}, acc {acc:.3f})")
    for x in ax: x.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{a.out}/{name}{a.tag}.png", dpi=90); log("done ->", a.out)

if __name__ == "__main__":
    main()
