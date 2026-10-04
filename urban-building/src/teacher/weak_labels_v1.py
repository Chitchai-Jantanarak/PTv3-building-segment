# weak-label teacher v1: configs/teacher/weak_labels_v1.yaml, proven on Sensat val 32/17 (scripts/overseg_sensat_check.py)
# label_points(xyz, rel_z, rgb) -> int8 labels per input point: 0 ground, 1 building, 2 other, -1 ignore
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DEFAULT_PARAMS = dict(voxel=0.2, k=16, theta=8.0, cmax=0.01, dz=0.5, radius=1.0, min_pts=30, drop_radius=3.0, drop_dz=1.5, exg=0.20)


def _seg_median(x, seg, m, S, q=0.5):
    order = np.argsort(seg[m], kind="stable"); ss = seg[m][order]; xs = x[m][order]
    u, st, ct = np.unique(ss, return_index=True, return_counts=True)
    out = np.zeros(S)
    for s_, i, n in zip(u, st, ct):
        blk = xs[i:i + n]
        out[s_] = np.partition(blk, int(q * (n - 1)))[int(q * (n - 1))]
    return out


def label_points(xyz: np.ndarray, rel_z: np.ndarray, rgb: np.ndarray | None, p: dict = DEFAULT_PARAMS) -> np.ndarray:
    xyz = np.asarray(xyz, np.float64); rel_z = np.asarray(rel_z, np.float64); n_full = len(xyz)
    key = np.floor((xyz - xyz.min(0)) / p["voxel"]).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)   # inv: full point -> voxel rep
    inv = np.asarray(inv).reshape(-1)
    X = xyz[first]; RZ = rel_z[first]; N = len(X)
    RGB = None
    if rgb is not None:
        RGB = np.asarray(rgb, np.float64)[first]
        if RGB.max() > 1.5: RGB = RGB / 255.0

    # local covariance features
    t = cKDTree(X); nd, nn = t.query(X, k=p["k"] + 1); nd, nn = nd[:, 1:], nn[:, 1:]
    nbr = X[nn]; c = nbr - nbr.mean(1, keepdims=True); cov = np.einsum("nki,nkj->nij", c, c) / p["k"]
    ev, evec = np.linalg.eigh(cov); l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
    nrm = evec[:, :, 0]; nrm *= np.sign(nrm[:, 2:3] + 1e-9)
    planarity = (l2 - l3) / (l1 + 1e-9); curv = l3 / (l1 + l2 + l3 + 1e-9); vert = 1 - np.abs(nrm[:, 2])
    valid = nd < 2.5

    # region growing as edge-filtered graph
    ei = np.repeat(np.arange(N), p["k"]); ej = nn.ravel(); ok = valid.ravel() & (ei < ej); ei, ej = ei[ok], ej[ok]
    keep = (np.abs((nrm[ei] * nrm[ej]).sum(1)) > np.cos(np.radians(p["theta"]))) & (curv[ei] < p["cmax"] * 3) & (curv[ej] < p["cmax"] * 3) \
        & (np.abs(X[ei, 2] - X[ej, 2]) < p["dz"]) & (nd.ravel()[ok] < p["radius"])
    _, seg = connected_components(coo_matrix((np.ones(keep.sum()), (ei[keep], ej[keep])), shape=(N, N)), directed=False)
    has_seed = np.zeros(seg.max() + 1, bool); has_seed[seg[curv <= p["cmax"]]] = True
    seg = np.where(has_seed[seg], seg, -1)
    ids, cnt = np.unique(seg[seg >= 0], return_counts=True); seg[np.isin(seg, ids[cnt < p["min_pts"]])] = -1
    ids, inv_s = np.unique(seg, return_inverse=True); seg = inv_s - 1 if ids[0] == -1 else inv_s
    S = int(seg.max()) + 1; m = seg >= 0
    out_v = np.full(N, -1, np.int8)
    if S > 0:
        sz = np.bincount(seg[m], minlength=S)
        f_plan = _seg_median(planarity, seg, m, S); f_vert = _seg_median(vert, seg, m, S); f_curv = _seg_median(curv, seg, m, S)
        f_rzmed = _seg_median(RZ, seg, m, S); f_rzmax = _seg_median(RZ, seg, m, S, 0.95)
        xmin = np.full(S, np.inf); xmax = np.full(S, -np.inf); ymin = np.full(S, np.inf); ymax = np.full(S, -np.inf)
        np.minimum.at(xmin, seg[m], X[m, 0]); np.maximum.at(xmax, seg[m], X[m, 0]); np.minimum.at(ymin, seg[m], X[m, 1]); np.maximum.at(ymax, seg[m], X[m, 1])
        f_area = np.maximum(xmax - xmin, 0.1) * np.maximum(ymax - ymin, 0.1)
        # adjacency + boundary points
        si, sj = seg[ei], seg[ej]; diff = si != sj
        ok2 = (si >= 0) & (sj >= 0) & diff
        pairs = np.unique(np.c_[np.minimum(si[ok2], sj[ok2]), np.maximum(si[ok2], sj[ok2])], axis=0) if ok2.any() else np.zeros((0, 2), int)
        bnd = np.zeros(N, bool); bnd[ei[(si >= 0) & diff]] = True; bnd[ej[(sj >= 0) & diff]] = True
        rng = np.random.default_rng(1); bidx = rng.permutation(np.nonzero(bnd)[0]); seen = np.zeros(S, int); keep_b = np.zeros(len(bidx), bool)
        for q, s_ in enumerate(seg[bidx]):
            if seen[s_] < 150: keep_b[q] = True; seen[s_] += 1
        bidx = bidx[keep_b]
        f_drop = np.zeros(S)
        if len(bidx):
            balls = cKDTree(X[:, :2]).query_ball_point(X[bidx, :2], r=p["drop_radius"])
            drop_pt = np.array([(RZ[b] < RZ[i] - p["drop_dz"]).mean() if len(b) else 0.0 for i, b in zip(bidx, balls)])
            np.add.at(f_drop, seg[bidx], drop_pt); f_drop /= np.maximum(np.bincount(seg[bidx], minlength=S), 1)
        is_roof = (f_plan >= 0.45) & (f_vert <= 0.40) & (f_curv <= 0.03) & (f_rzmed >= 2.0) & (f_area >= 6) & (f_drop >= 0.05)
        touches_roof = np.zeros(S, bool)
        if len(pairs):
            touches_roof[pairs[:, 0][is_roof[pairs[:, 1]]]] = True; touches_roof[pairs[:, 1][is_roof[pairs[:, 0]]]] = True
        is_ground = (f_plan >= 0.45) & (f_vert <= 0.15) & (f_curv <= 0.02) & (f_rzmed <= 0.30)
        is_elev_ground = (f_plan >= 0.45) & (f_vert <= 0.15) & (f_curv <= 0.02) & (f_rzmed >= 2.0) & (f_area >= 50) & (f_drop < 0.02)
        is_wall = (f_vert >= 0.75) & (f_rzmax >= 2.0) & touches_roof
        fires = np.zeros((S, 3), bool); fires[:, 0] = is_ground | is_elev_ground; fires[:, 1] = is_roof | is_wall
        teach = np.where(fires.sum(1) == 1, fires.argmax(1), -1).astype(np.int8)
        out_v[m] = teach[seg[m]]
    # residual: vegetation by greenness on unassigned points
    if RGB is not None:
        exg = 2 * RGB[:, 1] - RGB[:, 0] - RGB[:, 2]
        resid = (~m) & (curv >= 0.03) & (RZ >= 0.5) & (exg >= p["exg"])
        out_v[resid] = 2
    return out_v[inv].astype(np.int8)   # broadcast voxel label to every original point
