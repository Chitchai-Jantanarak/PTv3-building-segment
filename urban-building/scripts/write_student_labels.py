"""Self-training round: student (probe checkpoint) labels every point, teacher v1 sidecar keeps the veto.
Rule: two-view agreement. teacher says X and student says X -> X. teacher ignore and student confident (p >= --conf) -> student.
teacher says X and student says Y != X -> ignore (never override the rule). Writes <tile>.labels3_round{r}.npy.
usage: python scripts/write_student_labels.py --ckpt checkpoints/probe_B/seg_a/best.pt --round 1 --splits train --gpu 1 [--report]
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from hydra import compose, initialize
from src.core.utils.checkpoint import load_ckpt
from src.models.seg_heads import SegAModel

COLLAPSE = {0: 0, 5: 0, 7: 0, 10: 0, 2: 1, 3: 1}
MAX_PTS = 80_000   # = data.max_points for sensat; the model was trained on 80k random points per whole tile

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--teacher-key", default="labels3_teacher"); ap.add_argument("--splits", nargs="+", default=["train"]); ap.add_argument("--conf", type=float, default=0.90)
    ap.add_argument("--root", default="data/processed/sensat"); ap.add_argument("--gpu", default="1"); ap.add_argument("--report", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--passes", type=int, default=2); ap.add_argument("--policy", default="anchor", choices=["anchor", "replace"], help="anchor: teacher kept, student fills ignore; replace: student wins where conf >= --conf, teacher elsewhere")
    a = ap.parse_args(); os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu); out_key = f"labels3_round{a.round}"
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["task=seg_a3", "data.num_classes=3"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegAModel(cfg); load_ckpt(path=a.ckpt, model=model, optimizer=None, strict=False, device=str(dev)); model.to(dev).eval()
    use_rgb = bool(cfg.task.get("use_rgb", False)); report = {}
    for split in a.splits:
        d = os.path.join(a.root, split)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"): continue
            path = os.path.join(d, f); side = os.path.join(d, f[:-4] + f".{out_key}.npy"); tside = os.path.join(d, f[:-4] + f".{a.teacher_key}.npy")
            if os.path.exists(side) and not a.force: print("skip", f, flush=True); continue
            if not os.path.exists(tside): print("no teacher sidecar, skip", f, flush=True); continue
            t0 = time.time(); z = np.load(path); xyz = z["xyz"].astype(np.float32); feats = z["features"].astype(np.float32); n = len(xyz)
            rgb = None
            if use_rgb and "rgb" in z.files:
                r = z["rgb"]; rgb = r[z["indices"]] if len(r) != n and "indices" in z.files else (r if len(r) == n else None)
                if rgb is not None: rgb = rgb.astype(np.float32); rgb = rgb / 255.0 if rgb.max() > 1.5 else rgb
            teach = np.load(tside).astype(np.int8)
            # inference must match training input: MAX_PTS random points from the WHOLE tile per forward, centred.
            # permutation passes cover every point once per pass; probabilities averaged over --passes.
            probs = np.zeros((n, 3), np.float32); rng = np.random.default_rng(0)
            with torch.no_grad():
                for _ in range(a.passes):
                    perm = rng.permutation(n)
                    for s in range(0, n, MAX_PTS):
                        idx = perm[s:s + MAX_PTS]; c = xyz[idx]; cen = c.mean(0); c = c - cen
                        fe = feats[idx].copy(); fe[:, :3] -= cen
                        ft = torch.from_numpy(fe).to(dev); ct = torch.from_numpy(c).to(dev); bt = torch.zeros(len(idx), dtype=torch.long, device=dev)
                        rt = torch.from_numpy(rgb[idx]).to(dev) if rgb is not None else None
                        probs[idx] += torch.softmax(model(ft, ct, bt, rgb=rt)["logits"].float(), -1).cpu().numpy()
            probs /= a.passes; pred = probs.argmax(1).astype(np.int8); conf = probs.max(1)
            # anchor policy: teacher labels are never dropped; student fills the ignore region above --conf
            out = teach.copy()
            fill = (teach < 0) & (conf >= a.conf); out[fill] = pred[fill]
            if a.policy == "replace":
                rep = (teach >= 0) & (pred != teach) & (conf >= a.conf); out[rep] = pred[rep]
            np.save(side, out)
            h = np.bincount(out + 1, minlength=4); th = np.bincount(teach + 1, minlength=4)
            agree = (teach >= 0) & (pred == teach)
            line = {"n": int(n), "coverage_teacher": round(float(1 - th[0] / n), 4), "coverage_round": round(float(1 - h[0] / n), 4), "student_agrees_with_teacher": round(float(agree.sum() / max((teach >= 0).sum(), 1)), 4),
                    "student_filled": int(fill.sum()), "ground": int(h[1]), "building": int(h[2]), "other": int(h[3]), "conf": a.conf, "sec": round(time.time() - t0, 1)}
            if a.report and "labels" in z.files:
                real = z["labels"]; real3 = np.full_like(real, 2)
                for k_, v_ in COLLAPSE.items(): real3[real == k_] = v_
                m = out >= 0; line["acc_on_covered"] = round(float((out[m] == real3[m]).mean()), 4)
                line["acc_student_filled"] = round(float((out[fill] == real3[fill]).mean()), 4) if fill.any() else None
                b = out == 1; line["building_precision"] = round(float((real3[b] == 1).mean()), 4) if b.any() else None
                ign = teach < 0; dis = (teach >= 0) & (pred != teach); sweep = {}
                line["student_acc_all"] = round(float((pred == real3).mean()), 4)
                for thr in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995):
                    fm = ign & (conf >= thr); dm = dis & (conf >= thr)
                    sweep[str(thr)] = {"fill_n": int(fm.sum()), "fill_acc": round(float((pred[fm] == real3[fm]).mean()), 4) if fm.any() else None,
                                       "fill_bld_prec": round(float((real3[fm & (pred == 1)] == 1).mean()), 4) if (fm & (pred == 1)).any() else None,
                                       "dis_n": int(dm.sum()), "dis_teacher_right": round(float((teach[dm] == real3[dm]).mean()), 4) if dm.any() else None,
                                       "dis_student_right": round(float((pred[dm] == real3[dm]).mean()), 4) if dm.any() else None}
                line["fill_sweep"] = sweep
            report[f"{split}/{f}"] = line; print(split, f, json.dumps(line), flush=True)
    os.makedirs("output/teacher", exist_ok=True); json.dump(report, open(f"output/teacher/round{a.round}_report.json", "w"), indent=1)

if __name__ == "__main__":
    main()
