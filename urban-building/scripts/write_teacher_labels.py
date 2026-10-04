"""Write weak-label teacher v1 sidecars: <tile>.labels3_teacher.npy next to each processed npz.
usage: python scripts/write_teacher_labels.py --root data/processed/sensat --splits train val [--force] [--report]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
from src.teacher import label_points

COLLAPSE = {0: 0, 5: 0, 7: 0, 10: 0, 2: 1, 3: 1}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="data/processed/sensat"); ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--key", default="labels3_teacher"); ap.add_argument("--force", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(); report = {}
    for split in a.splits:
        d = os.path.join(a.root, split)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"): continue
            path = os.path.join(d, f); side = os.path.join(d, f[:-4] + f".{a.key}.npy")
            if os.path.exists(side) and not a.force: print("skip", f, flush=True); continue
            t0 = time.time(); z = np.load(path)
            xyz = z["xyz"]; relz = z["features"][:, 3]
            rgb = None
            if "rgb" in z.files:
                rgb = z["rgb"]; rgb = rgb[z["indices"]] if len(rgb) != len(xyz) and "indices" in z.files else (rgb if len(rgb) == len(xyz) else None)
            lab = label_points(xyz, relz, rgb)
            np.save(side, lab)
            h = np.bincount(lab + 1, minlength=4); cov = 1 - h[0] / len(lab)
            line = {"n": int(len(lab)), "coverage": round(float(cov), 4), "ground": int(h[1]), "building": int(h[2]), "other": int(h[3]), "sec": round(time.time() - t0, 1)}
            if a.report and "labels" in z.files:
                real = z["labels"]; real3 = np.full_like(real, 2)
                for k_, v_ in COLLAPSE.items(): real3[real == k_] = v_
                mm = lab >= 0; line["acc_on_covered"] = round(float((lab[mm] == real3[mm]).mean()), 4)
                b = lab == 1; line["building_precision"] = round(float((real3[b] == 1).mean()), 4) if b.any() else None
            report[f"{split}/{f}"] = line; print(split, f, json.dumps(line), flush=True)
    os.makedirs("output/teacher", exist_ok=True); json.dump(report, open("output/teacher/write_report.json", "w"), indent=1)

if __name__ == "__main__":
    main()
