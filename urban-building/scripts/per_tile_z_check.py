"""Per-tile MAE reconstruction check: is the two-branch z / rel_z scatter one bad tile or model error?
usage: python scripts/per_tile_z_check.py [--ckpt checkpoints/mae/best.pt] [--out output/per_tile_z]
"""
import argparse, json, os, sys
import numpy as np, torch
from hydra import compose, initialize
from src.core.utils.checkpoint import load_ckpt
from src.datasets import build_dataloader
from src.models.mae import MAEForPretraining
from src.eval.evaluate import _collect_mae_predictions
from src.eval.metrics import per_feature_r2, per_feature_rmse, per_feature_bias

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/mae/best.pt"); ap.add_argument("--out", default="output/per_tile_z"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["task=mae"])
    dev = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    model = MAEForPretraining(cfg); load_ckpt(path=a.ckpt, model=model, optimizer=None, strict=False, device=str(dev)); model.to(dev).eval()
    names = list(model.target_feature_names)
    loader = build_dataloader(cfg, split="val"); ds = loader.dataset; files = list(ds.file_list)
    print("val tiles:", [f.name for f in files], flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = {}; fig, axes = plt.subplots(2, len(files), figsize=(5 * len(files), 9), squeeze=False)
    for k, f in enumerate(files):
        ds.file_list = [f]; ds._cache = {}
        sub = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=loader.collate_fn)
        with torch.no_grad(): data = _collect_mae_predictions(model, sub, dev)
        pred, tgt = data["reconstructed"], data["target"]
        r2 = per_feature_r2(pred, tgt, names); rmse = per_feature_rmse(pred, tgt, names); bias = per_feature_bias(pred, tgt, names)
        rows[f.stem] = {"n_masked": int(len(pred)), "r2": {n: round(float(r2[n]), 4) for n in names}, "rmse": {n: round(float(rmse[n]), 3) for n in names}, "bias": {n: round(float(bias[n]), 3) for n in names}}
        print(f.stem, json.dumps(rows[f.stem]), flush=True)
        rng = np.random.default_rng(0); s = rng.choice(len(pred), min(len(pred), 20000), replace=False)
        for r, feat in enumerate(("z", "rel_z")):
            if feat not in names: continue
            i = names.index(feat); ax = axes[r, k]
            ax.scatter(tgt[s, i], pred[s, i], s=1, alpha=0.4); lo, hi = tgt[:, i].min(), tgt[:, i].max(); ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(f"{f.stem}\n{feat} R2={r2[feat]:.3f} rmse={rmse[feat]:.2f}"); ax.set_xlabel("actual"); ax.set_ylabel("pred")
    fig.tight_layout(); fig.savefig(f"{a.out}/per_tile_z.png", dpi=100)
    json.dump(rows, open(f"{a.out}/per_tile_z.json", "w"), indent=1); print("done ->", a.out, flush=True)

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd()); main()
