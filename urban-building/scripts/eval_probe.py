"""Score a SEG-A probe checkpoint on the REAL collapsed 3-class val labels (independent of the label_key it was trained with).
usage: python scripts/eval_probe.py --ckpt checkpoints/probe_B/seg_a/best.pt --name B [--gpu 1]
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from hydra import compose, initialize
from src.core.utils.checkpoint import load_ckpt
from src.datasets import build_dataloader
from src.models.seg_heads import SegAModel

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--name", required=True); ap.add_argument("--gpu", default="1"); ap.add_argument("--out", default="output/probe")
    a = ap.parse_args(); os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["task=seg_a3", "data.num_classes=3", "task.label_key=null", "+task.num_workers=2"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegAModel(cfg); load_ckpt(path=a.ckpt, model=model, optimizer=None, strict=False, device=str(dev)); model.to(dev).eval()
    loader = build_dataloader(cfg, split="val")
    n = 3; cm = np.zeros((n, n), np.int64); per_tile = {}
    with torch.no_grad():
        for batch in loader:
            feat = batch["points"].to(dev); coord = batch["coords"].to(dev); bidx = batch["batch"].to(dev)
            rgb = batch.get("rgb"); rgb = rgb.to(dev) if rgb is not None else None
            lab = batch["labels"].to(dev); pred = model(feat, coord, bidx, rgb=rgb)["logits"].argmax(-1)
            v = (lab >= 0) & (lab < n)
            c = torch.bincount(lab[v] * n + pred[v], minlength=n * n).reshape(n, n).cpu().numpy(); cm += c
            for b in torch.unique(bidx).tolist():
                mb = (bidx == b) & v
                cb = torch.bincount(lab[mb] * n + pred[mb], minlength=n * n).reshape(n, n).cpu().numpy()
                fp = batch.get("file_path"); key = os.path.basename(fp[b] if isinstance(fp, (list, tuple)) else str(fp))
                per_tile[key] = per_tile.get(key, np.zeros((n, n), np.int64)) + cb
    def iou(c):
        tp = np.diag(c); return (tp / np.maximum(c.sum(0) + c.sum(1) - tp, 1)).tolist()
    res = {"name": a.name, "ckpt": a.ckpt, "classes": ["ground", "building", "other"], "iou": iou(cm), "mIoU": float(np.mean(iou(cm))), "confusion": cm.tolist(),
           "per_tile_building_iou": {k: iou(v)[1] for k, v in per_tile.items()}}
    print(json.dumps(res, indent=1)); os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/eval_{a.name}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
