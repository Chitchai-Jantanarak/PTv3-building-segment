# tools/dianose_color_knn.py

'''
 oracle knn's
 Using for diagnose sight rgba, color collapsing
''' 
from __future__ import annotations

import argparse
import os

os.environ.setdefault("SPCONV_ALGO", "native")

import torch
from hydra import compose, initialize

from src.datasets import build_dataloader
from src.models.mae.masking import BlockMasking, RandomMasking
from src.models.mae_features import (
    get_feature_indices,
    resolve_input_feature_names,
    resolve_target_feature_names,
)


def r2_per_channel(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    ss_res = ((actual - pred) ** 2).sum(dim=0)
    ss_tot = ((actual - actual.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return 1.0 - ss_res / ss_tot.clamp(min=1e-12)


def knn_color_predict(
    coord_msk: torch.Tensor,
    coord_vis: torch.Tensor,
    color_vis: torch.Tensor,
    k: int,
    chunk: int = 8192,
) -> torch.Tensor:
    kk = min(k, coord_vis.shape[0])
    out = []
    for i in range(0, coord_msk.shape[0], chunk):
        d = torch.cdist(coord_msk[i : i + chunk], coord_vis)   # [m, n_vis]
        idx = d.topk(kk, dim=1, largest=False).indices         # [m, kk]
        out.append(color_vis[idx].mean(dim=1))                 # [m, C]
    return torch.cat(out, dim=0)


def block_mask(coord: torch.Tensor, block_size: int, ratio: float):
    bm = BlockMasking(ratio=ratio, block_size=block_size)
    batch = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
    vis, msk, _ = bm(coord, batch)
    return vis, msk


def rand_mask(coord: torch.Tensor, ratio: float):
    rm = RandomMasking(ratio=ratio)
    batch = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
    vis, msk, _ = rm(coord, batch)
    return vis, msk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--ks", default="1,4,8,16")
    parser.add_argument("--ratio", type=float, default=0.65)
    parser.add_argument("--chunk", type=int, default=8192)
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["task=mae"])

    input_names = resolve_input_feature_names(cfg)
    target_names = resolve_target_feature_names(cfg, input_names)
    tgt_idx = get_feature_indices(input_names, target_names)
    color_names = [n for n in ("r", "g", "b") if n in target_names]
    if not color_names:
        raise SystemExit("No r/g/b in target features -- nothing to diagnose")
    color_pos = [target_names.index(n) for n in color_names]

    loader = build_dataloader(cfg, split=args.split)

    variants = [
        (f"block b=32 r={args.ratio}", lambda c: block_mask(c, 32, args.ratio)),
        (f"block b=16 r={args.ratio}", lambda c: block_mask(c, 16, args.ratio)),
        (f"block b=8  r={args.ratio}", lambda c: block_mask(c, 8, args.ratio)),
        (f"block b=4  r={args.ratio}", lambda c: block_mask(c, 4, args.ratio)),
        (f"random     r={args.ratio}", lambda c: rand_mask(c, args.ratio)),
    ]

    accum: dict[tuple[str, int], list[list[torch.Tensor]]] = {
        (name, k): [[], []] for name, _ in variants for k in ks
    }

    n_tiles = 0
    with torch.no_grad():
        for batch in loader:
            feat = batch["points"].to(device)
            coord = batch["coords"].to(device)
            bt = batch["batch"].to(device)
            color_all = feat[:, tgt_idx][:, color_pos]  # [N, C] raw rgb

            for b in bt.unique():
                m = bt == b
                c = coord[m]
                col = color_all[m]
                if c.shape[0] < 8:
                    continue
                n_tiles += 1
                for name, mask_fn in variants:
                    vis, msk = mask_fn(c)
                    if vis.numel() == 0 or msk.numel() == 0:
                        continue
                    for k in ks:
                        pred = knn_color_predict(
                            c[msk], c[vis], col[vis], k, chunk=args.chunk
                        )
                        accum[(name, k)][0].append(pred)
                        accum[(name, k)][1].append(col[msk])

    print(f"\nOracle kNN color predictability  (split={args.split}, tiles={n_tiles})")
    print("Higher R2 = color recoverable from nearest visible points.\n")
    header = f"{'masking':<20} {'k':>3} " + " ".join(f"{n:>8}" for n in color_names)
    print(header)
    print("-" * len(header))
    for name, _ in variants:
        for k in ks:
            preds, actuals = accum[(name, k)]
            if not preds:
                continue
            pred = torch.cat(preds, dim=0)
            actual = torch.cat(actuals, dim=0)
            r2 = r2_per_channel(pred, actual)
            row = " ".join(f"{v:>8.3f}" for v in r2.tolist())
            print(f"{name:<20} {k:>3} {row}")
        print()


if __name__ == "__main__":
    main()
