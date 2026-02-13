#!/usr/bin/env python3
"""
Preprocess SensatUrban with the official train/val/test split.

Official split (from QingyongHu/SensatUrban):
  - Test (6 tiles): birmingham_block_{2,8}, cambridge_block_{15,16,22,27}
  - Val  (4 tiles): cambridge_block_{10,17,20,32}
  - Train: all remaining tiles

Usage:
    python scripts/split_sensat.py \
        --raw-dir data/raw/sansat \
        --out-dir data/processed/sansat

    # Dry run
    python scripts/split_sensat.py \
        --raw-dir data/raw/sansat \
        --out-dir data/processed/sansat \
        --dry-run
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Official SensatUrban split (from the benchmark)
# Test tiles: held out for benchmark evaluation
OFFICIAL_TEST = {
    "birmingham_block_2",
    "birmingham_block_8",
    "cambridge_block_15",
    "cambridge_block_16",
    "cambridge_block_22",
    "cambridge_block_27",
}

# Validation tiles: used for model selection during training
OFFICIAL_VAL = {
    "cambridge_block_10",
    "cambridge_block_17",
    "cambridge_block_20",
    "cambridge_block_32",
}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SensatUrban with official split"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/sansat",
        help="Directory with raw .ply files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/sansat",
        help="Output directory",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also preprocess test tiles into test/ dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split plan without processing",
    )

    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        print(f"[ERROR] Raw directory does not exist: {raw_dir}")
        sys.exit(1)

    # Find all PLY files
    files = sorted(raw_dir.glob("*.ply"))
    if not files:
        print(f"[ERROR] No .ply files found in {raw_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} PLY files in {raw_dir}")

    # Categorize
    train_files, val_files, test_files, skipped = [], [], [], []
    for f in files:
        stem = f.stem
        if stem in OFFICIAL_TEST:
            test_files.append(f)
        elif stem in OFFICIAL_VAL:
            val_files.append(f)
        else:
            train_files.append(f)

    print(f"\n[SPLIT] Official SensatUrban split:")
    print(f"  Train: {len(train_files)} tiles")
    print(f"  Val:   {len(val_files)} tiles")
    print(f"  Test:  {len(test_files)} tiles")

    for label, flist in [("TRAIN", train_files), ("VAL", val_files), ("TEST", test_files)]:
        print(f"\n--- {label} ---")
        for f in flist:
            print(f"  {f.stem}")

    if args.dry_run:
        print("\n[DRY RUN] No files processed.")
        return

    # Load config
    from omegaconf import OmegaConf
    from src.core.preprocessing.preprocess import preprocess_file

    default = OmegaConf.load(project_root / "configs" / "default.yaml")
    config = OmegaConf.load(project_root / "configs" / "config.yaml")
    cfg = OmegaConf.merge(default, config)
    cfg.data.format = "ply"

    def process_split(name, flist):
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for f in flist:
            out_path = split_dir / f"{f.stem}.npz"
            if out_path.exists():
                print(f"  [SKIP] {out_path.name} exists")
                ok += 1
                continue
            if preprocess_file(f, out_path, cfg):
                print(f"  [OK]   {f.stem} -> {name}/{out_path.name}")
                ok += 1
            else:
                print(f"  [FAIL] {f.stem}")
        return ok

    print(f"\n[PROC] Processing train...")
    n_train = process_split("train", train_files)

    print(f"\n[PROC] Processing val...")
    n_val = process_split("val", val_files)

    if args.include_test:
        print(f"\n[PROC] Processing test...")
        n_test = process_split("test", test_files)
    else:
        n_test = 0

    print(f"\n[DONE] train: {n_train}, val: {n_val}, test: {n_test}")
    print(f"  Output: {out_dir}/{{train,val}}/*.npz")


if __name__ == "__main__":
    main()
