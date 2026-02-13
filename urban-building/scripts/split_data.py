#!/usr/bin/env python3
"""
Preprocess raw point cloud files and split into train/val directories.

Usage:
    # WHU ALS (h5 format)
    python scripts/split_data.py \
        --raw-dir data/raw/whu/als/h5 \
        --out-dir data/processed/whu/als \
        --format h5 \
        --val-ratio 0.15 \
        --seed 42

    # SensatUrban (ply format) - random split
    python scripts/split_data.py \
        --raw-dir data/raw/sansat \
        --out-dir data/processed/sansat \
        --format ply \
        --val-ratio 0.15 \
        --seed 42

    # SensatUrban - use official split file
    python scripts/split_data.py \
        --raw-dir data/raw/sansat \
        --out-dir data/processed/sansat \
        --format ply \
        --val-files cambridge_block_10 cambridge_block_17 cambridge_block_20 cambridge_block_32

    # WHU ALS - specify val files manually
    python scripts/split_data.py \
        --raw-dir data/raw/whu/als/h5 \
        --out-dir data/processed/whu/als \
        --format h5 \
        --val-files 0005 0012 0018

    # Dry run (preview split without processing)
    python scripts/split_data.py \
        --raw-dir data/raw/whu/als/h5 \
        --out-dir data/processed/whu/als \
        --format h5 \
        --val-ratio 0.15 \
        --dry-run
"""

import argparse
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def find_raw_files(raw_dir: Path, fmt: str) -> list[Path]:
    """Find all raw point cloud files in directory."""
    ext_map = {
        "h5": ["*.h5", "*.hdf5"],
        "ply": ["*.ply"],
        "las": ["*.las", "*.laz"],
    }
    exts = ext_map.get(fmt, [f"*.{fmt}"])
    files = []
    for ext in exts:
        files.extend(sorted(raw_dir.glob(ext)))
    return files


def determine_split(
    files: list[Path],
    val_ratio: float,
    val_files: list[str] | None,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split files into train and val sets.

    If val_files is provided, uses those stems for val.
    Otherwise uses val_ratio with random shuffle.
    """
    if val_files:
        val_stems = set(val_files)
        val = [f for f in files if f.stem in val_stems]
        train = [f for f in files if f.stem not in val_stems]

        missing = val_stems - {f.stem for f in val}
        if missing:
            print(f"[WARN] val_files not found in raw dir: {missing}")
    else:
        shuffled = files.copy()
        random.seed(seed)
        random.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * val_ratio))
        val = shuffled[:n_val]
        train = shuffled[n_val:]

        # Sort for reproducibility in logs
        train.sort()
        val.sort()

    return train, val


def preprocess_and_save(
    files: list[Path],
    out_dir: Path,
    cfg,
) -> int:
    """Preprocess files and save as .npz to out_dir."""
    from src.core.preprocessing.preprocess import preprocess_file

    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for f in files:
        out_path = out_dir / f"{f.stem}.npz"
        if out_path.exists():
            print(f"  [SKIP] {out_path.name} already exists")
            success += 1
            continue
        ok = preprocess_file(f, out_path, cfg)
        if ok:
            print(f"  [OK]   {f.name} -> {out_path.name}")
            success += 1
        else:
            print(f"  [FAIL] {f.name}")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and split point cloud data into train/val"
    )
    parser.add_argument(
        "--raw-dir", type=str, required=True, help="Directory with raw files"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Output processed directory"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="h5",
        choices=["h5", "ply", "las"],
        help="Raw file format",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of files for validation (default: 0.15)",
    )
    parser.add_argument(
        "--val-files",
        nargs="+",
        default=None,
        help="Explicit file stems (without extension) to use as val set",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
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

    # Find files
    files = find_raw_files(raw_dir, args.format)
    if not files:
        print(f"[ERROR] No .{args.format} files found in {raw_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} raw files in {raw_dir}")

    # Split
    train_files, val_files = determine_split(
        files, args.val_ratio, args.val_files, args.seed
    )

    print(f"\n[SPLIT] train: {len(train_files)} files, val: {len(val_files)} files")
    print(f"  Val ratio: {len(val_files)/len(files):.1%}")

    print(f"\n--- TRAIN ({len(train_files)}) ---")
    for f in train_files:
        print(f"  {f.name}")

    print(f"\n--- VAL ({len(val_files)}) ---")
    for f in val_files:
        print(f"  {f.name}")

    if args.dry_run:
        print("\n[DRY RUN] No files processed.")
        return

    # Load config for preprocessing
    from omegaconf import OmegaConf

    default = OmegaConf.load(project_root / "configs" / "default.yaml")
    config = OmegaConf.load(project_root / "configs" / "config.yaml")
    cfg = OmegaConf.merge(default, config)

    # Override format in config
    cfg.data.format = args.format

    # Process train
    train_out = out_dir / "train"
    print(f"\n[PROC] Processing train -> {train_out}")
    n_train = preprocess_and_save(train_files, train_out, cfg)

    # Process val
    val_out = out_dir / "val"
    print(f"\n[PROC] Processing val -> {val_out}")
    n_val = preprocess_and_save(val_files, val_out, cfg)

    print(f"\n[DONE] train: {n_train}/{len(train_files)}, val: {n_val}/{len(val_files)}")
    print(f"  Output: {out_dir}/{{train,val}}/*.npz")


if __name__ == "__main__":
    main()
