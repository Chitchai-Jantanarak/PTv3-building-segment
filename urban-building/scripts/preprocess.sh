#!/bin/bash
# scripts/preprocess.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[PREP] Starting preprocessing..."

DATA_ROOT="${1:-data/raw}"
OUTPUT_ROOT="${2:-data/processed}"

mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] Data root: ${DATA_ROOT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"

python -c "
from pathlib import Path
from src.core.preprocessing import preprocess_file
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/config.yaml')
cfg = OmegaConf.merge(OmegaConf.load('configs/default.yaml'), cfg)

data_root = Path('${DATA_ROOT}')
output_root = Path('${OUTPUT_ROOT}')

for f in data_root.glob('**/*.las'):
    out = output_root / f.relative_to(data_root).with_suffix('.npz')
    print(f'[PREP] {f} -> {out}')
    preprocess_file(f, out, cfg)

for f in data_root.glob('**/*.ply'):
    out = output_root / f.relative_to(data_root).with_suffix('.npz')
    print(f'[PREP] {f} -> {out}')
    preprocess_file(f, out, cfg)
"

echo "[DONE] Preprocessing complete"
