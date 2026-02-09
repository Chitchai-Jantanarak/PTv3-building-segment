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

python - "${DATA_ROOT}" "${OUTPUT_ROOT}" <<'PYEOF'
import sys
from pathlib import Path
from src.core.preprocessing import preprocess_file
from omegaconf import OmegaConf

default = OmegaConf.load("configs/default.yaml")
config = OmegaConf.load("configs/config.yaml")
cfg = OmegaConf.merge(default, config)

data_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])

for ext in ("*.las", "*.laz", "*.ply"):
    for f in sorted(data_root.glob("**/" + ext)):
        out = output_root / f.relative_to(data_root).with_suffix(".npz")
        print(f"[PREP] {f} -> {out}")
        preprocess_file(f, out, cfg)
PYEOF

echo "[DONE] Preprocessing complete"
