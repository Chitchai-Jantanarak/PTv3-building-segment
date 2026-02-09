#!/bin/bash
# scripts/run_full_pipeline.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[PIPE] Stage 0: MAE pretraining"
bash scripts/run_mae.sh

echo "[PIPE] Stage 1: Seg-A semantic segmentation"
bash scripts/run_seg_a.sh

echo "[PIPE] Stage 2-1: Seg-B geometry inpainting"
bash scripts/run_seg_b.sh geom

echo "[PIPE] Stage 2-2: Seg-B color inpainting"
bash scripts/run_seg_b.sh color

echo "[PIPE] Stage 3: HAZUS classification"
bash scripts/run_hazus.sh

echo "[DONE] Full pipeline complete"