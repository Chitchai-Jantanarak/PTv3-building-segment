#!/bin/bash
# scripts/run_seg_a.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[SEGA] Starting Seg-A training..."

sed -i 's/task: .*/task: seg_a/' configs/config.yaml

python main.py

echo "[DONE] Seg-A training complete"