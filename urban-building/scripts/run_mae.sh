#!/bin/bash
# scripts/run_mae.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[MAE] Starting MAE pretraining..."

sed -i 's/task: .*/task: mae/' configs/config.yaml

python main.py

echo "[DONE] MAE pretraining complete"