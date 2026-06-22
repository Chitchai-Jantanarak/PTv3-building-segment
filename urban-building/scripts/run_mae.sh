#!/bin/bash
# scripts/run_mae.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[MAE] Starting MAE pretraining..."

python main.py task=mae

echo "[DONE] MAE pretraining complete"