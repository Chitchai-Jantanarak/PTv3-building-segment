#!/bin/bash
# scripts/run_seg_a.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[SEGA] Starting Seg-A training..."

python main.py task=seg_a

echo "[DONE] Seg-A training complete"