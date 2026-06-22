#!/bin/bash
# scripts/run_seg_b.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

MODE="${1:-geom}"

echo "[SEGB] Starting Seg-B ${MODE} training..."

if [ "${MODE}" == "geom" ]; then
    TASK="seg_b_geom"
elif [ "${MODE}" == "color" ]; then
    TASK="seg_b_color"
else
    echo "[ERR] Unknown mode: ${MODE}"
    exit 1
fi

python main.py task="${TASK}"

echo "[DONE] Seg-B ${MODE} training complete"