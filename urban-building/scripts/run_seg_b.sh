#!/bin/bash
# scripts/run_seg_b.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

MODE="${1:-geom}"

echo "[SEGB] Starting Seg-B ${MODE} training..."

if [ "${MODE}" == "geom" ]; then
    sed -i 's/task: .*/task: seg_b_geom/' configs/config.yaml
elif [ "${MODE}" == "color" ]; then
    sed -i 's/task: .*/task: seg_b_color/' configs/config.yaml
else
    echo "[ERR] Unknown mode: ${MODE}"
    exit 1
fi

python main.py

echo "[DONE] Seg-B ${MODE} training complete"