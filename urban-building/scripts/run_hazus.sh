#!/bin/bash
# scripts/run_hazus.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[HAZ] Starting HAZUS training..."

sed -i 's/task: .*/task: hazus/' configs/config.yaml

python main.py

echo "[DONE] HAZUS training complete"   