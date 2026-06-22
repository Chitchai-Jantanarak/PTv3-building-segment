#!/bin/bash
# scripts/run_hazus.sh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[HAZ] Starting HAZUS training..."

python main.py task=hazus

echo "[DONE] HAZUS training complete"   