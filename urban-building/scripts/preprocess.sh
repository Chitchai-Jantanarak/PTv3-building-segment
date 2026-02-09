#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")/.." && pwd)"

DATA="${1:-sansat}"
echo "[PREP] Preprocessing dataset: ${DATA}"
python runner.py mode=preprocess data="${DATA}"
echo "[DONE] Preprocessing complete"
