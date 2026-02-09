#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")/.." && pwd)"

MODE="${1:-geom}"
shift 2>/dev/null || true
python runner.py mode=train task="seg_b_${MODE}" "$@"
