#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")/.." && pwd)"
python runner.py mode=train task=seg_a "$@"
