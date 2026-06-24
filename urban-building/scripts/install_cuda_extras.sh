#!/usr/bin/env bash
# scripts/install_cuda_extras.sh
#
# Installs CUDA-specific packages that live outside uv's dependency graph
# (spconv, torch-scatter, flash-attn, pointops).  Run this:
#   - After the initial uv venv setup
#   - After any `uv sync` / `uv add` that may have wiped these packages
#
# These packages are deliberately NOT in pyproject.toml because they require
# matching exact torch/CUDA versions and are not redistributable via PyPI.
#
# Usage:
#   bash scripts/install_cuda_extras.sh          # install / reinstall all
#   SKIP_FLASH_ATTN=1 bash scripts/...           # skip flash-attn (slow build)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PTv3_DIR="$(cd "${PROJECT_DIR}/../PointTransformerV3" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

# ── activate venv if not already active ──────────────────────────────────────
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -f "${VENV_DIR}/bin/activate" ]; then
        echo "[ERR] No venv at ${VENV_DIR}. Run setup_env2.sh first."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

PYTHON="$(which python)"
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -z "${TORCH_VER}" ]; then
    echo "[ERR] torch not importable. Install torch first (setup_env2.sh)."
    exit 1
fi

# Parse torch major.minor and CUDA tag from torch version string (e.g. 2.10.0+cu128)
TORCH_VER_SHORT=$(echo "${TORCH_VER}" | sed 's/+.*//')   # 2.10.0
CUDA_TAG=$(echo "${TORCH_VER}" | grep -oP 'cu\d+' || echo "cu128")
SPCONV_CUDA_TAG="${SPCONV_CUDA_TAG:-cu124}"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VER_SHORT}+${CUDA_TAG}.html"

echo "[INFO] torch=${TORCH_VER} cuda_tag=${CUDA_TAG} pyg_url=${PYG_WHEEL_URL}"

# ── 1. spconv ─────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] spconv-${SPCONV_CUDA_TAG}"
if python -c "import spconv" 2>/dev/null; then
    echo "      already importable, skipping"
else
    uv pip install "spconv-${SPCONV_CUDA_TAG}"
fi

# ── 2. torch-scatter ─────────────────────────────────────────────────────────
echo ""
echo "[2/4] torch-scatter"
if python -c "import torch_scatter" 2>/dev/null; then
    echo "      already importable, skipping"
else
    uv pip install torch-scatter -f "${PYG_WHEEL_URL}"
fi

# ── 3. flash-attn ─────────────────────────────────────────────────────────────
echo ""
echo "[3/4] flash-attn"
if [ "${SKIP_FLASH_ATTN:-0}" = "1" ]; then
    echo "      SKIP_FLASH_ATTN=1, skipping"
else
    # Prebuilt flash-attn wheels often link GLIBC_2.32 symbols that older
    # container base images don't export even when ldd reports a newer glibc.
    # Always build from source with pip (not uv pip — uv ignores --no-binary).
    FLASH_OK=0
    python -c "import flash_attn" 2>/dev/null && FLASH_OK=1

    if [ "${FLASH_OK}" = "1" ]; then
        echo "      already importable, skipping"
    else
        echo "      building flash-attn from source (takes ~20-40 min)..."
        # Wipe any broken prebuilt first
        python -m pip uninstall -y flash-attn 2>/dev/null || true

        if [ -z "${CUDA_HOME:-}" ]; then
            for p in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.4; do
                if [ -d "${p}" ]; then CUDA_HOME="${p}"; break; fi
            done
        fi
        echo "      CUDA_HOME=${CUDA_HOME:-<not set>}"
        export MAX_JOBS="${MAX_JOBS:-4}"
        export FLASH_ATTENTION_FORCE_BUILD=TRUE

        # Use pip (not uv pip) so --no-binary is honoured and no prebuilt wheel
        # is pulled from cache.
        python -m pip install flash-attn --no-build-isolation --no-binary flash-attn \
            || echo "[WARN] flash-attn source build failed — set enable_flash: false in configs/model/ptv3.yaml"
    fi
fi

# ── 4. pointops ──────────────────────────────────────────────────────────────
echo ""
echo "[4/4] pointops"
if python -c "import pointops" 2>/dev/null; then
    echo "      already importable, skipping"
else
    POINTOPS_DIR="${PTv3_DIR}/Pointcept/libs/pointops"
    if [ -d "${POINTOPS_DIR}" ]; then
        python -m pip install --no-build-isolation -e "${POINTOPS_DIR}"
    else
        echo "[WARN] pointops not found at ${POINTOPS_DIR} — init submodule first:"
        echo "       git submodule update --init PointTransformerV3"
    fi
fi

# ── verification ─────────────────────────────────────────────────────────────
echo ""
echo "[INFO] Verification:"
python -c "import spconv;       print('  spconv:       OK')"
python -c "import torch_scatter;print('  torch-scatter: OK')"
python -c "import flash_attn;   print('  flash-attn:    OK')" 2>/dev/null \
    || echo "  flash-attn:    FAILED (set enable_flash: false in ptv3.yaml)"
python -c "import pointops;     print('  pointops:      OK')" 2>/dev/null \
    || echo "  pointops:      FAILED"
echo ""
echo "[DONE] Run ./scripts/run_mae.sh"
