#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.7.0"
CUDA_TAG="cu128"
SPCONV_PKG="spconv-cu124==2.3.8"
FLASH_ATTN_VERSION="2.8.3"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PTv3_DIR="$(cd "${PROJECT_DIR}/../PointTransformerV3" && pwd)"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_TAG}.html"

export UV_LINK_MODE=copy
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda12_9}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export PTV3_CONV_ALGO="MaskSplitImplicitGemm"

echo "[INFO] CUDA_HOME=${CUDA_HOME} arch=${TORCH_CUDA_ARCH_LIST} spconv_algo=${PTV3_CONV_ALGO}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERR] nvidia-smi not found. Install the NVIDIA driver first."
    exit 1
fi
echo "[INFO] driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "[ERR] nvcc not found at ${CUDA_HOME}/bin/nvcc. Point CUDA_HOME at a CUDA 12.x toolkit."
    exit 1
fi
echo "[INFO] nvcc: $("${CUDA_HOME}/bin/nvcc" --version | grep release)"

if ! command -v uv &>/dev/null; then
    echo "[INFO] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi
echo "[INFO] uv: $(uv --version)"

if [ -d "${VENV_DIR}" ]; then
    echo "[ERR] ${VENV_DIR} exists. For a clean build: rm -rf ${VENV_DIR}"
    exit 1
fi

echo "[INFO] creating venv (python ${PYTHON_VERSION})..."
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed
source "${VENV_DIR}/bin/activate"
echo "[INFO] python: $(python --version) @ $(which python)"

echo "[1/7] torch ${PYTORCH_VERSION}+${CUDA_TAG}"
uv pip install "torch==${PYTORCH_VERSION}" torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "[2/7] ${SPCONV_PKG}"
uv pip install "${SPCONV_PKG}"

echo "[3/7] torch-scatter"
uv pip install torch-scatter -f "${PYG_WHEEL_URL}"

echo "[4/7] flash-attn ${FLASH_ATTN_VERSION} (source build, sm_90)"
uv pip uninstall flash-attn 2>/dev/null || true
uv pip install packaging ninja psutil wheel "setuptools>=70.1"
python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" \
    --no-build-isolation --no-cache-dir --no-binary=:all: --no-deps \
    || echo "[WARN] flash-attn build failed — set enable_flash: false in configs/model/ptv3.yaml"

echo "[5/7] timm, addict, einops"
uv pip install "timm>=0.9.0" addict einops

echo "[6/7] pointops"
if [ -d "${PTv3_DIR}/Pointcept/libs/pointops" ]; then
    uv pip install -e "${PTv3_DIR}/Pointcept/libs/pointops" --no-build-isolation
else
    echo "[WARN] pointops not found at ${PTv3_DIR}/Pointcept/libs/pointops"
fi

echo "[7/7] urban-building (editable)"
cd "${PROJECT_DIR}"
uv pip install -e ".[all]"

echo "[INFO] verifying..."
python - <<'PY'
import torch, flash_attn, torch_scatter, pointops, spconv.pytorch, hydra
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("spconv OK  flash_attn OK  torch_scatter OK  pointops OK  hydra OK")
PY

echo "[DONE] activate with: source ${VENV_DIR}/bin/activate"
