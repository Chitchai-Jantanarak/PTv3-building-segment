#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
CUDA_TAG="cu128"
SPCONV_PKG="spconv-cu124==2.3.8"
FLASH_ATTN_VERSION="2.8.3"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PTv3_DIR="$(cd "${PROJECT_DIR}/../PointTransformerV3" && pwd)"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_TAG}.html"
FLASH_LOG="${PROJECT_DIR}/flash_attn_build.log"

export UV_LINK_MODE=copy

# CUDA toolkit path.
# For your machine this is usually: $HOME/cuda12_9
export CUDA_HOME="${CUDA_HOME:-$HOME/cuda12_9}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# CUDA extension build flags.
export FORCE_CUDA=1

# PyTorch extension arch format.
export TORCH_CUDA_ARCH_LIST="9.0"

# flash-attn setup.py uses its own arch variable.
# H100 / Hopper = sm_90.
export FLASH_ATTN_CUDA_ARCHS="90"

# Safer defaults for source build.
# If build succeeds and you want faster compile later, try MAX_JOBS=2 NVCC_THREADS=1.
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

# Force flash-attn to build from source and not skip CUDA kernels.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

# PTv3/spconv setting.
export PTV3_CONV_ALGO="MaskSplitImplicitGemm"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] PTv3_DIR=${PTv3_DIR}"
echo "[INFO] CUDA_HOME=${CUDA_HOME}"
echo "[INFO] torch=${PYTORCH_VERSION}+${CUDA_TAG} torchvision=${TORCHVISION_VERSION}"
echo "[INFO] arch torch=${TORCH_CUDA_ARCH_LIST} flash=${FLASH_ATTN_CUDA_ARCHS}"
echo "[INFO] MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS}"
echo "[INFO] spconv_algo=${PTV3_CONV_ALGO}"

if ! nvidia-smi &>/dev/null; then
    echo "[ERR] nvidia-smi not found. Install/check NVIDIA driver first."
    exit 1
fi

echo "[INFO] driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "[ERR] nvcc not found at ${CUDA_HOME}/bin/nvcc"
    echo "[ERR] Point CUDA_HOME at a CUDA 12.x toolkit, for example:"
    echo "      export CUDA_HOME=\$HOME/cuda12_9"
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
    echo "[ERR] ${VENV_DIR} exists."
    echo "[ERR] For a clean build, run:"
    echo "      rm -rf ${VENV_DIR}"
    exit 1
fi

echo "[INFO] creating venv with Python ${PYTHON_VERSION}..."
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[INFO] python: $(python --version) @ $(which python)"
echo "[INFO] pip: $(python -m pip -V)"

echo "[1/7] torch ${PYTORCH_VERSION}+${CUDA_TAG}"
uv pip install \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "[2/7] ${SPCONV_PKG}"
uv pip install "${SPCONV_PKG}"

echo "[3/7] torch-scatter"
uv pip install torch-scatter -f "${PYG_WHEEL_URL}"

echo "[4/7] flash-attn ${FLASH_ATTN_VERSION} source build for sm_90"

# Remove old broken/cached flash-attn leftovers inside the venv.
python -m pip uninstall -y flash-attn flash_attn 2>/dev/null || true

SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

find "${SITE_PACKAGES}" -iname "*flash*attn*" -exec rm -rf {} + 2>/dev/null || true

echo "[INFO] installing flash-attn build requirements into active venv"
python -m pip install packaging ninja psutil wheel "setuptools>=70.1"

echo "[INFO] build env before flash-attn:"
python - <<'PY'
import os
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch cuda available:", torch.cuda.is_available())

for key in [
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "FORCE_CUDA",
    "TORCH_CUDA_ARCH_LIST",
    "FLASH_ATTN_CUDA_ARCHS",
    "MAX_JOBS",
    "NVCC_THREADS",
    "FLASH_ATTENTION_FORCE_BUILD",
    "FLASH_ATTENTION_SKIP_CUDA_BUILD",
]:
    value = os.getenv(key)
    if key == "PATH" and value:
        value = value.split(":")[0] + ":..."
    print(f"{key}={value}")
PY

FLASH_ATTN_OK=0

set +e
python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" \
    --no-build-isolation \
    --no-cache-dir \
    --no-binary=:all: \
    --no-deps \
    -v \
    2>&1 | tee "${FLASH_LOG}"
FLASH_STATUS=${PIPESTATUS[0]}
set -e

if [ "${FLASH_STATUS}" -eq 0 ]; then
    FLASH_ATTN_OK=1
    echo "[INFO] flash-attn build/install succeeded"
else
    echo "[WARN] flash-attn build failed."
    echo "[WARN] Build log saved at: ${FLASH_LOG}"
    echo "[WARN] You can continue, but set enable_flash: false in configs/model/ptv3.yaml"
fi

echo "[INFO] flash-attn build log quick check:"
grep -E "Guessing wheel URL|Raw wheel path|Building wheel|building 'flash_attn|nvcc|error:" "${FLASH_LOG}" | tail -80 || true

echo "[5/7] timm, addict, einops"
uv pip install "timm>=0.9.0" addict einops

echo "[6/7] pointops"
if [ -d "${PTv3_DIR}/Pointcept/libs/pointops" ]; then
    uv pip install -e "${PTv3_DIR}/Pointcept/libs/pointops" --no-build-isolation
else
    echo "[WARN] pointops not found at ${PTv3_DIR}/Pointcept/libs/pointops"
fi

echo "[7/7] urban-building editable install"
cd "${PROJECT_DIR}"

uv pip install -e ".[all]"

echo "[INFO] verifying environment..."
python - <<PY
import importlib
import sys

checks = [
    ("torch", "torch"),
    ("torch_scatter", "torch_scatter"),
    ("spconv.pytorch", "spconv.pytorch"),
    ("hydra", "hydra"),
    ("timm", "timm"),
    ("addict", "addict"),
    ("einops", "einops"),
]

failed = []

for import_name, label in checks:
    try:
        importlib.import_module(import_name)
        print(f"{label}: OK")
    except Exception as exc:
        failed.append((label, exc))
        print(f"{label}: FAIL -> {exc}")

try:
    import torch
    print("torch version:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
except Exception as exc:
    failed.append(("torch details", exc))
    print("torch details: FAIL ->", exc)

try:
    import pointops
    print("pointops: OK")
except Exception as exc:
    failed.append(("pointops", exc))
    print("pointops: FAIL ->", exc)

try:
    import flash_attn
    print("flash_attn: OK")
except Exception as exc:
    if ${FLASH_ATTN_OK} == 1:
        failed.append(("flash_attn", exc))
        print("flash_attn: FAIL even though install said success ->", exc)
    else:
        print("flash_attn: SKIPPED/FAILED BUILD")
        print("flash_attn note: set enable_flash: false in configs/model/ptv3.yaml")

if failed:
    print("\\n[ERR] Some required checks failed:")
    for label, exc in failed:
        print(f" - {label}: {exc}")
    sys.exit(1)

print("\\n[INFO] verification completed")
PY

echo "[DONE] activate with:"
echo "source ${VENV_DIR}/bin/activate"