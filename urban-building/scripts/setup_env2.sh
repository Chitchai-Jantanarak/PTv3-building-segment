set -euo pipefail


PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.7.0"
CUDA_TAG="cu128"                         
SPCONV_CUDA_TAG="cu124"                      
FLASH_ATTN_VERSION="2.7.4"                    
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PTv3_DIR="$(cd "${PROJECT_DIR}/../PointTransformerV3" && pwd)"

# PyG wheel index for torch-scatter
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_TAG}.html"

if ! nvidia-smi &>/dev/null; then
    echo "[ERR] nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
fi
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "[INFO] NVIDIA driver: ${DRIVER_VER}"

# Check nvcc 
if ! nvcc --version &>/dev/null; then
    echo "[WARN] nvcc not found. flash-attn source build will fail."
    echo "[WARN] Install CUDA toolkit 12.x or set CUDA_HOME."
fi

if ! command -v uv &>/dev/null; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "[INFO] uv: $(uv --version)"

if [ -d "${VENV_DIR}" ]; then
    echo "[WARN] ${VENV_DIR} already exists. Remove it first if you want a clean env."
    echo "[WARN] Run: rm -rf ${VENV_DIR}"
    exit 1
fi

echo "[INFO] Creating venv with Python ${PYTHON_VERSION}..."
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

source "${VENV_DIR}/bin/activate"
echo "[INFO] Python: $(python --version) @ $(which python)"

echo ""
echo "[1/7] Installing PyTorch ${PYTORCH_VERSION} + ${CUDA_TAG}..."
uv pip install \
    "torch==${PYTORCH_VERSION}" \
    "torchvision" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ""
echo "[2/7] Installing spconv-${SPCONV_CUDA_TAG}..."
uv pip install "spconv-${SPCONV_CUDA_TAG}"

echo ""
echo "[3/7] Installing torch-scatter..."
uv pip install torch-scatter -f "${PYG_WHEEL_URL}"

echo ""
echo "[4/7] Installing flash-attn..."

if ! uv pip install "flash-attn>=${FLASH_ATTN_VERSION}" 2>/dev/null; then
    echo "[INFO] No prebuilt wheel found, building flash-attn from source..."
    echo "[INFO] This may take 10-20 minutes. Go grab coffee."

    uv pip install packaging ninja psutil

    if [ -z "${CUDA_HOME:-}" ]; then
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME="/usr/local/cuda"
        elif [ -d "/usr/local/cuda-12.4" ]; then
            export CUDA_HOME="/usr/local/cuda-12.4"
        fi
        echo "[INFO] CUDA_HOME=${CUDA_HOME:-<not set>}"
    fi

    export MAX_JOBS=${MAX_JOBS:-4}

    uv pip install "flash-attn>=${FLASH_ATTN_VERSION}" --no-build-isolation \
        || echo "[WARN] flash-attn build failed — PTv3 will fall back to non-flash attention (slower but functional)"
fi

echo ""
echo "[5/7] Installing timm, addict, einops..."
uv pip install "timm>=0.9.0" addict einops

echo ""
echo "[6/7] Building Pointcept custom CUDA extensions..."

if [ -d "${PTv3_DIR}/Pointcept/libs/pointops" ]; then
    echo "[INFO] Building pointops..."
    uv pip install -e "${PTv3_DIR}/Pointcept/libs/pointops" --no-build-isolation
else
    echo "[WARN] pointops not found at ${PTv3_DIR}/Pointcept/libs/pointops"
fi

if [ -d "${PTv3_DIR}/Pointcept/libs/pointgroup_ops" ]; then
    echo "[INFO] Building pointgroup_ops..."
    uv pip install -e "${PTv3_DIR}/Pointcept/libs/pointgroup_ops" --no-build-isolation
else
    echo "[WARN] pointgroup_ops not found at ${PTv3_DIR}/Pointcept/libs/pointgroup_ops"
fi

echo ""
echo "[7/7] Installing urban-building project..."
cd "${PROJECT_DIR}"
uv pip install -e ".[all]"

