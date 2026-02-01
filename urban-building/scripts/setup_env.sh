#!/bin/bash
# scripts/setup_env.sh
set -e

ENV_NAME="urban-building"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[ENVS] Starting environment setup..."

if ! command -v conda &> /dev/null; then
    echo "[ERR] conda not found. Please install miniconda or anaconda."
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "[INFO] Creating conda environment: ${ENV_NAME}"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo "[INFO] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "[INFO] Installing PyTorch with CUDA ${CUDA_VERSION} (conda only for CUDA base)..."
conda install pytorch=2.1.0 torchvision=0.16.0 pytorch-cuda=${CUDA_VERSION} mkl=2023 -c pytorch -c nvidia -y

echo "[INFO] Creating .venv symlink to conda env..."
CONDA_ENV_PATH=$(conda info --envs | grep ${ENV_NAME} | awk '{print $NF}')
ln -sf "${CONDA_ENV_PATH}" "${PROJECT_DIR}/.venv"

echo "[INFO] Installing all dependencies with uv..."
cd "${PROJECT_DIR}"

uv pip install spconv-cu118
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
uv pip install timm>=0.9.0 addict
uv pip install flash-attn --no-build-isolation || echo "[WARN] flash-attn failed, continuing"

uv pip install -e ".[all]"

echo "[INFO] Verifying installation..."
python -c "import torch; print(f'[INFO] PyTorch: {torch.__version__}')"
python -c "import torch; print(f'[INFO] CUDA: {torch.cuda.is_available()}')"
python -c "import spconv; print('[INFO] spconv: OK')"
python -c "import torch_scatter; print('[INFO] torch-scatter: OK')"
python -c "import timm; print('[INFO] timm: OK')"
python -c "import hydra; print('[INFO] hydra: OK')"

echo "[DONE] Environment setup complete."
echo "[INFO] Activate with: conda activate ${ENV_NAME}"
