#!/bin/bash
# =============================================================================
# Setup Script for Flood Assessment (RTX 5090 / CUDA 12.4)
# =============================================================================
# This script performs the following:
# 1. CLEANS UP old environments and caches from the root disk to free space.
# 2. Creates a NEW Conda environment in /workspace (which has 800TB+ space).
# 3. Installs all dependencies for RTX 5090.
#
# Usage:
#   bash scripts/setup_env.sh
# =============================================================================

echo "==========================================================="
echo "STEP 1: CLEANUP ROOT DISK"
echo "==========================================================="

# Remove the old environment from the default location (if it exists)
echo "Removing old 'flood_assess' environment from default location..."
conda remove -n flood_assess --all -y || true

# Clean Conda cache (removes downloaded tarballs)
echo "Cleaning Conda package cache..."
conda clean --all -y

# Clean Pip cache
echo "Cleaning Pip cache..."
pip cache purge
rm -rf ~/.cache/pip
rm -rf /root/.cache/pip

# Clean Apt cache (if running as root)
if [ "$EUID" -eq 0 ]; then 
  echo "Cleaning Apt cache..."
  apt-get clean
  rm -rf /var/lib/apt/lists/*
fi

echo "Root disk cleanup complete. Current disk usage:"
df -h /

echo "==========================================================="
echo "STEP 2: SETUP NEW ENVIRONMENT IN /workspace"
echo "==========================================================="

# Configure Storage paths on the large volume
export WORKSPACE_DIR="/workspace"
export ENV_PREFIX="$WORKSPACE_DIR/envs/flood_assess"
export TMP_DIR="$WORKSPACE_DIR/tmp"
export CACHE_DIR="$WORKSPACE_DIR/.cache"

echo "Configuring storage to use $WORKSPACE_DIR..."
mkdir -p "$TMP_DIR"
mkdir -p "$CACHE_DIR/pip"
mkdir -p "$CACHE_DIR/conda_pkgs"

# Set Environment Variables for Build/Cache to force usage of /workspace
export TMPDIR="$TMP_DIR"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export CONDA_PKGS_DIRS="$CACHE_DIR/conda_pkgs"

# Create Conda Environment in /workspace
echo "Creating Conda environment in $ENV_PREFIX with Python 3.10..."
conda create --prefix "$ENV_PREFIX" python=3.10 -y

# Activate Environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Install PyTorch 2.5.1 + CUDA 12.4
echo "Installing PyTorch 2.5.1 (CUDA 12.4)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Core Dependencies
echo "Installing core dependencies..."
pip install ninja h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Install spconv (CUDA 12.x) & Open3D
echo "Installing spconv and Open3D..."
pip install spconv-cu120
pip install open3d
pip install laspy[lazrs] rasterio scikit-learn pandas tqdm ftfy regex

# Install Flash Attention
echo "Installing Flash Attention (using /workspace/tmp for compilation)..."
pip install flash-attn --no-build-isolation

echo "==========================================================="
echo "Setup complete!"
echo ""
echo "IMPORTANT: The environment is now located in $ENV_PREFIX"
echo "To activate this environment, you must use:"
echo ""
echo "    source \"$(conda info --base)/etc/profile.d/conda.sh\""
echo "    conda activate $ENV_PREFIX"
echo ""
echo "==========================================================="
