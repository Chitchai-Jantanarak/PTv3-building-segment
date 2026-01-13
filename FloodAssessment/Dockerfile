# Use PyTorch 2.1.0 with CUDA 11.8 as the base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libhdf5-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies from Pointcept v1.5 recommendations
RUN pip install --upgrade pip && \
    pip install \
    ninja \
    h5py \
    pyyaml \
    sharedarray \
    tensorboard \
    tensorboardx \
    yapf \
    addict \
    einops \
    scipy \
    plyfile \
    termcolor \
    timm \
    torch-geometric \
    spconv-cu118 \
    open3d \
    laspy[lazrs] \
    rasterio \
    scikit-learn \
    pandas \
    matplotlib \
    ftfy \
    regex \
    tqdm

# Install PyTorch Scatter/Sparse/Cluster manually to ensure compatibility
# These URLs might need adjustment based on exact PyTorch/CUDA versions
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Flash Attention installation (Optional but recommended for PTv3)
# Note: This often requires a long compile time.
RUN pip install flash-attn==2.4.2 --no-build-isolation

# Create workspace directory
WORKDIR /workspace

# Copy the PointTransformerV3 code (Assuming it's mounted or copied)
# We will mount it via docker-compose, but we need to compile pointops
# Since we don't have the files yet in this image during build, we can't compile it here
# UNLESS we copy it now. 
# Strategy: We will assume the user mounts the project and we run a compilation script on start 
# OR we clone Pointcept/PointTransformerV3 here to get the libs.
# Let's clone Pointcept to get the libs/pointops source for compilation if needed, 
# or we can rely on the 'DoingSomeWeirdLikeWeirdo/PointTransformerV3' directory being mounted.

# For now, let's keep the image cleaner and assume a startup script or manual compilation step 
# is acceptable for the custom pointops, OR we can try to install a pre-built wheel if available.
# Pointcept usually requires compilation.

# Let's add a script to compile pointops at runtime if not present
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
