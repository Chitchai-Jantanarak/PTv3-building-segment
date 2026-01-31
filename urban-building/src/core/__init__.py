"""
Core module for the Urban Building Pipeline.

Provides:
- io: Point cloud and raster file I/O (LAS, PLY, DEM)
- preprocessing: Voxelization, feature computation, preprocessing pipeline
- utils: Logging, memory management, seeding utilities
"""

from . import io, preprocessing, utils

# Re-export commonly used items
from .io import (
    # DEM
    DEMData,
    DEMHandler,
    # LAS
    LASData,
    # PLY
    PLYData,
    compute_relative_height,
    get_las_info,
    get_ply_info,
    load_dem,
    load_las,
    load_ply,
    query_dem_height,
    write_las,
    write_ply,
)
from .preprocessing import (
    # Preprocessing
    PreprocessingPipeline,
    ProcessedData,
    # Voxelization
    VoxelData,
    compute_features,
    create_blocks,
    devoxelize,
    load_point_cloud,
    mask_blocks,
    preprocess,
    preprocess_file,
    structured_masking,
    voxelize,
)
from .utils import (
    MemoryMonitor,
    TensorBoardLogger,
    clear_all,
    clear_cache,
    get_gpu_memory_info,
    # Logging
    get_logger,
    get_seed_generator,
    log_config,
    # Memory
    log_memory,
    log_metrics,
    memory_tracker,
    # Seed
    set_seed,
    setup_logging,
    worker_init_fn,
)

__all__ = [
    # Submodules
    "io",
    "preprocessing",
    "utils",
    # LAS
    "LASData",
    "load_las",
    "write_las",
    "get_las_info",
    # PLY
    "PLYData",
    "load_ply",
    "write_ply",
    "get_ply_info",
    # DEM
    "DEMData",
    "DEMHandler",
    "load_dem",
    "query_dem_height",
    "compute_relative_height",
    # Preprocessing
    "PreprocessingPipeline",
    "ProcessedData",
    "preprocess",
    "preprocess_file",
    "load_point_cloud",
    "compute_features",
    # Voxelization
    "VoxelData",
    "voxelize",
    "devoxelize",
    "create_blocks",
    "mask_blocks",
    "structured_masking",
    # Logging
    "get_logger",
    "setup_logging",
    "log_config",
    "log_metrics",
    "TensorBoardLogger",
    # Memory
    "log_memory",
    "clear_all",
    "clear_cache",
    "get_gpu_memory_info",
    "memory_tracker",
    "MemoryMonitor",
    # Seed
    "set_seed",
    "worker_init_fn",
    "get_seed_generator",
]
