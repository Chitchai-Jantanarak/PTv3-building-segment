"""
Core module for the Urban Building Pipeline.

Provides:
- io: Point cloud and raster file I/O (LAS, PLY, DEM)
- preprocessing: Voxelization, feature computation, preprocessing pipeline
- utils: Logging, memory management, seeding utilities
"""

from . import io, preprocessing, utils
from .io import (
    compute_rel_z,
    get_h5_keys,
    get_las_bounds,
    read_dem,
    read_h5,
    read_las,
    read_ply,
    sample_dem_at_points,
    write_h5,
    write_las,
    write_ply,
)
from .preprocessing import (
    Preprocessor,
    compute_grid_coords,
    denormalize_features,
    denormalize_xyz,
    inverse_voxelize,
    load_preprocessed,
    normalize_features,
    normalize_xyz,
    preprocess_file,
    voxelize,
)
from .utils import (
    clear_cuda_cache,
    get_gpu_memory,
    get_latest_ckpt,
    get_seed,
    load_ckpt,
    log_memory,
    save_ckpt,
    set_seed,
)

__all__ = [
    "read_las",
    "write_las",
    "get_las_bounds",
    "read_ply",
    "write_ply",
    "read_dem",
    "compute_rel_z",
    "sample_dem_at_points",
    "read_h5",
    "write_h5",
    "get_h5_keys",
    "voxelize",
    "compute_grid_coords",
    "inverse_voxelize",
    "Preprocessor",
    "preprocess_file",
    "load_preprocessed",
    "normalize_xyz",
    "denormalize_xyz",
    "normalize_features",
    "denormalize_features",
    "set_seed",
    "get_seed",
    "Logger",
    "get_logger",
    "get_gpu_memory",
    "clear_cuda_cache",
    "log_memory",
    "save_ckpt",
    "load_ckpt",
    "get_latest_ckpt",
]
