# src/core/io/dem.py
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator


def read_dem(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    path = Path(path)
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        bounds = src.bounds

        meta = {
            "transform": transform,
            "bounds": bounds,
            "crs": src.crs,
            "shape": dem.shape,
        }

    return dem, meta


def compute_rel_z(
    xyz: np.ndarray,
    dem: Optional[np.ndarray] = None,
    dem_meta: Optional[dict] = None,
) -> np.ndarray:
    if dem is None or dem_meta is None:
        ground_z = np.percentile(xyz[:, 2], 5)
        return xyz[:, 2] - ground_z

    transform = dem_meta["transform"]
    h, w = dem.shape

    x_coords = np.arange(w) * transform.a + transform.c
    y_coords = np.arange(h) * transform.e + transform.f

    interp = RegularGridInterpolator(
        (y_coords[::-1], x_coords),
        dem[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    ground_z = interp(xyz[:, :2][:, ::-1])
    nan_mask = np.isnan(ground_z)
    if nan_mask.any():
        ground_z[nan_mask] = np.percentile(xyz[~nan_mask, 2], 5)

    return xyz[:, 2] - ground_z


def sample_dem_at_points(
    xyz: np.ndarray,
    dem: np.ndarray,
    dem_meta: dict,
) -> np.ndarray:
    transform = dem_meta["transform"]
    h, w = dem.shape

    col = ((xyz[:, 0] - transform.c) / transform.a).astype(int)
    row = ((xyz[:, 1] - transform.f) / transform.e).astype(int)

    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)

    return dem[row, col]
