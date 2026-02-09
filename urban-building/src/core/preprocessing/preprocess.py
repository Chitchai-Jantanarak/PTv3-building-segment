# src/core/preprocessing/preprocess.py
from pathlib import Path
from typing import Optional, Union

import numpy as np
from omegaconf import DictConfig

from src.core.io.dem import compute_rel_z, read_dem
from src.core.io.las import read_las
from src.core.io.ply import read_ply
from src.core.preprocessing.voxel import compute_grid_coords, voxelize


class Preprocessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.grid_size = cfg.data.grid_size
        self.dem = None
        self.dem_meta = None

    def load_dem(self, dem_path: Union[str, Path]) -> None:
        self.dem, self.dem_meta = read_dem(dem_path)

    def load_point_cloud(self, path: Union[str, Path]) -> dict[str, np.ndarray]:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in [".las", ".laz"]:
            data = read_las(path)
        elif suffix == ".ply":
            data = read_ply(path)
        elif suffix in [".h5", ".hdf5"]:
            data = read_h5(path)
            if "xyz" not in data and "x" in data:
                data["xyz"] = np.stack([data["x"], data["y"], data["z"]], axis=-1)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return data

    def process(
        self,
        data: dict[str, np.ndarray],
        voxelize_data: bool = True,
    ) -> dict[str, np.ndarray]:
        xyz = data["xyz"].astype(np.float32)

        rel_z = compute_rel_z(xyz, self.dem, self.dem_meta)
        data["rel_z"] = rel_z.astype(np.float32)

        features = [xyz, rel_z[:, np.newaxis]]

        if "intensity" in data and self.cfg.data.features.intensity:
            features.append(data["intensity"][:, np.newaxis])

        data["features"] = np.concatenate(features, axis=-1).astype(np.float32)

        if voxelize_data:
            voxel_result = voxelize(
                xyz=xyz,
                voxel_size=self.grid_size,
                features=data["features"],
                labels=data.get("labels"),
                mode="random",
            )
            data.update(voxel_result)

        data["grid_coords"] = compute_grid_coords(xyz, self.grid_size)

        return data

    def normalize_coords(
        self,
        xyz: np.ndarray,
        center: bool = True,
        scale: bool = True,
    ) -> np.ndarray:
        xyz = xyz.copy()

        if center:
            centroid = xyz.mean(axis=0)
            xyz = xyz - centroid

        if scale:
            max_dist = np.abs(xyz).max()
            if max_dist > 0:
                xyz = xyz / max_dist

        return xyz


def read_h5(path: Union[str, Path]) -> dict[str, np.ndarray]:
    import h5py

    path = Path(path)
    data = {}
    with h5py.File(path, "r") as f:
        for key in f:
            data[key] = np.array(f[key])
    return data


def preprocess_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    cfg: DictConfig,
    dem_path: Optional[Union[str, Path]] = None,
) -> None:
    preprocessor = Preprocessor(cfg)

    if dem_path:
        preprocessor.load_dem(dem_path)

    data = preprocessor.load_point_cloud(input_path)
    data = preprocessor.process(data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def load_preprocessed(path: Union[str, Path]) -> dict[str, np.ndarray]:
    path = Path(path)
    data = dict(np.load(path, allow_pickle=True))
    return data
