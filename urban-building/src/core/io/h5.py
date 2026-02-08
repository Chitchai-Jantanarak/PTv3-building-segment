# src/core/io/h5.py
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np


def read_h5(
    path: Union[str, Path],
    keys: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    path = Path(path)
    data = {}

    with h5py.File(path, "r") as f:
        available_keys = list(f.keys())
        read_keys = keys if keys else available_keys

        for key in read_keys:
            if key in f:
                data[key] = np.array(f[key])

    if "coords" in data and "xyz" not in data:
        data["xyz"] = data["coords"].astype(np.float32)

    if "semantics" in data and "labels" not in data:
        data["labels"] = data["semantics"].astype(np.int32)

    return data


def write_h5(
    path: Union[str, Path],
    data: dict[str, np.ndarray],
    compression: Optional[str] = "gzip",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            if compression:
                f.create_dataset(key, data=arr, compression=compression)
            else:
                f.create_dataset(key, data=arr)


def get_h5_keys(path: Union[str, Path]) -> list[str]:
    path = Path(path)
    with h5py.File(path, "r") as f:
        return list(f.keys())
