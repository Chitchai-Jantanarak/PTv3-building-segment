# src/core/io/las.py
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import laspy
import numpy as np


def read_las(
    path: Union[str, Path],
    fields: Optional[list] = None,
) -> Dict[str, np.ndarray]:
    path = Path(path)
    las = laspy.read(path)

    data = {
        "xyz": np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32),
    }

    if hasattr(las, "intensity") and (fields is None or "intensity" in fields):
        data["intensity"] = las.intensity.astype(np.float32) / 65535.0

    if hasattr(las, "red") and (fields is None or "rgb" in fields):
        r = las.red.astype(np.float32) / 65535.0
        g = las.green.astype(np.float32) / 65535.0
        b = las.blue.astype(np.float32) / 65535.0
        data["rgb"] = np.stack([r, g, b], axis=-1)

    if hasattr(las, "classification") and (fields is None or "labels" in fields):
        data["labels"] = las.classification.astype(np.int32)

    return data


def write_las(
    path: Union[str, Path],
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    extra_dims: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = laspy.LasHeader(point_format=2, version="1.2")
    if rgb is not None:
        header = laspy.LasHeader(point_format=3, version="1.2")

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    if intensity is not None:
        las.intensity = (intensity * 65535).astype(np.uint16)

    if rgb is not None:
        las.red = (rgb[:, 0] * 65535).astype(np.uint16)
        las.green = (rgb[:, 1] * 65535).astype(np.uint16)
        las.blue = (rgb[:, 2] * 65535).astype(np.uint16)

    if labels is not None:
        las.classification = labels.astype(np.uint8)

    las.write(path)


def get_las_bounds(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    las = laspy.read(path)
    mins = np.array([las.x.min(), las.y.min(), las.z.min()])
    maxs = np.array([las.x.max(), las.y.max(), las.z.max()])
    return mins, maxs
