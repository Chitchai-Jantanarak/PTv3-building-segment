# src/core/io/ply.py
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from plyfile import PlyData, PlyElement


def read_ply(
    path: Union[str, Path],
    fields: Optional[list] = None,
) -> Dict[str, np.ndarray]:
    path = Path(path)
    ply = PlyData.read(path)
    vertex = ply["vertex"]

    data = {
        "xyz": np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(
            np.float32
        ),
    }

    if "red" in vertex.data.dtype.names and (fields is None or "rgb" in fields):
        r = vertex["red"].astype(np.float32) / 255.0
        g = vertex["green"].astype(np.float32) / 255.0
        b = vertex["blue"].astype(np.float32) / 255.0
        data["rgb"] = np.stack([r, g, b], axis=-1)

    if "intensity" in vertex.data.dtype.names and (
        fields is None or "intensity" in fields
    ):
        data["intensity"] = vertex["intensity"].astype(np.float32)

    if "label" in vertex.data.dtype.names and (fields is None or "labels" in fields):
        data["labels"] = vertex["label"].astype(np.int32)
    elif "class" in vertex.data.dtype.names and (fields is None or "labels" in fields):
        data["labels"] = vertex["class"].astype(np.int32)

    return data


def write_ply(
    path: Union[str, Path],
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = xyz.shape[0]
    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    if rgb is not None:
        dtypes.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    if intensity is not None:
        dtypes.append(("intensity", "f4"))
    if labels is not None:
        dtypes.append(("label", "i4"))

    vertex = np.empty(n, dtype=dtypes)
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]

    if rgb is not None:
        vertex["red"] = (rgb[:, 0] * 255).astype(np.uint8)
        vertex["green"] = (rgb[:, 1] * 255).astype(np.uint8)
        vertex["blue"] = (rgb[:, 2] * 255).astype(np.uint8)

    if intensity is not None:
        vertex["intensity"] = intensity

    if labels is not None:
        vertex["label"] = labels

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(path)
