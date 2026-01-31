# src/infer/export.py
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core.io import write_las, write_ply


def get_timestamp() -> str:
    return datetime.now().strftime("%y%m%d--%H%M")


def generate_output_name(task: str, suffix: str = ".las") -> str:
    timestamp = get_timestamp()
    return f"{task}_res-{timestamp}{suffix}"


def export_las(
    path: Union[str, Path],
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_las(
        path=path,
        xyz=xyz,
        intensity=intensity,
        rgb=rgb,
        labels=labels,
    )


def export_ply(
    path: Union[str, Path],
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_ply(
        path=path,
        xyz=xyz,
        rgb=rgb,
        intensity=intensity,
        labels=labels,
    )


def export_csv(
    path: Union[str, Path],
    data: List[Dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def export_npz(
    path: Union[str, Path],
    **data: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, **data)
