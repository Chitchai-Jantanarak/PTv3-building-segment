# src/infer/seg_b.py
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.infer.base import InferenceEngine, prepare_input
from src.models.seg_heads import SegBColorModel, SegBGeomModel


class SegBGeomInference(InferenceEngine):
    def _build_model(self) -> nn.Module:
        return SegBGeomModel(self.cfg)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        data = prepare_input(data, self.device)

        feat = data["features"]
        coord = data["coords"]
        batch = data.get("batch")

        if batch is None:
            batch = torch.zeros(feat.shape[0], dtype=torch.long, device=self.device)

        output = self.model(feat, coord, batch)

        return {
            "xyz_pred": output["xyz_pred"].cpu(),
            "features": output["features"].cpu(),
        }


class SegBColorInference(InferenceEngine):
    def _build_model(self) -> nn.Module:
        return SegBColorModel(self.cfg)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        data = prepare_input(data, self.device)

        feat = data["features"]
        coord = data["coords"]
        batch = data.get("batch")

        if batch is None:
            batch = torch.zeros(feat.shape[0], dtype=torch.long, device=self.device)

        output = self.model(feat, coord, batch)

        return {
            "xyz_pred": output["xyz_pred"].cpu(),
            "rgb_pred": output["rgb_pred"].cpu(),
            "features": output["features"].cpu(),
        }


def run_seg_b_inference(
    cfg: DictConfig,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    with_color: bool = False,
) -> None:
    from src.core.io import read_las, read_ply
    from src.core.preprocessing import Preprocessor
    from src.infer.export import export_las, export_ply

    if with_color:
        engine = SegBColorInference(cfg, checkpoint_path)
    else:
        engine = SegBGeomInference(cfg, checkpoint_path)

    input_path = Path(input_path)
    if input_path.suffix.lower() in [".las", ".laz"]:
        data = read_las(input_path)
    else:
        data = read_ply(input_path)

    preprocessor = Preprocessor(cfg)
    data = preprocessor.process(data, voxelize_data=False)

    result = engine.predict(data)

    if with_color and "rgb_pred" in result:
        export_ply(
            output_path,
            xyz=result["xyz_pred"].numpy(),
            rgb=result["rgb_pred"].numpy(),
        )
    else:
        export_las(
            output_path,
            xyz=result["xyz_pred"].numpy(),
        )

    engine.logger.info(f"Results saved to: {output_path}")
