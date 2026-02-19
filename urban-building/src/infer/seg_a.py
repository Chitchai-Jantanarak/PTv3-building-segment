# src/infer/seg_a.py
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.infer.base import InferenceEngine, prepare_input
from src.models.seg_heads import SegAModel


class SegAInference(InferenceEngine):
    def _build_model(self) -> nn.Module:
        return SegAModel(self.cfg)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        data = prepare_input(data, self.device)

        feat = data["features"]
        coord = data["coords"]
        batch = data.get("batch")

        if batch is None:
            batch = torch.zeros(feat.shape[0], dtype=torch.long, device=self.device)

        output = self.model(feat, coord, batch)
        predictions = torch.argmax(output["logits"], dim=-1)
        probs = torch.softmax(output["logits"], dim=-1)

        return {
            "predictions": predictions.cpu(),
            "probabilities": probs.cpu(),
            "features": output["features"].cpu(),
        }


def run_seg_a_inference(
    cfg: DictConfig,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
) -> None:
    from src.core.io import read_las, read_ply
    from src.core.preprocessing import Preprocessor
    from src.infer.export import export_las
    from src.infer.pipeline import _get_las_mapping, _remap_labels_to_las

    engine = SegAInference(cfg, checkpoint_path)

    input_path = Path(input_path)
    if input_path.suffix.lower() in [".las", ".laz"]:
        data = read_las(input_path)
    else:
        data = read_ply(input_path)

    original_xyz = data["xyz"].copy()

    preprocessor = Preprocessor(cfg)
    data = preprocessor.process(data, voxelize_data=False)

    # Center features[:,:3] to match training behavior
    coords = data["xyz"]
    centroid = coords.mean(axis=0)
    data["xyz"] = coords - centroid
    features = data["features"].copy()
    n_xyz = min(3, features.shape[1], centroid.shape[0])
    features[:, :n_xyz] -= centroid[:n_xyz]
    data["features"] = features

    result = engine.predict(data)

    # Remap internal class IDs to ASPRS LAS standard codes
    labels = result["predictions"].numpy()
    las_mapping = _get_las_mapping(cfg)
    labels_las = _remap_labels_to_las(labels, las_mapping)

    export_las(
        output_path,
        xyz=original_xyz,
        labels=labels_las,
    )

    engine.logger.info(f"Results saved to: {output_path}")
