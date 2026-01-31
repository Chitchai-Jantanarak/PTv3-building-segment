# src/infer/hazus.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.infer.base import InferenceEngine, prepare_input
from src.models.hazus_head import HazusModel, decode_predictions


class HazusInference(InferenceEngine):
    def _build_model(self) -> nn.Module:
        return HazusModel(self.cfg)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = prepare_input(data, self.device)

        xyz = data["coords"]
        batch = data["batch"]
        mae_errors = data.get("mae_errors")

        output = self.model(xyz, batch, mae_errors)

        decoded = decode_predictions(output["predictions"])

        return {
            "predictions": output["predictions"].cpu(),
            "logits": output["logits"].cpu(),
            "features": output["features"].cpu(),
            "decoded": decoded,
        }

    def predict_single(
        self, xyz: torch.Tensor, mae_error: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        xyz = xyz.to(self.device)
        if mae_error is not None:
            mae_error = mae_error.to(self.device)

        result = self.model.predict_single(xyz, mae_error)

        return {
            "prediction": result["prediction"].cpu(),
            "logits": result["logits"].cpu(),
            "features": result["features"].cpu(),
            "decoded": result["decoded"],
        }


def run_hazus_inference(
    cfg: DictConfig,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
) -> None:
    import pandas as pd
    from sklearn.cluster import DBSCAN

    from src.core.io import read_las, read_ply
    from src.core.preprocessing import Preprocessor
    from src.infer.export import export_csv, export_las

    engine = HazusInference(cfg, checkpoint_path)

    input_path = Path(input_path)
    if input_path.suffix.lower() in [".las", ".laz"]:
        data = read_las(input_path)
    else:
        data = read_ply(input_path)

    preprocessor = Preprocessor(cfg)
    data = preprocessor.process(data, voxelize_data=False)

    xyz = data["xyz"]

    clustering = DBSCAN(eps=2.0, min_samples=50)
    cluster_labels = clustering.fit_predict(xyz[:, :2])

    results = []
    unique_clusters = set(cluster_labels) - {-1}

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_xyz = torch.from_numpy(xyz[mask]).float()

        result = engine.predict_single(cluster_xyz)

        results.append(
            {
                "cluster_id": cluster_id,
                "n_points": mask.sum(),
                "prediction": result["decoded"]["code"],
                "name": result["decoded"]["name"],
            }
        )

    output_path = Path(output_path)
    csv_path = output_path.with_suffix(".csv")
    export_csv(csv_path, results)

    las_path = output_path.with_suffix(".las")
    building_labels = cluster_labels.copy()
    building_labels[cluster_labels == -1] = 0
    export_las(las_path, xyz=xyz, labels=building_labels)

    engine.logger.info(f"Results saved to: {csv_path} and {las_path}")
