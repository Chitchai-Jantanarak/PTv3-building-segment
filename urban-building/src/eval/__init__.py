# src/eval/__init__.py
from src.eval.evaluate import run_evaluation
from src.eval.metrics import (
    boundary_iou,
    chamfer_stats,
    confusion_matrix,
    height_wise_error,
    per_class_iou,
    spatial_error_grid,
)
from src.eval.plots import plot_all

__all__ = [
    "confusion_matrix",
    "per_class_iou",
    "boundary_iou",
    "chamfer_stats",
    "height_wise_error",
    "spatial_error_grid",
    "plot_all",
    "run_evaluation",
]
