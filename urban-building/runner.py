"""
Urban Building Point Cloud Processing Pipeline
==============================================

Single Hydra entry point for:
- MAE pretraining
- SEG-A: Urban semantic segmentation
- SEG-B: Building inpainting (geometry + color)
- DLP/HAZUS: FEMA building classification

Usage:
    python runner.py task=mae
    python runner.py task=seg_a
    python runner.py task=seg_b_geom
    python runner.py task=seg_b_color
    python runner.py task=hazus
    python runner.py mode=preprocess data=sensat
    python runner.py mode=infer task=seg_a checkpoint=checkpoints/seg_a/best.pth
"""

import importlib
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.utils.logging import get_logger, setup_logging
from src.core.utils.seed import set_seed


def check_dependencies(cfg: DictConfig) -> None:
    """Verify that required checkpoints from previous stages exist."""
    task_cfg = cfg.task
    if not hasattr(task_cfg, "requires") or task_cfg.requires is None:
        return

    ckpt_root = Path(cfg.paths.checkpoints)
    missing = []

    for req in task_cfg.requires:
        ckpt_path = ckpt_root / req / "best.pth"
        if not ckpt_path.exists():
            missing.append(str(ckpt_path))

    if missing:
        raise FileNotFoundError(
            f"Missing required checkpoints for task '{cfg.task.name}':\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nPlease train the required stages first."
        )


def run_preprocess(cfg: DictConfig) -> None:
    """Run preprocessing pipeline."""
    logger = get_logger(__name__)
    logger.info(f"Starting preprocessing for dataset: {cfg.data.name}")

    from src.core.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(cfg)
    pipeline.run()

    logger.info("Preprocessing complete!")


def run_train(cfg: DictConfig) -> None:
    """Run training pipeline."""
    logger = get_logger(__name__)

    # Check dependencies
    check_dependencies(cfg)

    task_name = cfg.task.name
    logger.info(f"Starting training for task: {task_name}")

    # Dynamically import the training module
    train_module = importlib.import_module(f"src.train.{task_name}")
    train_module.train(cfg)

    logger.info(f"Training complete for task: {task_name}")


def run_inference(cfg: DictConfig) -> None:
    """Run inference pipeline."""
    logger = get_logger(__name__)

    task_name = cfg.task.name
    logger.info(f"Starting inference for task: {task_name}")

    # Check checkpoint exists
    ckpt_path = Path(cfg.inference.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Dynamically import the inference module
    infer_module = importlib.import_module(f"src.infer.{task_name}")
    infer_module.infer(cfg)

    logger.info(f"Inference complete for task: {task_name}")


def run_evaluate(cfg: DictConfig) -> None:
    """Run evaluation pipeline."""
    logger = get_logger(__name__)

    task_name = cfg.task.name
    logger.info(f"Starting evaluation for task: {task_name}")

    from src.eval.evaluator import Evaluator

    evaluator = Evaluator(cfg)
    metrics = evaluator.run()

    logger.info(f"Evaluation complete. Metrics:\n{OmegaConf.to_yaml(metrics)}")


def run_export(cfg: DictConfig) -> None:
    """Export model to ONNX format."""
    logger = get_logger(__name__)

    task_name = cfg.task.name
    logger.info(f"Exporting model for task: {task_name}")

    from src.export.onnx_export import export_to_onnx

    export_to_onnx(cfg)

    logger.info("Export complete!")


def run_full_pipeline(cfg: DictConfig) -> None:
    """Run complete pipeline: SEG-A -> SEG-B -> HAZUS."""
    logger = get_logger(__name__)
    logger.info("Starting full inference pipeline")

    from src.infer.pipeline import FullPipeline

    pipeline = FullPipeline(cfg)
    results = pipeline.run()

    logger.info("Full pipeline complete!")
    return results


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for all operations.

    Modes:
        - preprocess: Voxelize and prepare datasets
        - train: Train a specific task
        - infer: Run inference on data
        - eval: Evaluate model performance
        - export: Export model to ONNX
        - pipeline: Run full SEG-A -> SEG-B -> HAZUS pipeline
    """
    # Setup logging
    setup_logging(cfg)
    logger = get_logger(__name__)

    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    set_seed(cfg.run.seed)

    # Create necessary directories
    for dir_key in ["checkpoints", "logs", "cache"]:
        if hasattr(cfg.paths, dir_key):
            Path(cfg.paths[dir_key]).mkdir(parents=True, exist_ok=True)

    # Route to appropriate handler based on mode
    mode = cfg.mode

    try:
        if mode == "preprocess":
            run_preprocess(cfg)
        elif mode == "train":
            run_train(cfg)
        elif mode == "infer":
            run_inference(cfg)
        elif mode == "eval":
            run_evaluate(cfg)
        elif mode == "export":
            run_export(cfg)
        elif mode == "pipeline":
            run_full_pipeline(cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return None
    except Exception as e:
        logger.exception(f"Error during {mode}: {e}")
        raise

    return None


if __name__ == "__main__":
    main()
