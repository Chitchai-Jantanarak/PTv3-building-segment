# main.py
import os
import sys
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")

import hydra
from omegaconf import DictConfig

from src.core.utils import get_logger, set_seed


def check_dependencies() -> bool:
    logger = get_logger("MAIN")

    try:
        import torch

        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA: {torch.cuda.is_available()}")
    except ImportError:
        logger.error("PyTorch not installed")
        return False

    try:
        import spconv  # noqa: F401

        logger.info("spconv: OK")
    except ImportError:
        logger.error("spconv not installed")
        return False

    try:
        import hydra  # noqa: F401

        logger.info("hydra: OK")
    except ImportError:
        logger.error("hydra not installed")
        return False

    return True


def dispatch_task(cfg: DictConfig) -> None:
    logger = get_logger("MAIN")
    task_name = cfg.task.name

    logger.info(f"Dispatching task: {task_name}")

    if task_name == "mae":
        from src.train import train_mae

        train_mae(cfg)
    elif task_name == "seg_a":
        from src.train import train_seg_a

        train_seg_a(cfg)
    elif task_name == "seg_b_geom":
        from src.train import train_seg_b_geom

        train_seg_b_geom(cfg)
    elif task_name == "seg_b_color":
        from src.train import train_seg_b_color

        train_seg_b_color(cfg)
    elif task_name == "seg_b_v2":
        from src.train import train_seg_b_v2

        train_seg_b_v2(cfg)
    elif task_name == "hazus":
        from src.train import train_hazus

        train_hazus(cfg)
    elif task_name == "infer":
        target = cfg.task.get("target", "pipeline")
        if target == "seg_a":
            from src.infer.seg_a import run_seg_a_inference

            ckpt = cfg.task.checkpoints.get("seg_a", "checkpoints/seg_a/best.pth")
            out = Path(cfg.task.output_dir) / "segmented.las"
            run_seg_a_inference(cfg, cfg.task.input_path, out, ckpt)
        else:
            from src.infer.pipeline import run_full_inference

            run_full_inference(cfg)
    else:
        logger.error(f"Unknown task: {task_name}")
        raise ValueError(f"Unknown task: {task_name}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = get_logger("MAIN")

    logger.info("Urban Building Pipeline")
    logger.info(f"Task: {cfg.task.name}")
    logger.info(f"Data: {cfg.data.name}")
    logger.info(f"Model: {cfg.model.name}")

    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)

    set_seed(cfg.run.seed, cfg.run.deterministic)
    logger.info(f"Seed: {cfg.run.seed}")

    dispatch_task(cfg)

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
