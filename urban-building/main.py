import importlib
import os

import hydra
from omegaconf import DictConfig

from src.core.io.seed import set_seed


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.run.seed)
    task_module = cfg.task.module
    trainer = importlib.import_module(task_module)
    trainer.train(cfg)


def check_dependencies(task_cfg, paths) -> None:
    for req in task_cfg.requires:
        ckpt = f"{paths.ckpt_root}/{req}/best.pth"
        if not os.path.exists(ckpt):
            raise ValueError(f"[ERROR] Missing dependency: {req}")


if __name__ == "__main__":
    main()
