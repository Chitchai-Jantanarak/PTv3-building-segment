# src/train/mae.py
import torch
from src.common.checkpoint import save_ckpt
from src.datasets.builder import build_dataset
from src.models.mae.model import MAEModel
from src.train._base import train_loop
from torch.utils.data import DataLoader


def train(cfg):
    dataset = build_dataset(cfg, task="mae")
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    model = MAEModel(cfg).to(cfg.run.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    def loss_fn(model, batch):
        pts = batch["points"].to(cfg.run.device)
        return model(pts)

    train_loop(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        cfg=cfg,
        save_fn=lambda m, n: save_ckpt(m, cfg, n),
    )
