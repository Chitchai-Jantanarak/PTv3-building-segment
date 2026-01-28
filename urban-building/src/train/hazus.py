import torch
from src.common.checkpoint import save_ckpt
from src.datasets.builder import build_dataset
from src.models.hazus_head.model import HazusHead
from src.train._base import train_loop
from torch.utils.data import DataLoader


def train(cfg):
    dataset = build_dataset(cfg, task="hazus")
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size)

    model = HazusHead(cfg).to(cfg.run.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = torch.nn.CrossEntropyLoss()

    def loss_fn(model, batch):
        dlp = batch["dlp"].to(cfg.run.device)
        target = batch["label"].to(cfg.run.device)

        logits = model(dlp)
        return criterion(logits, target)

    train_loop(
        model,
        loader,
        optimizer,
        loss_fn,
        cfg,
        save_fn=lambda m, n: save_ckpt(m, cfg, n),
    )
