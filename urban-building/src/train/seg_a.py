import torch
from src.common.checkpoint import save_ckpt
from src.datasets.builder import build_dataset
from src.models.seg_heads.semantic import SegmentationHead
from src.train._base import train_loop
from torch.utils.data import DataLoader


def train(cfg):
    dataset = build_dataset(cfg, task="seg_a")
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    model = SegmentationHead(cfg).to(cfg.run.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def loss_fn(model, batch):
        pts = batch["points"].to(cfg.run.device)
        labels = batch["labels"].to(cfg.run.device)
        logits = model(pts)
        return criterion(logits, labels)

    train_loop(
        model,
        loader,
        optimizer,
        loss_fn,
        cfg,
        save_fn=lambda m, n: save_ckpt(m, cfg, n),
    )
