import torch
from src.common.checkpoint import save_ckpt
from src.datasets.builder import build_dataset
from src.losses.chamfer import chamfer_loss
from src.models.seg_heads.inpaint import GeometryInpaintHead
from src.train._base import train_loop
from torch.utils.data import DataLoader


def train(cfg):
    dataset = build_dataset(cfg, task="seg_b_geom")
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size)

    model = GeometryInpaintHead(cfg).to(cfg.run.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    def loss_fn(model, batch):
        visible = batch["visible"].to(cfg.run.device)
        target = batch["target"].to(cfg.run.device)
        pred = model(visible)

        loss = chamfer_loss(pred, target)
        assert pred[:, :, 2].std() > 0.05  # no Z collapse
        return loss

    train_loop(
        model,
        loader,
        optimizer,
        loss_fn,
        cfg,
        save_fn=lambda m, n: save_ckpt(m, cfg, n),
    )
