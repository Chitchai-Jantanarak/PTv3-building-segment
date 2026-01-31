# src/train/_base.py
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.core.utils import Logger, clear_cuda_cache, log_memory, save_ckpt


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device,
    epoch: int,
    logger: Logger,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        loss = criterion(model, batch, device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")

    return total_loss / max(n_batches, 1)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss = criterion(model, batch, device)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_loop(
    cfg: DictConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    criterion: Callable,
    logger: Logger,
) -> nn.Module:
    device = torch.device(cfg.run.device)
    model = model.to(device)

    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.task.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, cfg.task.epochs + 1):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger,
        )

        logger.epoch(epoch, f"Train Loss: {train_loss:.6f}")

        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, criterion, device)
            logger.epoch(epoch, f"Val Loss: {val_loss:.6f}")
            current_loss = val_loss
        else:
            current_loss = train_loss

        if scheduler is not None:
            scheduler.step()
            logger.info(f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if current_loss < best_loss:
            best_loss = current_loss
            save_ckpt(model, optimizer, epoch, ckpt_dir, best=True)
            logger.info(f"Best model saved at epoch {epoch}")

        if epoch % 10 == 0:
            save_ckpt(model, optimizer, epoch, ckpt_dir, best=False)

        if cfg.runtime.clear_cuda_cache_each_epoch:
            clear_cuda_cache()

        if cfg.runtime.log_memory_each_epoch:
            log_memory(logger)

    return model


def build_optimizer(cfg: DictConfig, model: nn.Module) -> Optimizer:
    opt_type = cfg.task.optimizer.type.lower()
    lr = cfg.task.optimizer.lr
    weight_decay = cfg.task.optimizer.weight_decay

    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(cfg: DictConfig, optimizer: Optimizer) -> Optional[_LRScheduler]:
    sched_type = cfg.task.scheduler.type.lower()

    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.task.epochs - cfg.task.scheduler.warmup_epochs,
        )
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.task.scheduler.step_size,
            gamma=cfg.task.scheduler.gamma,
        )
    else:
        return None
