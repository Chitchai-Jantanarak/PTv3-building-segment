# src/train/_base.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.core.utils import Logger, clear_cuda_cache, log_memory, save_ckpt


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    device: torch.device,
    epoch: int,
    logger: Logger,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = criterion(model, batch, device)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    use_amp: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            with torch.cuda.amp.autocast(enabled=use_amp):
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
) -> TrainResult:
    device = torch.device(cfg.run.device)
    model = model.to(device)

    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.task.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = cfg.run.get("precision", "fp32") != "fp32"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    patience = cfg.task.get("early_stopping_patience", 0)
    no_improve = 0
    best_loss = float("inf")

    train_losses = []
    val_losses = []

    if train_loader is None:
        raise ValueError("No training data available. Check data path and file format.")

    for epoch in range(1, cfg.task.epochs + 1):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger,
            scaler=scaler,
            use_amp=use_amp,
        )

        logger.epoch(epoch, f"Train Loss: {train_loss:.6f}")
        train_losses.append(train_loss)

        if val_loader is not None:
            val_loss = validate_epoch(
                model, val_loader, criterion, device, use_amp=use_amp
            )
            logger.epoch(epoch, f"Val Loss: {val_loss:.6f}")
            val_losses.append(val_loss)
            current_loss = val_loss
        else:
            current_loss = train_loss

        if scheduler is not None:
            scheduler.step()
            logger.info(f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if current_loss < best_loss:
            best_loss = current_loss
            no_improve = 0
            save_ckpt(model, optimizer, epoch, ckpt_dir, best=True)
            logger.info(f"Best model saved at epoch {epoch}")
        else:
            no_improve += 1

        if epoch % 10 == 0:
            save_ckpt(model, optimizer, epoch, ckpt_dir, best=False)

        if cfg.runtime.clear_cuda_cache_each_epoch:
            clear_cuda_cache()

        if cfg.runtime.log_memory_each_epoch:
            log_memory(logger)

        if patience > 0 and no_improve >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
            )
            break

    return TrainResult(model=model, train_losses=train_losses, val_losses=val_losses)


def build_optimizer(cfg: DictConfig, model: nn.Module) -> Optimizer:
    opt_type = cfg.task.optimizer.type.lower()
    lr = cfg.task.optimizer.lr
    weight_decay = cfg.task.optimizer.weight_decay

    # Only optimize trainable parameters -- frozen params must NOT receive
    # weight-decay updates (AdamW applies decay *before* the gradient step,
    # so even params with requires_grad=False get decayed every step).
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(cfg: DictConfig, optimizer: Optimizer) -> Optional[_LRScheduler]:
    sched_type = cfg.task.scheduler.type.lower()
    warmup_epochs = cfg.task.scheduler.get("warmup_epochs", 0)

    if sched_type == "cosine":
        main_epochs = max(cfg.task.epochs - warmup_epochs, 1)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs
        )
    elif sched_type == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.task.scheduler.step_size,
            gamma=cfg.task.scheduler.gamma,
        )
    else:
        return None

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

    return main_scheduler
