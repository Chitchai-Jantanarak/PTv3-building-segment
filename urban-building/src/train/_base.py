import torch
from src.core.utils.logging import get_logger
from src.core.utils.memory import clear_all, log_memory


def train_loop(model, loader, optimizer, loss_fn, cfg, save_fn):
    logger = get_logger(__name__)
    best_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}: Loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_fn(model, "best.pth")

        save_fn(model, "last.pth")

        if cfg.runtime.log_memory_each_epoch:
            log_memory(f"Epoch {epoch + 1}")

        if cfg.runtime.clear_cuda_cache_each_epoch:
            clear_all()
