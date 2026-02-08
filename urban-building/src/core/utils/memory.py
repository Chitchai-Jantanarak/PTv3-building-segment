# src/core/utils/memory.py
from typing import Optional

import torch


def get_gpu_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "total": round(total, 2),
    }


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_memory(logger: Optional[object] = None) -> str:
    mem = get_gpu_memory()
    msg = f"GPU: {mem['allocated']:.2f}GB / {mem['total']:.2f}GB"
    if logger and hasattr(logger, "info"):
        logger.info(msg)
    return msg
