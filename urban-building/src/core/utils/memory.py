import gc

import psutil
import torch


def log_memory(tag="") -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{tag}]: Alloc {alloc:.2f} GB, Reserved {reserved:.2f} GB")

    ram = psutil.virtual_memory().used / 1e9
    print(f"[{tag}]: RAM used {ram:.2f} GB")


def clear_all() -> None:
    gc.collect()
    torch.cuda.empty_cache()
