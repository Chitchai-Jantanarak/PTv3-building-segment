# src/core/utils/logging.py
import sys
from datetime import datetime
from typing import Optional


class Logger:
    def __init__(self, name: str = "MAIN", log_file: Optional[str] = None):
        self.name = name
        self.log_file = log_file

    def _format(self, prefix: str, msg: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{timestamp} {prefix} {msg}"

    def _write(self, text: str) -> None:
        print(text, file=sys.stdout, flush=True)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(text + "\n")

    def info(self, msg: str) -> None:
        self._write(self._format("[INFO]", msg))

    def error(self, msg: str) -> None:
        self._write(self._format("[ERR]", msg))

    def warn(self, msg: str) -> None:
        self._write(self._format("[WARN]", msg))

    warning = warn

    def exception(self, msg: str) -> None:
        import traceback

        self._write(self._format("[ERR]", msg))
        self._write(traceback.format_exc())

    def epoch(self, n: int, msg: str) -> None:
        self._write(self._format(f"[EP {n:03d}]", msg))

    def debug(self, msg: str) -> None:
        self._write(self._format("[DBG]", msg))


def get_logger(name: str = "MAIN", log_file: Optional[str] = None) -> Logger:
    return Logger(name=name, log_file=log_file)


def setup_logging(cfg=None):
    import logging

    level = logging.INFO
    if cfg is not None and hasattr(cfg, "run"):
        level_name = getattr(cfg.run, "log_level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
