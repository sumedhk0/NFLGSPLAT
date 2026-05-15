"""Rich-based logging + VRAM/wall-clock timing decorator.

Every expensive function should be wrapped with ``@log_timing`` so we can
attribute slow stages without bolting on a profiler later.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str, level: str = "INFO", json_file: Path | str | None = None) -> logging.Logger:
    """Return a logger configured with rich console output and optional JSON sink."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.handlers.clear()

    console = RichHandler(rich_tracebacks=True, markup=False, show_path=False)
    console.setLevel(level)
    logger.addHandler(console)

    if json_file is not None:
        json_path = Path(json_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(json_path)
        file_h.setLevel(level)
        file_h.setFormatter(_JSONFormatter())
        logger.addHandler(file_h)

    logger.propagate = False
    _CONFIGURED = True
    return logger


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "t": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _current_vram_gb() -> float | None:
    """Return current-device VRAM-used in GB, or None if CUDA unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 ** 3)


@contextmanager
def log_timing_block(logger: logging.Logger, label: str) -> Iterator[None]:
    """Context manager: logs wall-clock seconds + VRAM delta around a block."""
    vram_start = _current_vram_gb()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        vram_end = _current_vram_gb()
        vram_msg = ""
        if vram_start is not None and vram_end is not None:
            vram_msg = f" vram {vram_start:.2f}->{vram_end:.2f} GB"
        logger.info(f"[{label}] {dt:.2f}s{vram_msg}")


def log_timing(label: str | None = None) -> Callable:
    """Decorator: wrap a function in ``log_timing_block``."""

    def _decorate(fn: Callable) -> Callable:
        tag = label or fn.__qualname__
        logger = get_logger(fn.__module__)

        @wraps(fn)
        def _wrap(*args: Any, **kwargs: Any) -> Any:
            with log_timing_block(logger, tag):
                return fn(*args, **kwargs)

        return _wrap

    return _decorate
