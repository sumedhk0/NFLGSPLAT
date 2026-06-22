"""Data model for detected field features + landmark-name mapping.

Bridges raw detections (image-space lines/hashes/numbers) to the named
``NFL_LANDMARKS`` correspondences the PnP solver consumes. Pure / CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YardLineSeg:
    """A detected painted yard line as an image segment (two endpoints)."""
    p0: tuple[float, float]
    p1: tuple[float, float]


@dataclass(frozen=True)
class OCRNumber:
    """A painted yard number read by OCR: value is the multiple of 10 (10..50)."""
    value: int
    center: tuple[float, float]


@dataclass(frozen=True)
class DetectedFeatures:
    yard_lines: list[YardLineSeg]
    sidelines: list[YardLineSeg]
    hashes: list[tuple[float, float]]
    numbers: list[OCRNumber]
    image_size: tuple[int, int]


def yardline_label(side: str, yd: int) -> tuple[str, int]:
    """Normalize a (side, yard) pair. ``side`` in {home, away, mid}; mid → 50."""
    if yd == 50:
        return ("mid", 50)
    if side not in ("home", "away"):
        raise ValueError(f"side must be home/away/mid, got {side!r}")
    if yd < 5 or yd > 45 or yd % 5 != 0:
        raise ValueError(f"yard {yd} invalid (5..45 step 5)")
    return (side, yd)


def landmark_name(side: str, yd: int, lr: str, row: str) -> str:
    """Build an NFL_LANDMARKS name: ``{side}_{yd}_{lr}_{row}`` (mid_50_...)."""
    s, y = yardline_label(side, yd)
    if lr not in ("left", "right") or row not in ("hash", "sideline"):
        raise ValueError(f"bad lr/row: {lr}/{row}")
    base = "mid_50" if s == "mid" else f"{s}_{y}"
    return f"{base}_{lr}_{row}"
