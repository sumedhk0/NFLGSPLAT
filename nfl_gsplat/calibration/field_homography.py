"""Robust field-line labeling via planar-homography consensus.

The per-camera hint pins one detected yard line's absolute yardage (the anchor).
Real yard lines are monotonic in world-X; the two hash rows are world Y = ±HASH.
A correct labeling makes every (yard-line × hash-row) point consistent with one
field→image homography; spurious lines (painted numbers, jersey scraps) fit no
consensus and are rejected. Hypotheses are enumerated deterministically — the
space is tiny, so no randomness is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelResult:
    correspondences: list[tuple[str, tuple[float, float]]]
    homography: np.ndarray | None
    inlier_count: int


def fit_plane_homography(world_xy, image_uv) -> np.ndarray | None:
    """Least-squares field→image homography (3×3) from ≥4 point pairs, else None."""
    import cv2
    world = np.asarray(world_xy, dtype=np.float64)
    image = np.asarray(image_uv, dtype=np.float64)
    if len(world) < 4 or len(image) != len(world):
        return None
    H, _ = cv2.findHomography(world, image, 0)
    return H
