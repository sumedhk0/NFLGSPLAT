"""Cheap CPU jersey-color features for team assignment and referee detection.

Two jobs, both from a player's torso crop, no GPU:

1. **Team split.** Per play, cluster tracks into two groups by dominant jersey
   color (k=2). Labels are arbitrary (0/1) until mapped to real teams via the
   per-game home/away colors.
2. **Referee detection.** Officials wear black-and-white vertical stripes — a
   distinctive signal: many alternating dark/bright vertical bands plus a
   near-grayscale (low-saturation) palette. Used to route non-roster tracks to
   the generic referee avatar instead of dropping them.

Crops are ``[H, W, 3]`` uint8 BGR (OpenCV convention). We sample the torso
(central vertical band) to avoid helmet / grass / pants contamination.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _torso_region(crop: np.ndarray) -> np.ndarray:
    """Central torso band: middle 50% width, upper-middle 25–60% height."""
    h, w = crop.shape[:2]
    y0, y1 = int(0.25 * h), int(0.60 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    region = crop[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    return region if region.size else crop


def dominant_jersey_color(crop: np.ndarray) -> np.ndarray:
    """Return the mean HSV color (3-vector, float) of the torso region.

    HSV is more robust than BGR to broadcast brightness changes.
    """
    region = _torso_region(np.asarray(crop, dtype=np.uint8))
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    return hsv.mean(axis=0)


def _hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hue is circular (0..180 in OpenCV); S/V are linear (0..255)."""
    dh = abs(a[0] - b[0])
    dh = min(dh, 180.0 - dh) * 2.0          # scale hue to comparable magnitude
    ds = a[1] - b[1]
    dv = a[2] - b[2]
    return float(np.sqrt(dh * dh + ds * ds + dv * dv))


def split_two_teams(colors: np.ndarray, *, iters: int = 25, seed: int = 0) -> np.ndarray:
    """2-means on HSV jersey colors. ``colors`` is ``[K, 3]``; returns ``[K]``
    int labels in {0, 1}. Fewer than 2 samples → all-zeros.
    """
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    k = colors.shape[0]
    if k < 2:
        return np.zeros(k, dtype=np.int64)

    rng = np.random.default_rng(seed)
    # Seed centers at the two most-distant samples for a stable, deterministic
    # split (k-means++ flavor without the randomness sensitivity).
    i0 = int(rng.integers(k))
    d0 = np.array([_hsv_distance(colors[i0], c) for c in colors])
    i1 = int(np.argmax(d0))
    centers = np.stack([colors[i0], colors[i1]])

    labels = np.zeros(k, dtype=np.int64)
    for _ in range(iters):
        dists = np.stack(
            [[_hsv_distance(c, centers[j]) for j in range(2)] for c in colors]
        )                                                       # [K, 2]
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(2):
            members = colors[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels


@dataclass(frozen=True)
class RefereeConfig:
    min_stripe_transitions: int = 4     # alternating dark/bright vertical bands
    max_mean_saturation: float = 60.0   # near-grayscale palette (0..255)
    dark_bright_gap: float = 40.0       # contrast between band extremes


def is_referee(crop: np.ndarray, cfg: RefereeConfig | None = None) -> bool:
    """Heuristic black-and-white-vertical-stripe detector.

    Looks for (a) a low-saturation (grayscale) torso and (b) several
    alternating dark/bright transitions across columns.
    """
    cfg = cfg or RefereeConfig()
    region = _torso_region(np.asarray(crop, dtype=np.uint8))
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    if float(hsv[..., 1].mean()) > cfg.max_mean_saturation:
        return False

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float64)
    col_mean = gray.mean(axis=0)                       # [W] per-column brightness
    if col_mean.size < 4 or (col_mean.max() - col_mean.min()) < cfg.dark_bright_gap:
        return False
    thresh = 0.5 * (col_mean.max() + col_mean.min())
    binary = col_mean > thresh
    transitions = int(np.sum(binary[1:] != binary[:-1]))
    return transitions >= cfg.min_stripe_transitions
