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


def split_two_teams_balanced(colors: np.ndarray, *, iters: int = 25,
                             seed: int = 0, tolerance: int = 2) -> np.ndarray:
    """2-means, then forced toward the 11-v-11 split the rules guarantee.

    Plain 2-means ignores a fact that is always true: both teams field the same
    number of players. Left free it happily returns 16-6, because one team's
    jersey is closer in HSV to the turf, or because a few tracks are lit
    differently -- and a lopsided split is worse than useless downstream, where
    the team label is used to penalise cross-team jersey pairings.

    So the tracks are ordered by how strongly they prefer one cluster over the
    other, and the split is placed at the middle. Tracks with a strong
    preference keep it; only the genuinely ambiguous ones near the boundary get
    moved, which is exactly where a colour split should be overruled by
    counting.

    ``tolerance`` allows a real imbalance: the tracked set is not always exactly
    11 and 11 once officials or missed players are in it.
    """
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    k = colors.shape[0]
    if k < 4:
        return split_two_teams(colors, iters=iters, seed=seed)

    labels = split_two_teams(colors, iters=iters, seed=seed)
    counts = np.bincount(labels, minlength=2)
    if abs(int(counts[0]) - int(counts[1])) <= tolerance:
        return labels

    centers = np.stack([
        colors[labels == j].mean(axis=0) if (labels == j).any() else colors[0]
        for j in range(2)])
    margin = np.array([_hsv_distance(c, centers[0]) - _hsv_distance(c, centers[1])
                       for c in colors])
    order = np.argsort(margin)          # most cluster-0-like first
    out = np.ones(k, dtype=np.int64)
    out[order[: k // 2]] = 0
    return out


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


SATURATED = 1        # the label of the higher-saturation cluster: the coloured kit


def two_means_1d(x, iters: int = 30):
    """1-D two-means seeded at the 25th and 75th percentiles. Returns
    ``(centres, labels)``; a label is 1 above the midpoint of the centres."""
    x = np.asarray(x, float)
    if x.size == 0:
        raise ValueError("two_means_1d: no values")
    c = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    for _ in range(iters):
        lab = (np.abs(x - c[1]) < np.abs(x - c[0])).astype(int)
        new = c.copy()
        for j in (0, 1):
            if (lab == j).any():
                new[j] = x[lab == j].mean()
        if np.allclose(new, c):
            break
        c = new
    return c, (x > 0.5 * (c[0] + c[1])).astype(int)


def split_by_saturation_votes(sats_by_cam, *, min_detections: int = 8):
    """Rule D: per CAMERA a 1-D two-means on every detection's torso
    saturation; each detection votes for its cluster; a track's label is the
    majority of its detections' votes over the cameras (a tie goes to the
    side its mean normalised margin lands on). ``SATURATED`` (1) is the
    higher-saturation cluster in EVERY camera, so the label means the same
    kit in both views without a global fit.

    ``sats_by_cam``: ``{cam: (keys, S)}`` -- per detection its track key (any
    hashable, e.g. ``(cam, track_id)``) and its saturation (0-255); NaN
    abstains. Returns ``({key: label}, {cam: (centre_lo, centre_hi)})``.

    Why per camera and per detection (measured 2026-09-05 against
    crop-verified kits on plays 1, 2 and 4): the two cameras expose
    differently, so one global split mixes exposure with team; and ONE crop
    per track was 44-87 % right where per-detection votes were 88-100 %.
    """
    counts: dict = {}
    margin: dict = {}
    centres = {}
    for cam, (keys, sat) in sats_by_cam.items():
        sat = np.asarray(sat, float)
        keys = list(keys)
        if len(keys) != len(sat):
            raise ValueError(f"{cam}: {len(keys)} keys for {len(sat)} saturations")
        ok = np.isfinite(sat)
        if int(ok.sum()) < min_detections:
            raise ValueError(f"{cam}: only {int(ok.sum())} detections carry a torso colour; "
                             f"{min_detections} are needed to split the kits")
        c, lab = two_means_1d(sat[ok])
        centres[cam] = (float(c[0]), float(c[1]))
        mid, span = 0.5 * (c[0] + c[1]), max(float(c[1] - c[0]), 1e-6)
        for k, l, v in zip([k for k, o in zip(keys, ok) if o], lab, sat[ok]):
            counts.setdefault(k, [0, 0])[int(l)] += 1
            margin[k] = margin.get(k, 0.0) + (v - mid) / span
    labels = {}
    for k, (n0, n1) in counts.items():
        if n1 != n0:
            labels[k] = SATURATED if n1 > n0 else 1 - SATURATED
        else:
            labels[k] = SATURATED if margin[k] > 0 else 1 - SATURATED
    return labels, centres


def votes_from_margins(keys, margins):
    """``{key: label}`` from per-detection signed saturation margins (the
    quantity ``split_by_saturation_votes`` computes per camera): each finite
    margin votes its sign, the majority wins, a tie goes to the mean."""
    margins = np.asarray(margins, float)
    counts: dict = {}
    total: dict = {}
    for k, m in zip(keys, margins):
        if not np.isfinite(m):
            continue
        counts.setdefault(k, [0, 0])[int(m > 0)] += 1
        total[k] = total.get(k, 0.0) + m
    out = {}
    for k, (n0, n1) in counts.items():
        if n1 != n0:
            out[k] = SATURATED if n1 > n0 else 1 - SATURATED
        else:
            out[k] = SATURATED if total[k] > 0 else 1 - SATURATED
    return out
