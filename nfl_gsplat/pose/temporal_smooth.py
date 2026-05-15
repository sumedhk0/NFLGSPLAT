"""1€ filter for smoothing per-frame SMPL-X parameter sequences.

Reference: Casiez, Roussel, Vogel, "1€ Filter: a simple speed-based low-pass
filter for noisy input in interactive systems," CHI 2012.

Two tuning parameters per stream:

- ``min_cutoff`` — low-speed cutoff Hz. Lower = more smoothing at rest.
- ``beta``      — velocity sensitivity. Higher = filter opens faster when the
                  signal moves.

We apply the filter independently to each scalar in ``(body_pose, global_orient,
transl)``. Broadcast NFL is 30 fps so times default to uniform ``1/30 s``.

This module is numpy-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OneEuroConfig:
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    fps: float = 30.0


def _alpha(cutoff: float, dt: float) -> float:
    # Exponential-smoothing alpha for a given cutoff frequency and sample period.
    tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-9))


def one_euro_1d(x: np.ndarray, cfg: OneEuroConfig) -> np.ndarray:
    """Smooth a 1-D sequence with the 1€ filter. NaNs are carried through
    unchanged so gap-interpolation stays upstream of smoothing."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    y = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return y
    dt = 1.0 / cfg.fps

    last_x: float | None = None
    last_y: float | None = None
    last_dx: float = 0.0

    for i in range(n):
        xi = x[i]
        if not np.isfinite(xi):
            y[i] = xi
            continue
        if last_x is None:
            last_x = xi
            last_y = xi
            y[i] = xi
            continue
        dx = (xi - last_x) / dt
        a_d = _alpha(cfg.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * last_dx
        cutoff = cfg.min_cutoff + cfg.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        yi = a * xi + (1.0 - a) * (last_y if last_y is not None else xi)
        y[i] = yi
        last_x = xi
        last_y = yi
        last_dx = dx_hat
    return y


def smooth_param_sequence(
    params: np.ndarray,
    cfg: OneEuroConfig,
) -> np.ndarray:
    """Apply ``one_euro_1d`` independently to each scalar channel of a
    ``[T, D]`` sequence. Returns a new ``[T, D]`` array."""
    params = np.asarray(params, dtype=np.float64)
    if params.ndim == 1:
        return one_euro_1d(params, cfg)
    T, D = params.shape
    out = np.empty_like(params)
    for d in range(D):
        out[:, d] = one_euro_1d(params[:, d], cfg)
    return out


def interpolate_short_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    max_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate ``values[t]`` across gaps of length ≤ ``max_gap``.

    ``values`` has shape ``[T, ...]`` and ``valid`` has shape ``[T]`` (or a
    per-element mask that reduces over trailing axes with ``.all``; we
    accept either). Gaps longer than ``max_gap`` are left as NaN.

    Returns ``(filled_values, new_valid)``.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim > 1:
        valid = valid.all(axis=tuple(range(1, valid.ndim)))
    T = values.shape[0]
    assert valid.shape == (T,), "valid must reduce to shape (T,)"

    filled = values.copy()
    new_valid = valid.copy()

    # Walk runs of False.
    i = 0
    while i < T:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < T and not valid[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < T:
            a = filled[i - 1]
            b = filled[j]
            for k in range(gap_len):
                alpha = (k + 1) / (gap_len + 1)
                filled[i + k] = (1.0 - alpha) * a + alpha * b
                new_valid[i + k] = True
        i = j

    return filled, new_valid
