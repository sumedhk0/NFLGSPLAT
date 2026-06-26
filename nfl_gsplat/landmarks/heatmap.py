"""Gaussian heatmap rendering (training targets) + subpixel peak extraction."""
from __future__ import annotations

import numpy as np


def render_gaussian(hw, uv, sigma: float) -> np.ndarray:
    """(H,W) float32 heatmap, peak 1.0 at ``uv=(x,y)`` (image coords in heatmap res)."""
    h, w = hw
    x, y = uv
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def extract_peak(heat, *, thresh: float):
    """Argmax + 3×3 centroid subpixel refine. Returns ((u,v), conf) or None."""
    h, w = heat.shape
    idx = int(np.argmax(heat))
    iy, ix = divmod(idx, w)
    conf = float(heat[iy, ix])
    if conf < thresh:
        return None
    x0, x1 = max(0, ix - 1), min(w, ix + 2)
    y0, y1 = max(0, iy - 1), min(h, iy + 2)
    patch = heat[y0:y1, x0:x1].astype(np.float64)
    s = patch.sum()
    if s <= 1e-9:
        return (float(ix), float(iy)), conf
    yy, xx = np.mgrid[y0:y1, x0:x1]
    u = float((xx * patch).sum() / s)
    v = float((yy * patch).sum() / s)
    return (u, v), conf
