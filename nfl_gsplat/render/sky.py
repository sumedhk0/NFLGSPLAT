"""A night-stadium backdrop for the hi-fi render: what the camera sees beyond the field splat.

WHY. The splat renderer composites the field and the bodies over a flat colour. The follow camera, 26 m back and
10 m up, looks past the far sideline, and the top fifth of every frame was a black void (play 1, v117a): not sky --
the camera looks down, its horizon is above the frame -- but ground beyond the field texture's edge.

WHAT. Every pixel's ray from the camera is intersected with the ground (world z = 0, z up). Where it lands inside
the field (the splat covers it) or on the apron around it, turf; beyond, the stands: a dark bank with a crowd whose
pattern is fixed in world coordinates (cells of CROWD_CELL_M), so it does not flicker as the camera moves, in the home
crowd's colours. Rays that do not reach the ground (above the horizon) get a night sky that glows toward the horizon.
Pass the image to compositing.splat_torch.render as ``background``.
"""
from __future__ import annotations

import numpy as np

FIELD_HALF_X_M = 54.86      # end line to end line, 120 yd
FIELD_HALF_Y_M = 24.38      # sideline to sideline, 53 1/3 yd
APRON_M = 7.0               # turf beyond the lines before the stands (benches, the sideline area)
WALL_M = 1.2                # the dark wall at the front of the stands
TURF = (0.13, 0.24, 0.12)
WALL = (0.03, 0.03, 0.05)
STANDS = (0.06, 0.055, 0.07)
CROWD = ((0.45, 0.06, 0.10), (0.62, 0.10, 0.14), (0.80, 0.78, 0.74), (0.20, 0.18, 0.20), (0.10, 0.10, 0.12))
CROWD_WEIGHTS = (0.30, 0.18, 0.10, 0.22, 0.20)
CROWD_CELL_M = 0.55
CROWD_FADE_M = 60.0         # the crowd fades into the dark with distance from the field
CROWD_MIX = 0.42            # how much of a crowd cell's colour shows over the stands' dark (a night game)
SOFTEN_PX = 1.4             # a Gaussian over the backdrop (pixels)
SKY_TOP = (0.015, 0.020, 0.045)
SKY_RIM = (0.13, 0.15, 0.22)
RIM_PX_FRAC = 0.15           # the sky's glow above the horizon, as a share of the image height


def horizon_rows(K, R, width: int) -> tuple:
    """``(row at column 0, row at column width-1)`` where a pixel's ray is parallel to the ground (world z = 0).
    ``R`` maps world to camera (x_cam = R x_world + t), ``K`` the 3x3 intrinsics."""
    A = np.asarray(R, float).T @ np.linalg.inv(np.asarray(K, float))     # camera pixel -> world ray direction
    a, b, c = A[2]                                                       # world z of the ray for pixel (u, v, 1)
    if abs(b) < 1e-12:
        return (-1e9, -1e9)
    return (float(-(a * 0 + c) / b), float(-(a * (width - 1) + c) / b))


def _crowd(gx, gy, seed: int):
    """Colours of the crowd cells at ground points (gx, gy): a hashed palette pick per CROWD_CELL_M cell."""
    ix = np.floor(gx / CROWD_CELL_M).astype(np.int64)
    iy = np.floor(gy / CROWD_CELL_M).astype(np.int64)
    h = (ix * 73856093) ^ (iy * 19349663) ^ (seed * 83492791)
    u = (h % 100003).astype(float) / 100003.0
    edges = np.cumsum(CROWD_WEIGHTS) / np.sum(CROWD_WEIGHTS)
    pick = np.searchsorted(edges, u)
    pal = np.asarray(CROWD, float)
    return pal[np.clip(pick, 0, len(pal) - 1)]


def backdrop(K, R, t, width: int, height: int, *, seed: int = 7) -> np.ndarray:
    """``[height, width, 3]`` float RGB in [0, 1]: turf, the stands and their crowd where each pixel's ray meets the
    ground, a night sky where it does not."""
    K = np.asarray(K, float); R = np.asarray(R, float); t = np.asarray(t, float).reshape(3)
    C = -R.T @ t                                                          # camera centre in world
    u, v = np.meshgrid(np.arange(width, dtype=float) + 0.5, np.arange(height, dtype=float) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], -1)                           # [h, w, 3]
    rays = pix @ (R.T @ np.linalg.inv(K)).T                               # world ray directions
    rz = rays[..., 2]
    down = rz < -1e-6
    s = np.where(down, -C[2] / np.where(down, rz, -1.0), np.inf)
    gx = C[0] + s * rays[..., 0]
    gy = C[1] + s * rays[..., 1]
    img = np.empty((height, width, 3), float)
    # sky where the ray never meets the ground: dark above, glowing toward the horizon
    left, right = horizon_rows(K, R, width)
    hz = left + (right - left) * (u[0] - 0.5) / max(1, width - 1)
    d = v - hz[None, :]                                                   # < 0 above the horizon
    rim = max(1.0, RIM_PX_FRAC * height)
    g = np.clip(1.0 + d / rim, 0.0, 1.0)[..., None]
    img[:] = np.asarray(SKY_TOP)[None, None, :] * (1 - g) + np.asarray(SKY_RIM)[None, None, :] * g
    # the ground: beyond the lines by the apron is turf, then a wall, then the stands with the crowd
    ex = np.maximum(np.abs(gx) - FIELD_HALF_X_M, 0.0)
    ey = np.maximum(np.abs(gy) - FIELD_HALF_Y_M, 0.0)
    out = np.hypot(ex, ey)                                                # metres beyond the field's edge
    turf = down & (out <= APRON_M)
    wall = down & (out > APRON_M) & (out <= APRON_M + WALL_M)
    stands = down & (out > APRON_M + WALL_M)
    img[turf] = TURF
    img[wall] = WALL
    if stands.any():
        crowd = _crowd(gx[stands], gy[stands], seed)
        fade = np.clip(1.0 - (out[stands] - APRON_M - WALL_M) / CROWD_FADE_M, 0.15, 1.0)[:, None]
        img[stands] = np.asarray(STANDS)[None, :] * (1 - CROWD_MIX * fade) + crowd * CROWD_MIX * fade
    if SOFTEN_PX > 0:                                   # the crowd's cells read as blocks when sharp; the players are not in this image
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(img, sigma=(SOFTEN_PX, SOFTEN_PX, 0))
    return np.clip(img, 0.0, 1.0).astype(np.float32)
