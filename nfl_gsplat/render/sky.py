"""A night-stadium backdrop for the hi-fi render: what the camera sees above and beyond the field.

WHY. The splat renderer composites the field and the bodies over a flat colour; a follow camera 26 m back and 10 m
up looks past the far sideline, and the top fifth of every frame was a black void (play 1, v117a). The game was a
night kickoff: a dark sky over a lit rim of stands reads as the stadium and costs nothing per frame.

WHAT. ``horizon_rows`` gives the image rows of the camera's horizon (the ground plane's vanishing line, world z up)
at the left and right edges; ``backdrop`` paints, per pixel by its signed distance to that line, a sky gradient
(dark at the top, a cool glow toward the rim), a band of stands with sparse warm lights just under the rim, and
the stands' dark tone below it (the field covers that, except beyond the texture's extent). Pass it to
compositing.splat_torch.render as the ``background`` image.
"""
from __future__ import annotations

import numpy as np

SKY_TOP = (0.015, 0.020, 0.045)
SKY_RIM = (0.13, 0.15, 0.22)
STANDS = (0.055, 0.055, 0.070)
LIGHTS = (0.85, 0.78, 0.60)
RIM_PX_FRAC = 0.15           # the glow's height above the horizon, as a share of the image height
STANDS_PX_FRAC = 0.05        # the stands band under the horizon


def horizon_rows(K, R, width: int) -> tuple:
    """``(row at column 0, row at column width-1)`` where a pixel's ray is parallel to the ground (world z = 0).
    ``R`` maps world to camera (x_cam = R x_world + t), ``K`` the 3x3 intrinsics."""
    A = np.asarray(R, float).T @ np.linalg.inv(np.asarray(K, float))     # camera pixel -> world ray direction
    a, b, c = A[2]                                                       # world z of the ray for pixel (u, v, 1)
    if abs(b) < 1e-12:
        return (-1e9, -1e9)
    return (float(-(a * 0 + c) / b), float(-(a * (width - 1) + c) / b))


def backdrop(K, R, width: int, height: int, *, seed: int = 7) -> np.ndarray:
    """``[height, width, 3]`` float RGB in [0, 1]: sky above the horizon line, the stands at and under it."""
    left, right = horizon_rows(K, R, width)
    cols = np.arange(width, dtype=float)
    hz = left + (right - left) * cols / max(1, width - 1)                 # horizon row per column
    rows = np.arange(height, dtype=float)[:, None]
    d = rows - hz[None, :]                                                # < 0 above the horizon
    rim = max(1.0, RIM_PX_FRAC * height)
    band = max(1.0, STANDS_PX_FRAC * height)
    top = np.asarray(SKY_TOP, float); glow = np.asarray(SKY_RIM, float); stands = np.asarray(STANDS, float)
    # sky: the top colour far above the rim, easing into the glow over the last ``rim`` pixels above the horizon
    s = np.clip(1.0 + d / rim, 0.0, 1.0)                                  # 0 far above, 1 at the horizon
    far = np.clip(-d / max(1.0, -d.min() if d.min() < 0 else 1.0), 0.0, 1.0)
    sky = top[None, None, :] * (0.7 + 0.3 * (1 - far[..., None])) * (1 - s[..., None]) + glow[None, None, :] * s[..., None]
    img = np.where((d < 0)[..., None], sky, stands[None, None, :])
    # the stands band: sparse warm lights (seeded, fixed per pixel so they do not flicker between frames)
    rng = np.random.default_rng(seed)
    lights = rng.random((height, width)) > 0.992
    in_band = (d >= 0) & (d < band)
    img[in_band & lights] = np.asarray(LIGHTS, float) * 0.6
    return np.clip(img, 0.0, 1.0).astype(np.float32)
