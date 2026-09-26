"""A night-stadium backdrop for the hi-fi render: what the camera sees beyond the field splat.

WHY. The splat renderer composites the field and the bodies over a flat colour. The follow camera, 26 m back and
10 m up, looks past the far sideline, and the top fifth of every frame was a black void (play 1, v117a): not sky --
the camera looks down, its horizon is above the frame -- but ground beyond the field texture's edge.

WHAT. Every pixel's ray from the camera is intersected with the ground (world z = 0, z up). Where it lands inside
the field (the splat covers it) or on the apron around it, turf; beyond, the stands: a dark bank with a crowd whose
pattern is fixed in world coordinates (cells of CROWD_CELL_M), so it does not flicker as the camera moves, in the home
crowd's colours. Rays that do not reach the ground (above the horizon) get a night sky that glows toward the horizon.
Pass the image to compositing.splat_torch.render as ``background``.

STANDS_MODE "bowl" (the default since v120): flat stands on the ground read as a pixel mosaic squeezed into a band (v119's
crowd); a real bowl RISES. As in the viewer (viewer/play_room.html): a BOWL_WALL_H_M wall BOWL_APRON_M beyond the
lines with a dim red ribbon board along its top, then a bank of seats rising BOWL_RISE_M over BOWL_DEPTH_M, one planar
bank per side (the side banks run long enough to close the corners), each intersected analytically per pixel ray.
The bank is laid out in seats (SEAT_M) and rows (ROW_M up the slope), rows staggered, an aisle every AISLE_EVERY
seats; a seat holds a spectator -- a shirt in the home crowd's colours and a head -- over the row's dark step, or is
empty. The crowd dims up the bank. STANDS_MODE "flat" draws v119's stands exactly.
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
STANDS_MODE = "bowl"        # "bowl" (rising stands, the viewer's) or "flat" (v118-v119: the stands on the ground)
BOWL_APRON_M = 8.0
BOWL_WALL_H_M = 1.3
BOWL_RISE_M = 26.0
BOWL_DEPTH_M = 42.0
SEAT_M = 0.55
ROW_M = 0.9
AISLE_EVERY = 18            # seats between aisles
RISER_FRAC = 0.28           # the dark step at the foot of each row
SHIRTS = ((0.52, 0.07, 0.11), (0.70, 0.11, 0.15), (0.82, 0.80, 0.76), (0.24, 0.21, 0.23), (0.12, 0.11, 0.13),
          (0.60, 0.46, 0.34))
SHIRT_WEIGHTS = (0.30, 0.17, 0.10, 0.18, 0.15, 0.10)
HEADS = ((0.55, 0.40, 0.30), (0.36, 0.24, 0.17), (0.20, 0.14, 0.11), (0.62, 0.10, 0.14), (0.85, 0.83, 0.80))
SEAT_BACK = (0.10, 0.03, 0.05)   # the red seats behind and between the spectators
STEP = (0.035, 0.033, 0.04)
OCCUPIED = 0.93
BOWL_WALL = (0.03, 0.03, 0.045)
RIBBON = (0.55, 0.06, 0.09)      # a dim red ribbon board along the top of the wall
RIBBON_H_M = 0.45
CROWD_GAIN = 0.75                # the stands under the lights, dimmer than the field
LIGHT_TOP = 0.55                 # the crowd's brightness at the top row (1.0 at the front)
BOWL_SOFTEN_PX = 1.1
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


def _hash(ix, iy, seed: int):
    h = (np.asarray(ix).astype(np.int64) * 73856093) ^ (np.asarray(iy).astype(np.int64) * 19349663) ^ (seed * 83492791)
    return (np.abs(h) % 100003).astype(float) / 100003.0


def _spectators(a, d, seed: int):
    """RGB of the bank at along-coordinate ``a`` and up-slope coordinate ``d`` (metres): per seat a spectator (a shirt
    over the row's step and a head above it) or an empty seat; rows staggered; an aisle every AISLE_EVERY seats."""
    a = np.asarray(a, float); d = np.asarray(d, float)
    iy = np.floor(d / ROW_M)
    a = a + SEAT_M * _hash(iy, iy * 0 + 17, seed)
    ix = np.floor(a / SEAT_M)
    fa = a / SEAT_M - ix; fd = d / ROW_M - iy
    u = _hash(ix, iy, seed); occ = _hash(ix + 7919, iy + 104729, seed) < OCCUPIED
    edges = np.cumsum(SHIRT_WEIGHTS) / np.sum(SHIRT_WEIGHTS)
    shirts = np.asarray(SHIRTS, float); heads = np.asarray(HEADS, float)
    shirt = shirts[np.clip(np.searchsorted(edges, u), 0, len(shirts) - 1)]
    head = heads[np.clip((_hash(ix + 31, iy + 57, seed) * len(heads)).astype(int), 0, len(heads) - 1)]
    rgb = np.broadcast_to(np.asarray(SEAT_BACK, float), shirt.shape).copy()
    body = occ & (fd >= RISER_FRAC) & (fd < 0.74) & (np.abs(fa - 0.5) < 0.40)
    top = occ & (fd >= 0.74) & (fd < 0.94) & (np.abs(fa - 0.5) < 0.17)
    rgb[body] = shirt[body]
    rgb[top] = head[top]
    rgb[(fd < RISER_FRAC) | (np.mod(ix, AISLE_EVERY) == 0)] = STEP
    slope = np.hypot(BOWL_DEPTH_M, BOWL_RISE_M)
    light = CROWD_GAIN * (1.0 - (1.0 - LIGHT_TOP) * np.clip(d / slope, 0.0, 1.0))
    return rgb * light[:, None]


@np.errstate(divide="ignore", invalid="ignore")
def bowl_hits(C, rays):
    """Per ray (``[N, 3]`` world directions from the camera centre ``C``): ``(kind, a, d, z)`` -- kind 0 nothing (sky),
    1 the ground inside the walls, 2 a wall (``z`` its height), 3 a bank (``a`` along it, ``d`` up its slope)."""
    C = np.asarray(C, float); rays = np.asarray(rays, float)
    n = len(rays)
    best = np.full(n, np.inf)
    kind = np.zeros(n, np.int8)
    A = np.zeros(n); D = np.zeros(n); Z = np.zeros(n)
    rz = rays[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(rz < -1e-9, -C[2] / rz, np.inf)
    gx = C[0] + s * rays[:, 0]; gy = C[1] + s * rays[:, 1]
    inner = (FIELD_HALF_X_M + BOWL_APRON_M, FIELD_HALF_Y_M + BOWL_APRON_M)
    ok = np.isfinite(s) & (np.abs(gx) <= inner[0]) & (np.abs(gy) <= inner[1])
    best[ok] = s[ok]; kind[ok] = 1
    k = BOWL_RISE_M / BOWL_DEPTH_M
    for axis in (1, 0):                       # 1: the banks along the sidelines, 0: behind the end zones
        other = 1 - axis
        span = inner[other] + BOWL_DEPTH_M     # the side banks run on past the corners, closing them
        for sign in (1.0, -1.0):
            ra = rays[:, axis] * sign; ca = C[axis] * sign
            with np.errstate(divide="ignore", invalid="ignore"):
                sw = np.where(np.abs(ra) > 1e-9, (inner[axis] - ca) / ra, np.inf)
            zw = C[2] + sw * rays[:, 2]; ow = C[other] + sw * rays[:, other]
            hit = (sw > 0) & (sw < best) & (zw >= 0) & (zw <= BOWL_WALL_H_M) & (np.abs(ow) <= span)
            best[hit] = sw[hit]; kind[hit] = 2; Z[hit] = zw[hit]
            denom = rays[:, 2] - k * ra
            with np.errstate(divide="ignore", invalid="ignore"):
                sb = np.where(np.abs(denom) > 1e-9, (BOWL_WALL_H_M + k * (ca - inner[axis]) - C[2]) / denom, np.inf)
            pa = ca + sb * ra - inner[axis]; ob = C[other] + sb * rays[:, other]
            hit = (sb > 0) & (sb < best) & (pa >= 0) & (pa <= BOWL_DEPTH_M) & (np.abs(ob) <= span)
            best[hit] = sb[hit]; kind[hit] = 3
            A[hit] = ob[hit] * sign; D[hit] = pa[hit] * np.hypot(1.0, k)
    return kind, A, D, Z


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
    if STANDS_MODE == "bowl":
        kind, A, D, Z = bowl_hits(C, rays.reshape(-1, 3))
        flat = img.reshape(-1, 3)
        flat[kind == 1] = TURF
        w = kind == 2
        flat[w] = np.where((Z[w] > BOWL_WALL_H_M - RIBBON_H_M)[:, None], np.asarray(RIBBON)[None, :],
                           np.asarray(BOWL_WALL)[None, :])
        b = kind == 3
        if b.any():
            flat[b] = _spectators(A[b], D[b], seed)
        img = flat.reshape(height, width, 3)
        if BOWL_SOFTEN_PX > 0:
            from scipy.ndimage import gaussian_filter

            img = gaussian_filter(img, sigma=(BOWL_SOFTEN_PX, BOWL_SOFTEN_PX, 0))
        return np.clip(img, 0.0, 1.0).astype(np.float32)
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
