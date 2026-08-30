"""The venue's own turf, art and lighting, sampled from footage onto the spec field.

``procedural_field`` draws what every NFL field shares -- geometry and markings,
exact from the rule book. It names what it cannot draw, because those things
differ per venue and per game: endzone artwork and the midfield logo, turf
colour and mowing pattern and wear, and lighting. Those can only come from the
footage, and until the cameras were trustworthy there was no way to put them
where they belong.

HOW. With a camera per frame, every pixel of turf can be sent to the field
position it came from: intersect its ray with the ground plane and drop the
colour into a top-down texture. Do that for many frames and the field fills in.

WHY A MEDIAN, not a mean or the latest write. Players, officials and the ball
cover parts of the turf in any given frame, and they MOVE, so a given patch of
grass is occluded in a minority of the frames that see it. The median over
frames rejects them for free, whereas a mean smears every player into a green
ghost and a last-write-wins keeps whichever frame happened to be last.

COVERAGE IS THE LIMIT, and it is why this accumulates over many clips. One play
sees roughly a third of the field -- measured at 31.8% on play_001 -- because
the cameras follow the ball. Different plays of the same game start at
different yard lines and look at different thirds, so the union over a game's
clips covers far more than any one of them. Where nothing was ever seen, the
procedural turf shows through rather than a hole, and ``coverage`` reports how
much of the result is real observation.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.field.procedural_field import texture_extent
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# A texel is only believed once this many frames have seen it. One or two
# samples are as likely to be a player's jersey as they are to be turf, and the
# median cannot reject an outlier it has no majority against.
MIN_SAMPLES: int = 3


def ground_grid(res_m: float, extent):
    """``(X, Y)`` world coordinates of every texel centre, as a map image."""
    x_min, x_max, y_min, y_max = extent
    width = int(round((x_max - x_min) / res_m))
    height = int(round((y_max - y_min) / res_m))
    cols = np.arange(width) + 0.5
    rows = np.arange(height) + 0.5
    return np.meshgrid(x_min + cols * res_m, y_max - rows * res_m)


# cv2.remap refuses a map with either dimension >= SHRT_MAX, and a field
# texture has far more texels than that -- at 0.05 m it is over two million.
# Sample coordinates are therefore reshaped into a rectangle rather than a
# single long column.
_REMAP_COLS: int = 4096


def remap_points(image, u, v, interpolation=None):
    """Bilinear sample of ``image`` at scattered ``(u, v)``, any number of them."""
    import cv2

    interpolation = cv2.INTER_LINEAR if interpolation is None else interpolation
    u = np.asarray(u, np.float32).ravel()
    v = np.asarray(v, np.float32).ravel()
    n = u.size
    if n == 0:
        return np.zeros((0, image.shape[2]), image.dtype)
    cols = min(n, _REMAP_COLS)
    rows = int(np.ceil(n / cols))
    pad = rows * cols - n
    map_x = np.concatenate([u, np.zeros(pad, np.float32)]).reshape(rows, cols)
    map_y = np.concatenate([v, np.zeros(pad, np.float32)]).reshape(rows, cols)
    out = cv2.remap(image, map_x, map_y, interpolation)
    return out.reshape(-1, image.shape[2])[:n]


def sample_frame(image, K, R, t, *, res_m: float, extent, z_plane: float = 0.0):
    """``(colour, seen)`` -- the frame resampled onto the top-down field grid.

    Every texel is PROJECTED into the image rather than every pixel being
    unprojected, so the output grid is filled exactly once with no gaps or
    collisions, and texels the camera cannot see are simply marked unseen.
    """
    mesh_x, mesh_y = ground_grid(res_m, extent)
    shape = mesh_x.shape
    world = np.stack([mesh_x.ravel(), mesh_y.ravel(),
                      np.full(mesh_x.size, z_plane)], axis=1)

    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    cam = world @ R.T + t
    in_front = cam[:, 2] > 1e-6
    uvw = cam @ np.asarray(K, float).T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]

    h, w = image.shape[:2]
    inside = (in_front & np.isfinite(uv).all(axis=1)
              & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))

    colour = np.zeros((*shape, image.shape[2]), np.float32)
    seen = inside.reshape(shape)
    if inside.any():
        sampled = remap_points(image, uv[inside, 0], uv[inside, 1])
        flat = colour.reshape(-1, image.shape[2])
        flat[inside] = sampled
        colour = flat.reshape(*shape, image.shape[2])
    return colour, seen


def accumulate(frames, *, res_m: float = 0.05, extent=None,
               z_plane: float = 0.0, min_samples: int = MIN_SAMPLES):
    """``(texture, coverage_mask, n_samples)`` from ``(image, K, R, t)`` frames.

    Held in memory as a stack because the median needs every sample; at 0.05 m
    the field is about 2300x1000 texels, so keep the frame count sane or raise
    ``res_m``.
    """
    extent = texture_extent() if extent is None else extent
    stack, masks = [], []
    for image, K, R, t in frames:
        colour, seen = sample_frame(image, K, R, t, res_m=res_m,
                                    extent=extent, z_plane=z_plane)
        stack.append(colour)
        masks.append(seen)
    if not stack:
        raise ValueError("no frames given")

    values = np.stack(stack)                       # [N, H, W, C]
    seen = np.stack(masks)                         # [N, H, W]
    counts = seen.sum(axis=0)
    values[~seen] = np.nan
    good = counts >= min_samples
    # Take the median only where something was seen. Running it everywhere is
    # correct but noisy: every never-observed texel is an all-NaN slice, and
    # numpy warns once per call about a case that is expected and handled.
    texture = np.zeros(values.shape[1:], np.float64)
    if good.any():
        texture[good] = np.nanmedian(values[:, good], axis=0)
    _LOG.info("field appearance: %d frames, %.1f%% of the texture observed "
              "(>=%d samples)", len(stack), 100.0 * good.mean(), min_samples)
    return np.nan_to_num(texture).astype(np.uint8), good, counts


def composite(observed, coverage, procedural):
    """Observed turf where it was seen, the drawn field everywhere else.

    The markings are NOT redrawn over the observation. Paint that was actually
    photographed is already in the right place and carries the venue's real
    wear and lighting; drawing over it would replace a measurement with an
    idealisation and put a hard seam at the coverage boundary.
    """
    observed = np.asarray(observed)
    procedural = np.asarray(procedural)
    if observed.shape != procedural.shape:
        raise ValueError(
            f"observed {observed.shape} and procedural {procedural.shape} "
            "must match -- render both at the same res_m and margin.")
    out = procedural.copy()
    out[coverage] = observed[coverage]
    return out


def coverage_fraction(coverage) -> float:
    return float(np.asarray(coverage, bool).mean())
