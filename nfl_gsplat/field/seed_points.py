"""Seed point cloud for splatfacto, on the real field plane in metres.

Why this exists
---------------
splatfacto needs initial 3D points. Normally they come from COLMAP's sparse
reconstruction, which this pipeline has no use for otherwise: both feeds are
tripods, so there are only TWO camera centres in the entire play and
structure-from-motion has almost no parallax to work with.

Without seed points splatfacto random-initialises ``num_random`` gaussians
inside a cube of half-width ``random_scale`` (defaults 50000 and 10.0) centred on
the origin. Those defaults assume a scene normalised to roughly unit size. This
pipeline deliberately keeps poses METRIC and un-recentred (``--auto-scale-poses
False --center-method none``) so that world coordinates stay in field metres, and
a +-10 m cube is then a small box floating near midfield while the cameras sit
100+ m out. Measured on play_001: of 50000 random gaussians, 38148 were culled as
below-alpha at the FIRST densification step, gradient densification never
recovered ("Duplicating 0.0" for the whole run), and 30k iterations produced
11580 gaussians and a grey slab instead of a field.

The field's geometry is known exactly, so seeding it directly is both easier and
better than any amount of tuning: points go where the surface actually is.

Coordinate frame matches the calibration: X along the field's length, Y across
it, Z up, origin at the centre of the field, playing surface at Z = 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.field_landmarks import (GOAL_LINE_X_M,
                                                    HALF_LENGTH_M,
                                                    HALF_WIDTH_M,
                                                    HASH_OFFSET_M,
                                                    YARD_LINE_SPACING_M)
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Turf and paint, as plain 8-bit colour. These are only an INITIALISATION --
# splatfacto optimises spherical harmonics from the images within a few hundred
# iterations -- but starting near the right colour means the first culling pass
# does not discard points for being obviously wrong.
_TURF_RGB = (58, 122, 62)
_PAINT_RGB = (232, 232, 232)

_MARGIN_M = 6.0        # sideline apron kept in view by both cameras


def field_seed_points(spacing_m: float = 0.25, *, jitter_m: float = 0.05,
                      seed: int = 0):
    """``(xyz, rgb)`` covering the playing surface at ``spacing_m`` resolution.

    Points are laid on Z = 0 on a jittered grid. The jitter matters: a perfectly
    regular lattice gives every gaussian an identical neighbourhood, and the
    split/duplicate heuristics then fire on all of them at once.

    Yard lines, hash rows and the goal lines are re-coloured as paint so the
    initialisation already carries the high-frequency structure the cameras key
    on -- the field is otherwise a near-uniform green where a splat has little
    to latch onto.
    """
    if spacing_m <= 0:
        raise ValueError(f"spacing_m must be positive, got {spacing_m}")

    half_len = HALF_LENGTH_M + _MARGIN_M
    half_wid = HALF_WIDTH_M + _MARGIN_M
    xs = np.arange(-half_len, half_len + spacing_m, spacing_m)
    ys = np.arange(-half_wid, half_wid + spacing_m, spacing_m)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    xyz = np.column_stack([grid_x.ravel(), grid_y.ravel(),
                           np.zeros(grid_x.size)])

    rng = np.random.default_rng(seed)
    xyz[:, :2] += rng.uniform(-jitter_m, jitter_m, size=(len(xyz), 2))

    rgb = np.tile(np.array(_TURF_RGB, np.uint8), (len(xyz), 1))

    # Yard lines every 4.572 m between the goal lines, plus the goal lines.
    n_lines = int(round(2 * GOAL_LINE_X_M / YARD_LINE_SPACING_M))
    line_x = -GOAL_LINE_X_M + YARD_LINE_SPACING_M * np.arange(n_lines + 1)
    on_yard = (np.abs(xyz[:, 0][:, None] - line_x[None, :]) < 0.06).any(axis=1)
    in_field = np.abs(xyz[:, 1]) <= HALF_WIDTH_M
    rgb[on_yard & in_field] = _PAINT_RGB

    # Hash rows: short marks, not continuous lines, at +-HASH_OFFSET_M.
    on_hash_row = (np.abs(np.abs(xyz[:, 1]) - HASH_OFFSET_M) < 0.06)
    on_mark = (np.abs(xyz[:, 0]) % 0.9144) < 0.15
    rgb[on_hash_row & on_mark & (np.abs(xyz[:, 0]) < GOAL_LINE_X_M)] = _PAINT_RGB

    # Sidelines and end lines.
    on_sideline = np.abs(np.abs(xyz[:, 1]) - HALF_WIDTH_M) < 0.08
    on_endline = np.abs(np.abs(xyz[:, 0]) - HALF_LENGTH_M) < 0.08
    rgb[on_sideline | on_endline] = _PAINT_RGB

    return xyz, rgb


def write_seed_ply(path: Path | str, spacing_m: float = 0.25, **kwargs) -> Path:
    """Write the field seed cloud as a binary PLY nerfstudio can read.

    nerfstudio's Nerfstudio dataparser loads this when ``transforms.json``
    carries a ``ply_file_path`` key and ``load_3D_points`` is on (splatfacto
    sets it). It expects x/y/z plus uint8 red/green/blue.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz, rgb = field_seed_points(spacing_m, **kwargs)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    payload = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"),
                                        ("z", "<f4"), ("red", "u1"),
                                        ("green", "u1"), ("blue", "u1")])
    payload["x"], payload["y"], payload["z"] = xyz.T.astype(np.float32)
    payload["red"], payload["green"], payload["blue"] = rgb.T

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(payload.tobytes())

    painted = int((rgb != np.array(_TURF_RGB, np.uint8)).any(axis=1).sum())
    _LOG.info("field seed cloud: %d points at %.2f m spacing (%d painted) -> %s",
              len(xyz), spacing_m, painted, path)
    return path
