"""The NFL field drawn from its specification, not reconstructed from footage.

Why
---
Field GEOMETRY and MARKINGS are identical in every NFL stadium and already
encoded exactly in :mod:`nfl_gsplat.calibration.field_landmarks`. Recovering
them from broadcast video is strictly worse than drawing them: the footage is
zoomed, compressed, grazing at distance, occluded by players, and covers only
the part of the field the cameras happened to point at (measured on play_001:
31.8%). Drawn from spec, the markings are exact, sharp at any resolution,
complete, and free.

What this module does NOT cover, because it is venue- and game-specific and can
only come from footage:

  * endzone artwork and midfield logo
  * turf colour, mowing pattern, wear
  * lighting and shadows (Arizona's retractable roof throws a hard shadow line)

Those composite on top of this as later layers.

Output is a metric top-down texture, plus a conversion to a standard 3DGS PLY so
the field is the same primitive as everything else in the scene -- it loads with
:func:`nfl_gsplat.compositing.merge_ply.load_gaussian_ply` and merges with the
player avatars rather than needing a separate renderer.

Frame matches the calibration: X along the field's length, Y across it, Z up,
origin at the centre of the field, playing surface at Z = 0.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from nfl_gsplat.calibration.field_landmarks import (GOAL_LINE_X_M,
                                                    HALF_LENGTH_M,
                                                    HALF_WIDTH_M,
                                                    HASH_OFFSET_M,
                                                    NUMBER_BOTTOM_Y_M,
                                                    NUMBER_TOP_Y_M,
                                                    YARD_LINE_SPACING_M,
                                                    YARD_TO_M)
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# NFL rule book, in metres.
LINE_WIDTH_M: float = 4.0 * 0.0254          # 4 inch paint
HASH_MARK_LEN_M: float = 24.0 * 0.0254      # 2 ft
HASH_MARK_PITCH_M: float = YARD_TO_M        # one per yard
SIDELINE_BORDER_M: float = 6.0 * 0.3048     # 6 ft white border outside the field

_TURF_RGB = (58, 122, 62)
_PAINT_RGB = (240, 240, 240)

# SH degree-0 basis constant, the 3DGS convention for turning a colour into the
# DC spherical-harmonic coefficient.
_SH_C0 = 0.28209479177387814


def texture_extent(margin_m: float = SIDELINE_BORDER_M):
    """``(x_min, x_max, y_min, y_max)`` covered by the rendered texture."""
    return (-HALF_LENGTH_M - margin_m, HALF_LENGTH_M + margin_m,
            -HALF_WIDTH_M - margin_m, HALF_WIDTH_M + margin_m)


def _to_px(x_m, y_m, res_m, extent):
    """World metres -> (col, row). Row 0 is +Y, so the image reads like a map."""
    x_min, _x_max, _y_min, y_max = extent
    col = (np.asarray(x_m) - x_min) / res_m
    row = (y_max - np.asarray(y_m)) / res_m
    return col, row


def render_field_texture(res_m: float = 0.05, *, turf_rgb=_TURF_RGB,
                         paint_rgb=_PAINT_RGB, mow_stripes: bool = True,
                         numbers: bool = True,
                         margin_m: float = SIDELINE_BORDER_M) -> np.ndarray:
    """Top-down BGR texture of the regulation markings.

    ``res_m`` is metres per pixel. Returns ``uint8`` ``[H, W, 3]`` spanning
    :func:`texture_extent`.
    """
    if res_m <= 0:
        raise SetupError(f"res_m must be positive, got {res_m}")
    extent = texture_extent(margin_m)
    x_min, x_max, y_min, y_max = extent
    width = int(round((x_max - x_min) / res_m))
    height = int(round((y_max - y_min) / res_m))
    if width * height > 200_000_000:
        raise SetupError(
            f"res_m={res_m} would render {width}x{height} px. Pick a coarser "
            "resolution; the markings are vector-exact at any scale, so there "
            "is nothing to gain past the renderer's sampling rate.")

    img = np.zeros((height, width, 3), np.uint8)
    img[:] = turf_rgb[::-1]                      # cv2 is BGR

    if mow_stripes:
        # Mown bands run across the field every 5 yards. Real, but the pattern
        # and phase are groundskeeper's choice, so this is a plausible default
        # rather than a measurement -- layer 2 replaces it from footage.
        for i in range(-12, 12):
            x0 = i * YARD_LINE_SPACING_M
            if i % 2:
                continue
            c0, _ = _to_px(x0, 0, res_m, extent)
            c1, _ = _to_px(x0 + YARD_LINE_SPACING_M, 0, res_m, extent)
            band = img[:, int(c0):int(c1)].astype(np.int16) + 9
            img[:, int(c0):int(c1)] = np.clip(band, 0, 255).astype(np.uint8)

    paint = tuple(int(v) for v in paint_rgb[::-1])
    line_px = max(1, int(round(LINE_WIDTH_M / res_m)))

    def _line(p0, p1, thickness=line_px, colour=paint):
        c0, r0 = _to_px(p0[0], p0[1], res_m, extent)
        c1, r1 = _to_px(p1[0], p1[1], res_m, extent)
        cv2.line(img, (int(round(c0)), int(round(r0))),
                 (int(round(c1)), int(round(r1))), colour, thickness,
                 lineType=cv2.LINE_AA)

    # Yard lines every 5 yards, goal line to goal line, sideline to sideline.
    n_lines = int(round(2 * GOAL_LINE_X_M / YARD_LINE_SPACING_M))
    for i in range(n_lines + 1):
        x = -GOAL_LINE_X_M + i * YARD_LINE_SPACING_M
        _line((x, -HALF_WIDTH_M), (x, HALF_WIDTH_M))

    # Sidelines and end lines.
    _line((-HALF_LENGTH_M, -HALF_WIDTH_M), (HALF_LENGTH_M, -HALF_WIDTH_M))
    _line((-HALF_LENGTH_M, HALF_WIDTH_M), (HALF_LENGTH_M, HALF_WIDTH_M))
    _line((-HALF_LENGTH_M, -HALF_WIDTH_M), (-HALF_LENGTH_M, HALF_WIDTH_M))
    _line((HALF_LENGTH_M, -HALF_WIDTH_M), (HALF_LENGTH_M, HALF_WIDTH_M))

    # Hash marks: one per YARD, 2 ft long, on both hash rows. These are the
    # cross-field constraint the endzone calibration depends on, so they must
    # sit exactly at +-HASH_OFFSET_M.
    n_hash = int(GOAL_LINE_X_M / HASH_MARK_PITCH_M)
    for k in range(-n_hash, n_hash + 1):
        x = k * HASH_MARK_PITCH_M
        if abs(abs(x) - GOAL_LINE_X_M) < 1e-6:
            continue
        for sign in (-1.0, 1.0):
            y = sign * HASH_OFFSET_M
            _line((x, y - HASH_MARK_LEN_M / 2), (x, y + HASH_MARK_LEN_M / 2))

    # Yard-line numbers, 6 ft tall, tops toward the nearer sideline.
    if numbers:
        _draw_numbers(img, res_m, extent, paint)

    return img


def _draw_numbers(img, res_m, extent, colour):
    """Yard numerals at both number rows.

    Glyph shapes are approximated with a stroked OpenCV font; NFL numerals are a
    specific typeface. Position, height and orientation follow the rule book,
    which is what calibration and geometry care about -- swap in real glyphs
    later if photoreal close-ups need them.
    """
    height_m = NUMBER_TOP_Y_M - NUMBER_BOTTOM_Y_M          # 6 ft
    target_px = height_m / res_m
    font = cv2.FONT_HERSHEY_DUPLEX
    (_w, h), _b = cv2.getTextSize("50", font, 1.0, 2)
    scale = target_px / h
    thickness = max(1, int(round(0.12 * target_px)))

    for i in range(1, 20):
        x = -GOAL_LINE_X_M + i * YARD_LINE_SPACING_M
        yards = int(round((GOAL_LINE_X_M - abs(x)) / YARD_TO_M))
        if yards % 10 or yards == 0:
            continue
        label = f"{yards}"
        for sign in (-1.0, 1.0):
            y_centre = sign * 0.5 * (NUMBER_BOTTOM_Y_M + NUMBER_TOP_Y_M)
            canvas = np.zeros_like(img)
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            col, row = _to_px(x, y_centre, res_m, extent)
            org = (int(round(col - tw / 2)), int(round(row + th / 2)))
            cv2.putText(canvas, label, org, font, scale, colour, thickness,
                        cv2.LINE_AA)
            # Numerals face the nearer sideline: the row at -Y reads upright
            # from that side, so it is rotated 180 relative to the +Y row.
            if sign < 0:
                mat = cv2.getRotationMatrix2D((float(col), float(row)), 180.0, 1.0)
                canvas = cv2.warpAffine(canvas, mat, (img.shape[1], img.shape[0]))
            mask = canvas.any(axis=2)
            img[mask] = canvas[mask]


def texture_to_gaussians(texture: np.ndarray, res_m: float, *,
                         margin_m: float = SIDELINE_BORDER_M,
                         opacity: float = 0.99,
                         thickness_ratio: float = 0.05):
    """Flat, opaque, view-independent gaussians -- one per texel.

    A plane needs no volumetric model, so each texel becomes a disc lying IN the
    surface: two in-plane axes about a texel wide, the third almost zero.
    Spherical harmonics stop at degree 0 because turf and paint are effectively
    Lambertian; that also keeps the file small enough to merge with player
    avatars without special handling.

    Returns the arrays a 3DGS PLY needs, in the storage conventions the format
    uses: scale as log(sigma), opacity as logit(alpha), colour as the DC SH
    coefficient.
    """
    height, width = texture.shape[:2]
    extent = texture_extent(margin_m)
    x_min, _x_max, _y_min, y_max = extent

    cols = np.arange(width) + 0.5
    rows = np.arange(height) + 0.5
    grid_x = x_min + cols * res_m
    grid_y = y_max - rows * res_m
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    xyz = np.stack([mesh_x.ravel(), mesh_y.ravel(),
                    np.zeros(mesh_x.size)], axis=1).astype(np.float32)

    rgb = texture.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0   # BGR->RGB
    f_dc = ((rgb - 0.5) / _SH_C0).astype(np.float32)

    sigma_xy = 0.5 * res_m
    sigma_z = max(thickness_ratio * res_m, 1e-4)
    scale = np.tile(np.log([sigma_xy, sigma_xy, sigma_z]).astype(np.float32),
                    (len(xyz), 1))

    alpha = float(np.clip(opacity, 1e-4, 1 - 1e-4))
    opac = np.full(len(xyz), np.log(alpha / (1 - alpha)), np.float32)

    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (len(xyz), 1))
    return xyz, rot, scale, opac, f_dc


def write_field_gaussian_ply(path: Path | str, res_m: float = 0.05,
                             **render_kwargs) -> Path:
    """Render the markings and write them as a standard 3DGS PLY.

    Degree-0 SH, so the properties are x/y/z, rot_0..3, scale_0..2, opacity and
    f_dc_0..2 -- every value float32, which is what
    :func:`~nfl_gsplat.compositing.merge_ply.load_gaussian_ply` requires and
    what common splat viewers accept.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    margin = render_kwargs.get("margin_m", SIDELINE_BORDER_M)
    texture = render_field_texture(res_m, **render_kwargs)
    xyz, rot, scale, opac, f_dc = texture_to_gaussians(texture, res_m,
                                                       margin_m=margin)

    names = (["x", "y", "z"]
             + [f"rot_{i}" for i in range(4)]
             + [f"scale_{i}" for i in range(3)]
             + ["opacity"]
             + [f"f_dc_{i}" for i in range(3)])
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              + "".join(f"property float {n}\n" for n in names)
              + "end_header\n").encode("ascii")

    table = np.concatenate(
        [xyz, rot, scale, opac[:, None], f_dc], axis=1).astype(np.float32)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(table.tobytes())

    _LOG.info("procedural field: %d gaussians at %.3f m/px -> %s (%.1f MB)",
              len(xyz), res_m, path, path.stat().st_size / 1e6)
    return path
