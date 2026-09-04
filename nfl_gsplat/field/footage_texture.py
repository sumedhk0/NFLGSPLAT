"""The field as the footage shows it, warped onto the ground plane.

WHY. The procedural field is flat green with white lines. The real one has
the night lighting gradient, painted end zones, logos, wear and the exact
paint, and the render reads as "that stadium" only with them. The
calibration already says where every image pixel meets the ground.

WHAT. For each frame, the ground plane (Z = 0) maps to the image by the
homography ``K [r1 r2 t]``; the frame is warped onto the field's texture
grid (the same extent and metres-per-pixel as procedural_field) and
texels are kept where the ray comes from in front of the camera, lands
inside the frame, and a texel covers at least ``min_px2`` square pixels
(the far field is foreshortened into nothing). The per-texel median over
the play removes the players, who move; where fewer than ``min_count``
frames saw a texel the procedural texture shows through, feathered.

Both cameras contribute; each texel's median is over every frame of every
camera that saw it. Calibration jitter blurs the paint by its reprojection
error (5-9 px after refinement), so the lines are softer than the
procedural ones; the turf is right.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from nfl_gsplat.field.procedural_field import texture_extent
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

MIN_PX2: float = 0.2          # square image pixels per texel; below this the ground is smeared
MIN_COUNT: int = 8            # frames that must have seen a texel for its median to count
FEATHER_PX: int = 16          # seam between footage and procedural, in texels (~2 m)
MAX_STACK_BYTES = 1_500_000_000


def ground_homography(K, R, t) -> np.ndarray:
    """``[3, 3]`` mapping ground-plane points ``(x, y, 1)`` to image ``(u, v, w)``."""
    R = np.asarray(R, np.float64)
    return np.asarray(K, np.float64) @ np.column_stack([R[:, 0], R[:, 1], np.asarray(t, np.float64)])


def texel_to_ground(res_m: float, extent) -> np.ndarray:
    """``[3, 3]`` mapping texture pixel ``(col, row, 1)`` to ground ``(x, y, 1)``,
    texel centres, in procedural_field's convention (x right, y up the image)."""
    x_min, _x_max, _y_min, y_max = extent
    return np.array([[res_m, 0.0, x_min + 0.5 * res_m],
                     [0.0, -res_m, y_max - 0.5 * res_m],
                     [0.0, 0.0, 1.0]])


def texture_shape(res_m: float, extent) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = extent
    return int(round((y_max - y_min) / res_m)), int(round((x_max - x_min) / res_m))


def warp_frame(image, K, R, t, *, res_m: float, extent=None, min_px2: float = MIN_PX2):
    """``(texture [H, W, 3] uint8, valid [H, W] bool)`` of one frame on the ground grid."""
    import cv2

    extent = texture_extent() if extent is None else extent
    h, w = texture_shape(res_m, extent)
    M = ground_homography(K, R, t) @ texel_to_ground(res_m, extent)     # texel -> image
    img = np.ascontiguousarray(np.asarray(image))
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    p = np.stack([cols, rows, np.ones_like(cols)], axis=-1) @ M.T           # [H, W, 3]
    wgt = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = p[..., 0] / wgt, p[..., 1] / wgt
        px2 = np.abs(np.linalg.det(M)) / np.abs(wgt) ** 3                # image area per texel
    ih, iw = img.shape[:2]
    valid = (wgt > 0) & (u >= 0) & (u <= iw - 1) & (v >= 0) & (v <= ih - 1) & (px2 >= min_px2)
    return warped, valid


def median_texture(frames, *, min_count: int = MIN_COUNT):
    """``(median [H, W, 3] uint8, count [H, W] int)`` over ``(texture, valid)`` pairs."""
    textures = np.stack([f[0] for f in frames]).astype(np.uint8)          # [N, H, W, 3]
    valid = np.stack([f[1] for f in frames])                              # [N, H, W]
    count = valid.sum(axis=0)
    n, h, w, _ = textures.shape
    out = np.zeros((h, w, 3), np.uint8)
    rows_per_chunk = max(1, int(MAX_STACK_BYTES // max(1, n * w * 3 * 4)))
    for r0 in range(0, h, rows_per_chunk):
        r1 = min(h, r0 + rows_per_chunk)
        block = textures[:, r0:r1].astype(np.float32)
        block[~valid[:, r0:r1]] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(block, axis=0)
        med[~np.isfinite(med)] = 0
        out[r0:r1] = np.clip(np.rint(med), 0, 255).astype(np.uint8)
    out[count < min_count] = 0
    return out, count


def footage_colours(footage, count, *, min_count: int = MIN_COUNT):
    """``(turf, paint)`` of the footage, uint8 triples in the texture's own
    channel order (BGR from footage_texture): the median
    of the seen texels is turf; the paint is the median of the brightest
    2 % (lines and numerals). The procedural remainder of a composite is
    drawn in these so the seam is a change of detail, not of colour."""
    seen = np.asarray(footage)[np.asarray(count) >= min_count].astype(np.float32)
    if len(seen) < 100:
        return None, None
    lum = seen.mean(axis=1)
    turf = np.median(seen[lum <= np.percentile(lum, 90)], axis=0)
    bright = seen[lum >= np.percentile(lum, 98)]
    paint = np.median(bright, axis=0) if len(bright) >= 20 else np.array([240.0, 240.0, 240.0])
    return tuple(int(round(v)) for v in turf), tuple(int(round(v)) for v in paint)


def composite(procedural, footage, count, *, min_count: int = MIN_COUNT,
              feather_px: int = FEATHER_PX) -> np.ndarray:
    """Footage where enough frames saw the ground, procedural elsewhere, feathered."""
    import cv2

    alpha = (np.asarray(count) >= min_count).astype(np.float32)
    if feather_px > 0:
        k = 2 * feather_px + 1
        eroded = cv2.erode(alpha, np.ones((k, k), np.uint8))
        alpha = cv2.GaussianBlur(eroded, (k, k), 0)
    a = alpha[..., None]
    out = a * np.asarray(footage, np.float32) + (1 - a) * np.asarray(procedural, np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def footage_texture(videos, tracks, *, res_m: float, stride: int = 4, extent=None,
                    min_px2: float = MIN_PX2, min_count: int = MIN_COUNT):
    """Median ground texture over the play from ``{cam: video_path}`` and
    ``{cam: CameraTrack}``. Returns ``(median, count)``; the texture is BGR
    like procedural_field's, so the two composite and render alike."""
    import cv2

    extent = texture_extent() if extent is None else extent
    frames = []
    for cam, video in videos.items():
        tr = tracks[cam]
        cap = cv2.VideoCapture(str(video))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(n_video, tr.num_frames)
        used = 0
        for f in range(0, n, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            if not ok:
                break
            if tr.conf[f] <= 0:
                continue
            intr, pose = tr.at(f)
            K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
            # BGR, as procedural_field's texture: texture_to_gaussians flips once.
            frames.append(warp_frame(bgr, K, pose.R, pose.t, res_m=res_m, extent=extent, min_px2=min_px2))
            used += 1
        cap.release()
        _LOG.info("footage texture: %s, %d frames warped (stride %d)", cam, used, stride)
    if not frames:
        from nfl_gsplat.errors import SetupError

        raise SetupError("no frames with a calibrated camera; check cameras.npz conf")
    return median_texture(frames, min_count=min_count)


def save_texture(path: Path | str, texture, count, *, res_m: float) -> Path:
    path = Path(path)
    np.savez_compressed(path, texture=np.asarray(texture, np.uint8), count=np.asarray(count, np.int32),
                        res_m=np.array(float(res_m)))
    return path


def load_texture(path: Path | str):
    """``(texture uint8 [H, W, 3], res_m)``."""
    z = np.load(Path(path))
    return z["texture"], float(z["res_m"])
