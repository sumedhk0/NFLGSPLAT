"""Does the calibrated yard-line grid lie on the painted lines? In pixels.

WHY. A sideline camera can read both paint rulers at 1.00 and put players at
1.82 m and still be wrong: play 2 (2026-09-04) chose a camera whose
projected 5-yard lines ran skewed across the painted ones and labelled them
the wrong way round. The rulers test row positions along the lines the
camera believes in; the heights test the lens; neither asks whether the
grid is ON the paint. This does, directly: project every 5-yard line, detect
the white near-vertical segments, and report the median distance from a
detected segment's midpoint to the nearest projected line.

A right camera reads a few pixels (line width, detection jitter); the play-2
camera reads tens.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_landmarks import (GOAL_LINE_X_M, HALF_WIDTH_M,
                                                    YARD_LINE_SPACING_M)

# A candidate whose grid sits farther than this from the paint (median over
# detected segments, 1080p) is not on the field. Measured: play 1's right
# camera 10.1 px (line width, detector jitter, interpolated per-frame
# cameras); play 2's skewed one 148.9 px.
MAX_GRID_PX_1080: float = 25.0


def projected_lines(K, R, t, *, half_width_m: float = HALF_WIDTH_M):
    """Homogeneous 2-D lines ``[L, 3]`` (normalised so ``|(a, b)| = 1``) of
    every 5-yard line between the goal lines, plus the goal lines."""
    P = np.asarray(K, float) @ np.column_stack([np.asarray(R, float)[:, :2],
                                                np.asarray(t, float).reshape(3)])
    n = int(round(2 * GOAL_LINE_X_M / YARD_LINE_SPACING_M))
    xs = -GOAL_LINE_X_M + YARD_LINE_SPACING_M * np.arange(0, n + 1)
    out = []
    for x in xs:
        a = P @ np.array([x, -half_width_m, 1.0])
        b = P @ np.array([x, half_width_m, 1.0])
        if a[2] <= 1e-9 or b[2] <= 1e-9:
            continue
        pa, pb = a[:2] / a[2], b[:2] / b[2]
        line = np.cross([pa[0], pa[1], 1.0], [pb[0], pb[1], 1.0])
        norm = np.hypot(line[0], line[1])
        if norm < 1e-9:
            continue
        out.append(line / norm)
    return np.asarray(out).reshape(-1, 3)


def segment_distances_px(segments, lines):
    """Per segment, the mean distance of its two ENDPOINTS to the nearest
    projected line (nearest by the midpoint). Endpoints, not the midpoint: a
    grid rotated about the image centre leaves midpoints near the lines and
    swings the endpoints by tens of pixels -- the skew is the failure mode."""
    if len(segments) == 0 or len(lines) == 0:
        return np.zeros(0)
    p0 = np.asarray([[s.p0[0], s.p0[1], 1.0] for s in segments])
    p1 = np.asarray([[s.p1[0], s.p1[1], 1.0] for s in segments])
    mids = 0.5 * (p0 + p1)
    mids[:, 2] = 1.0
    nearest = np.argmin(np.abs(mids @ lines.T), axis=1)           # [S]
    L = lines[nearest]                                            # [S, 3]
    return 0.5 * (np.abs(np.einsum("ij,ij->i", p0, L)) + np.abs(np.einsum("ij,ij->i", p1, L)))


def grid_distance_px(image_bgr, K, R, t, *, cfg=None, player_boxes=None):
    """``(median_px, n_segments)`` for one frame. NaN when no segment is found."""
    from nfl_gsplat.calibration.field_detect import FieldDetectConfig, detect_lines

    cfg = cfg or FieldDetectConfig()
    segs = detect_lines(np.asarray(image_bgr), cfg, player_boxes)
    d = segment_distances_px(segs, projected_lines(K, R, t))
    if len(d) == 0:
        return float("nan"), 0
    return float(np.median(d)), int(len(d))


def grid_scores(video_path, cams, frames, *, cfg=None):
    """Median grid distance over ``frames`` through per-frame ``cams``
    (nearest solved frame), and the total segment count."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ds, n_total = [], 0
    for f in frames:
        key = min(cams, key=lambda k: abs(k - f))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(key))
        ok, img = cap.read()
        if not ok:
            continue
        K, R, t = cams[key]
        d, n = grid_distance_px(img, K, R, t, cfg=cfg)
        if np.isfinite(d):
            ds.append(d)
            n_total += n
    cap.release()
    return (float(np.median(ds)) if ds else float("nan")), n_total
