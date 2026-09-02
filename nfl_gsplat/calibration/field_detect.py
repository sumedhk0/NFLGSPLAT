"""Detect field markings in a frame (cv2 lines/hashes).

`detect_lines` (white-line detection + orientation split) is validated on
synthetic field images. `detect_hashes` uses connected-component analysis to
find hash ticks, masking out player bounding boxes first.
`detect_field_features` is the top-level entry point for the calibration
pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nfl_gsplat.calibration.field_features import (
    DetectedFeatures, YardLineSeg,
)


@dataclass(frozen=True)
class FieldDetectConfig:
    white_thresh: int = 180
    min_line_len_frac: float = 0.25
    max_line_gap_px: int = 30
    vertical_deg: float = 35.0
    hash_min_area: int = 8
    hash_max_area: int = 400
    hash_max_h_px: int = 22


def _hough_rows(segs) -> np.ndarray:
    """(N, 4) endpoint rows from cv2.HoughLinesP, whatever the OpenCV major.

    OpenCV <=4 returns (N, 1, 4); OpenCV 5 dropped the middle axis and returns
    (N, 4). Reshaping covers both instead of hard-coding one layout."""
    return np.asarray(segs).reshape(-1, 4)


def _white_mask(img_bgr: np.ndarray, cfg: FieldDetectConfig) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, cfg.white_thresh, 255, cv2.THRESH_BINARY)
    return m


def _zero_boxes(mask: np.ndarray, player_boxes) -> np.ndarray:
    if not player_boxes:
        return mask
    out = mask.copy()
    for x1, y1, x2, y2 in player_boxes:
        out[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 0
    return out


def detect_lines(
    img_bgr: np.ndarray,
    cfg: FieldDetectConfig,
    player_boxes=None,
) -> list[YardLineSeg]:
    """Detect near-vertical painted yard-line segments via HoughLinesP."""
    # Measured against the SHORT side, not the height. The two are the same
    # for every landscape frame, so this changes nothing that ever worked --
    # but a frame turned a quarter turn (see orientation.py) has height 1920
    # where it had 1080, which raised the bar 78% while the lines it has to
    # pass got no longer, and the endzone solve found nothing at all.
    short_side = min(img_bgr.shape[0], img_bgr.shape[1])
    mask = _zero_boxes(_white_mask(img_bgr, cfg), player_boxes)
    min_len = int(cfg.min_line_len_frac * short_side)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=80,
                           minLineLength=min_len, maxLineGap=cfg.max_line_gap_px)
    out: list[YardLineSeg] = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in _hough_rows(segs):
        ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if ang >= (90 - cfg.vertical_deg):
            out.append(YardLineSeg((float(x1), float(y1)), (float(x2), float(y2))))
    return _merge_collinear(out)


def _merge_collinear(segs: list[YardLineSeg], x_tol: float = 18.0) -> list[YardLineSeg]:
    """Merge near-vertical segments belonging to the same painted line,
    PRESERVING slant (real broadcast yard lines lean well off vertical; the
    old vertical-collapse merge displaced them by tens of px mid-frame).

    Grouping: segments sorted/compared by their x at the global mean-y of all
    endpoints (a common reference row — per-segment mean-x mis-groups slanted
    lines detected at different heights). Each group is refit by least squares
    as x = a*y + b through all member endpoints, spanning the members' y-range.
    """
    if not segs:
        return []
    all_y = [p[1] for s in segs for p in (s.p0, s.p1)]
    ref_y = float(np.mean(all_y))

    def x_at(s: YardLineSeg, y: float) -> float:
        (x0, y0), (x1, y1) = s.p0, s.p1
        if abs(y1 - y0) < 1e-6:
            return 0.5 * (x0 + x1)
        return x0 + (y - y0) / (y1 - y0) * (x1 - x0)

    segs = sorted(segs, key=lambda s: x_at(s, ref_y))
    groups: list[list[YardLineSeg]] = []
    for s in segs:
        if groups and abs(x_at(groups[-1][-1], ref_y) - x_at(s, ref_y)) < x_tol:
            groups[-1].append(s)
        else:
            groups.append([s])

    merged: list[YardLineSeg] = []
    for g in groups:
        pts = np.array([p for s in g for p in (s.p0, s.p1)], dtype=np.float64)
        A = np.stack([pts[:, 1], np.ones(len(pts))], axis=1)
        a, b = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        merged.append(YardLineSeg((float(a * y0 + b), y0), (float(a * y1 + b), y1)))
    return merged


def _detect_sidelines(img_bgr, cfg):
    """Detect sidelines via HoughLinesP: near-horizontal long white lines spanning
    at least 40 % of the image width. Thresholds are tuned against real footage."""
    mask = _white_mask(img_bgr, cfg)
    # Against the WIDTH, as it always was. detect_lines measured against the
    # height and had to move to the short side for quarter-turned frames; this
    # gate was width-based from the start, so on a turned (portrait) frame
    # shape[1] is already the short side and nothing needed changing. It was
    # changed anyway, to min(H, W), which on every ordinary landscape frame
    # cut the bar from 768 px to 432 px -- a regression that review caught and
    # the tests could not, since their synthetic lines were 1800 px long.
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=120,
                           minLineLength=int(0.4 * img_bgr.shape[1]),
                           maxLineGap=cfg.max_line_gap_px)
    out = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in _hough_rows(segs):
        ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if ang < cfg.vertical_deg:
            out.append(YardLineSeg((float(x1), float(y1)), (float(x2), float(y2))))
    return out


def detect_hashes(img_bgr, cfg, player_boxes=None):
    """Hash ticks = small bright connected components (players masked out)."""
    mask = _zero_boxes(_white_mask(img_bgr, cfg), player_boxes)
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pts = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        if (cfg.hash_min_area <= area <= cfg.hash_max_area and h <= cfg.hash_max_h_px
                and w <= cfg.hash_max_h_px * 3):
            pts.append((float(cents[i][0]), float(cents[i][1])))
    return pts


def detect_field_features(img_bgr, *, cfg=None, player_boxes=None):
    cfg = cfg or FieldDetectConfig()
    H, W = img_bgr.shape[:2]
    return DetectedFeatures(
        yard_lines=detect_lines(img_bgr, cfg, player_boxes),
        sidelines=_detect_sidelines(img_bgr, cfg),
        hashes=detect_hashes(img_bgr, cfg, player_boxes),
        numbers=[],
        image_size=(W, H),
    )
