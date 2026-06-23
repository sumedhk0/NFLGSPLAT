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
    H = img_bgr.shape[0]
    mask = _zero_boxes(_white_mask(img_bgr, cfg), player_boxes)
    min_len = int(cfg.min_line_len_frac * H)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=80,
                           minLineLength=min_len, maxLineGap=cfg.max_line_gap_px)
    out: list[YardLineSeg] = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in segs[:, 0, :]:
        ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if ang >= (90 - cfg.vertical_deg):
            out.append(YardLineSeg((float(x1), float(y1)), (float(x2), float(y2))))
    return _merge_collinear(out)


def _merge_collinear(segs: list[YardLineSeg], x_tol: float = 18.0) -> list[YardLineSeg]:
    """Merge near-vertical segments with similar mean-x into one spanning segment."""
    segs = sorted(segs, key=lambda s: 0.5 * (s.p0[0] + s.p1[0]))
    merged: list[YardLineSeg] = []
    for s in segs:
        x = 0.5 * (s.p0[0] + s.p1[0])
        if merged and abs(0.5 * (merged[-1].p0[0] + merged[-1].p1[0]) - x) < x_tol:
            prev = merged[-1]
            ys = [prev.p0[1], prev.p1[1], s.p0[1], s.p1[1]]
            xs = [prev.p0[0], prev.p1[0], s.p0[0], s.p1[0]]
            mx = float(np.mean(xs))
            merged[-1] = YardLineSeg((mx, float(min(ys))), (mx, float(max(ys))))
        else:
            merged.append(s)
    return merged


def _detect_sidelines(img_bgr, cfg):
    """Detect sidelines via HoughLinesP: near-horizontal long white lines spanning
    at least 40 % of the image width. Thresholds are tuned against real footage."""
    mask = _white_mask(img_bgr, cfg)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=120,
                           minLineLength=int(0.4 * img_bgr.shape[1]),
                           maxLineGap=cfg.max_line_gap_px)
    out = []
    if segs is None:
        return out
    for x1, y1, x2, y2 in segs[:, 0, :]:
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
