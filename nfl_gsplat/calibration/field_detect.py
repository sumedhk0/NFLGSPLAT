"""Detect field markings in a frame (cv2 lines/hashes + PaddleOCR numbers).

`detect_lines` (white-line detection + orientation split) is validated on
synthetic field images. `detect_field_features` adds hash + number detection,
whose thresholds are tuned against real footage at bring-up; PaddleOCR is reused
from the jersey-OCR path. The OCR/hash internals are the seam (monkeypatched in
register/orchestration tests).
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


def _white_mask(img_bgr: np.ndarray, cfg: FieldDetectConfig) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, cfg.white_thresh, 255, cv2.THRESH_BINARY)
    return m


def detect_lines(img_bgr: np.ndarray, cfg: FieldDetectConfig) -> list[YardLineSeg]:
    """Detect near-vertical painted yard-line segments via HoughLinesP."""
    H = img_bgr.shape[0]
    mask = _white_mask(img_bgr, cfg)
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
    """Near-horizontal long white lines = sidelines. Real-footage seam."""
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


def _ocr_numbers(img_bgr, masks, cfg):
    """OCR painted yard numbers. Real-footage seam — finalized at bring-up.
    Returns []; replaced with rectify-region + PaddleOCR (reuse jersey_ocr engine)
    against real frames."""
    return []


def _detect_hashes(img_bgr, cfg):
    """Detect hash ticks. Real-footage seam — finalized at bring-up. Returns []."""
    return []


def detect_field_features(
    img_bgr: np.ndarray, *, cfg: FieldDetectConfig = FieldDetectConfig(),
    masks=None,
) -> DetectedFeatures:
    H, W = img_bgr.shape[:2]
    return DetectedFeatures(
        yard_lines=detect_lines(img_bgr, cfg),
        sidelines=_detect_sidelines(img_bgr, cfg),
        hashes=_detect_hashes(img_bgr, cfg),
        numbers=_ocr_numbers(img_bgr, masks, cfg),
        image_size=(W, H),
    )
