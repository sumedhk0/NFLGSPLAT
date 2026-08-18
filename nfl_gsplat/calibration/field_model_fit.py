"""Fit the metric NFL field model to accumulated endzone paint.

Labeling (deciding WHICH yard line each detected line is) is the step that
defeated single-frame field-marking calibration: the painted numbers face the
sideline, so from the endzone they are edge-on and unreadable. Two things make
it tractable here: the accumulated mosaic spans far more of the field than any
one frame, and the calibrated SIDELINE camera says which yard range is in view.

When the evidence admits more than one labeling, this module raises rather than
guessing — a wrong label offset is exactly how the earlier approach produced a
plausible-looking camera that fit nothing.
"""
from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
from nfl_gsplat.errors import CalibrationError


def detect_accumulated_lines(votes, *, vote_thresh: float = 0.5,
                             min_len_frac: float = 0.10):
    """Yard-line candidates from the votes image, as (a, b) with x = a*y + b."""
    mask = (votes >= vote_thresh).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60,
                           minLineLength=int(min_len_frac * max(h, w)),
                           maxLineGap=25)
    if segs is None:
        return []
    out = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        if abs(y2 - y1) < 1e-6:
            continue
        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1
        out.append((float(a), float(b)))
    return sorted(out, key=lambda ab: ab[1])


def label_yard_lines(lines, *, yard_range_m, spacing_px_hint=None,
                      tol_m: float = 0.6):
    """World X (metres) for each detected line, or raise if ambiguous.

    Lines are equally spaced by YARD_LINE_SPACING_M in world X, so their image
    order maps to a uniform world sequence; only the OFFSET (which line is
    which) is unknown. We enumerate offsets that place every line inside the
    sideline-derived ``yard_range_m`` window and require exactly one to fit.

    ``spacing_px_hint`` is accepted for interface parity with future callers
    that pass an image-space spacing estimate (e.g. to sanity-check the pixel
    gaps against expected image spacing before world-labeling); it is not
    used by the current pure-prior-window enumeration.
    """
    if len(lines) < 2:
        raise CalibrationError(
            "endzone field fit: need >= 2 yard lines to label, got "
            f"{len(lines)} — accumulated paint too sparse.")
    bs = np.array([b for _a, b in lines], float)
    gaps = np.diff(bs)
    if gaps.min() <= 0:
        raise CalibrationError("endzone field fit: lines are not ordered.")
    # Uniform image spacing is only expected under mild perspective; require the
    # gaps to be self-consistent, else the detections are not a clean pencil.
    if gaps.max() / gaps.min() > 1.6:
        raise CalibrationError(
            "endzone field fit: inconsistent yard-line spacing "
            f"(gap ratio {gaps.max() / gaps.min():.2f}) — detections are not a "
            "clean set of yard lines; raise vote_thresh or re-check masking.")

    n = len(lines)
    lo, hi = float(min(yard_range_m)), float(max(yard_range_m))
    span = (n - 1) * YARD_LINE_SPACING_M
    # Candidate offsets: start X so that the whole run sits within the window
    # (widened by tol_m, since the prior is approximate).
    starts = []
    k = -200
    while True:
        x0 = k * YARD_LINE_SPACING_M
        if x0 > hi + tol_m:
            break
        if x0 >= lo - tol_m and x0 + span <= hi + tol_m:
            starts.append(x0)
        k += 1
    if not starts:
        raise CalibrationError(
            "endzone field fit: no yard-line labeling fits the sideline yard "
            f"range {yard_range_m} for {n} lines spanning {span:.1f} m — the "
            "prior and the detections disagree.")
    if len(starts) > 1:
        raise CalibrationError(
            f"endzone field fit: labeling ambiguous — {len(starts)} offsets fit "
            f"the yard range {yard_range_m} equally well. Narrow the range "
            "(more sideline evidence) or supply an absolute anchor.")
    x0 = starts[0]
    return [x0 + i * YARD_LINE_SPACING_M for i in range(n)]
