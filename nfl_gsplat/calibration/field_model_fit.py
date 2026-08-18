"""Fit the metric NFL field model to accumulated endzone paint.

Labeling — deciding WHICH yard line each detected line is — is the step that
defeated single-frame field-marking calibration. It cannot be solved from line
geometry alone: equally-spaced parallel lines are TRANSLATION-INVARIANT along
the field, so sliding the whole set by one spacing gives an identical image. An
earlier version of this module inferred the offset from a yard-range prior and
returned a confident, wrong answer whenever that prior was shifted — which is
the normal case, since the prior comes from where players happen to stand.

The offset therefore comes from an explicit ANCHOR supplied once per game (the
camera is a fixed tripod, so one anchor covers the whole first half). The
yard-range prior is kept only as a VALIDATOR: it can reject a labeling, never
choose one.
"""
from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.calibration.field_landmarks import (
    GOAL_LINE_X_M,
    YARD_LINE_SPACING_M,
)
from nfl_gsplat.errors import CalibrationError


def _line_normal(seg: YardLineSeg) -> np.ndarray:
    """Unit normal of a segment's line (orientation-agnostic)."""
    p0 = np.asarray(seg.p0, float)
    p1 = np.asarray(seg.p1, float)
    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        raise CalibrationError("endzone field fit: zero-length line segment.")
    d = d / n
    return np.array([-d[1], d[0]])


def _offset(seg: YardLineSeg, normal: np.ndarray) -> float:
    return float(normal @ np.asarray(seg.p0, float))


def detect_accumulated_lines(votes, *, vote_thresh: float = 0.5,
                             min_len_frac: float = 0.25,
                             merge_tol_px: float = 12.0) -> list[YardLineSeg]:
    """Yard lines from the accumulated votes image — ONE segment per line.

    HoughLinesP returns many collinear fragments per painted line (more so where
    a cable breaks it), so fragments are grouped by orientation + perpendicular
    offset and each group is refit to a single spanning segment."""
    mask = (np.asarray(votes) >= vote_thresh).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=50,
                           minLineLength=int(min_len_frac * max(h, w)),
                           maxLineGap=40)
    if segs is None:
        return []
    raw = [YardLineSeg((float(a), float(b)), (float(c), float(d)))
           for a, b, c, d in np.asarray(segs).reshape(-1, 4)]

    groups: list[list[YardLineSeg]] = []
    keys: list[tuple[np.ndarray, float]] = []
    for s in raw:
        nrm = _line_normal(s)
        off = _offset(s, nrm)
        placed = False
        for gi, (gn, go) in enumerate(keys):
            align = float(gn @ nrm)
            if abs(align) < 0.985:                  # not parallel: different line
                continue
            if abs(go - _offset(s, gn)) <= merge_tol_px:
                groups[gi].append(s)
                placed = True
                break
        if not placed:
            groups.append([s])
            keys.append((nrm, off))

    merged: list[YardLineSeg] = []
    for g in groups:
        pts = np.array([p for s in g for p in (s.p0, s.p1)], float)
        mean = pts.mean(axis=0)
        direction = np.linalg.svd(pts - mean)[2][0]
        ts = (pts - mean) @ direction
        merged.append(YardLineSeg(tuple(mean + ts.min() * direction),
                                  tuple(mean + ts.max() * direction)))
    if not merged:
        return []
    ref_n = _line_normal(merged[0])
    merged.sort(key=lambda s: _offset(s, ref_n))
    return merged


def label_yard_lines(lines, *, anchor, yard_range_m=None,
                     max_spacing_ratio: float = 6.0) -> list[float]:
    """World X (metres) per detected line, in the caller's input order.

    ``anchor`` is ``((x, y), world_x_m)``: the detected line nearest that image
    point IS the yard line at ``world_x_m``. Without it the offset is
    undeterminable, so we raise rather than guess."""
    if anchor is None:
        raise CalibrationError(
            "endzone field fit: no yard-line anchor supplied. Equally-spaced "
            "yard lines are translation-invariant, so the offset cannot be "
            "inferred from the image — add an 'endzone_anchor' block to "
            "meta.yaml naming one line's world X (once per game).")
    if len(lines) < 2:
        raise CalibrationError(
            f"endzone field fit: need >= 2 yard lines to label, got {len(lines)}"
            " — accumulated paint too sparse.")

    (ax, ay), anchor_x = anchor
    if not np.isfinite(np.array([ax, ay, anchor_x], float)).all():
        raise CalibrationError("endzone field fit: anchor must be finite.")

    ref_n = _line_normal(lines[0])
    offs = np.array([_offset(s, ref_n) for s in lines], float)
    order = np.argsort(offs)
    sorted_offs = offs[order]
    if np.any(np.diff(sorted_offs) <= 0):
        raise CalibrationError(
            "endzone field fit: duplicate/coincident line offsets — merging "
            "failed; raise merge_tol_px or vote_thresh.")

    gaps = np.diff(sorted_offs)
    if gaps.max() / gaps.min() > max_spacing_ratio:
        raise CalibrationError(
            "endzone field fit: yard-line spacing inconsistent with one pencil "
            f"of lines (gap ratio {gaps.max() / gaps.min():.2f} > "
            f"{max_spacing_ratio}) — a line is probably missing or spurious.")

    a_off = float(ref_n @ np.array([ax, ay], float))
    k = int(np.argmin(np.abs(offs - a_off)))
    rank = int(np.where(order == k)[0][0])

    xs_sorted = [anchor_x + (i - rank) * YARD_LINE_SPACING_M
                 for i in range(len(lines))]
    worst = max(abs(x) for x in xs_sorted)
    if worst > GOAL_LINE_X_M + 1e-6:
        raise CalibrationError(
            f"endzone field fit: labeling runs off the painted field (|X| up to "
            f"{worst:.1f} m > {GOAL_LINE_X_M:.1f} m) — check the anchor's world X.")

    if yard_range_m is not None:
        arr = np.array([min(yard_range_m), max(yard_range_m)], float)
        if not np.isfinite(arr).all():
            raise CalibrationError("endzone field fit: yard_range_m must be finite.")
        lo, hi = float(arr[0]), float(arr[1])
        # VALIDATOR only — it may reject a labeling, never choose one.
        if max(xs_sorted) < lo - 15.0 or min(xs_sorted) > hi + 15.0:
            raise CalibrationError(
                f"endzone field fit: anchored labels {min(xs_sorted):.1f}.."
                f"{max(xs_sorted):.1f} m contradict the sideline yard range "
                f"{yard_range_m} — the anchor is probably wrong.")

    out = [0.0] * len(lines)
    for pos, idx in enumerate(order):
        out[int(idx)] = xs_sorted[pos]
    return out
