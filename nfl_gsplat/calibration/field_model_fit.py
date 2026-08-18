"""Fit the metric NFL field model to accumulated endzone paint.

Labeling — deciding WHICH yard line each detected line is — is the step that
defeated single-frame field-marking calibration (labels 67-210 px wrong,
producing a plausible camera that fit nothing). It cannot be solved from line
geometry alone: equally-spaced parallel lines are TRANSLATION-INVARIANT along
the field, so sliding the whole set by one spacing gives an identical image.

v1 of this module inferred the offset from a yard-range prior and returned a
confident, wrong answer whenever that prior was shifted — which is the normal
case, since the prior comes from where players happen to stand.

v2 fixed that with a single explicit ANCHOR, but one anchor pins translation
only, not DIRECTION: the reference normal used to order the lines inherits an
arbitrary endpoint order (ultimately an SVD sign from the line-merge step), so
the identical physical lines could label either [-18.288 .. 0.0] or mirrored
[0.0 .. -18.288]. The yard-range validator provably cannot catch a mirror
about the anchor, since it leaves min/max unchanged. Separately, labels were
assigned purely by RANK, so one missing or over-merged line silently shifted
every later label by a full YARD_LINE_SPACING_M, and a loose gap-ratio gate
could not see it. v2 also measured offsets from a segment's p0, which made the
measured gap depend on which endpoint the detector happened to emit first —
under perspective that artifact was large enough to swamp the real spacing.

v3 (current): TWO anchors, offsets measured at the segment MIDPOINT. Two
anchored lines pin translation, direction, AND spacing consistency in one
stroke — the step implied by the two anchors must come out to exactly
+-YARD_LINE_SPACING_M, which is violated precisely when a line is missing,
spurious, or over-merged. This is still "one-time per game" for the human
(the tripod shares one camera centre across the half); they name two lines
instead of one. The yard-range prior remains a VALIDATOR only: it may reject
a labeling, never choose one.
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
    """Perpendicular offset of a segment, measured at its MIDPOINT.

    Projecting from p0 makes the measured gaps depend on which endpoint the
    detector happened to emit first, which under perspective injects an
    artifact large enough to swamp the real spacing (measured: true gap ratio
    1.23 read as 4.15)."""
    mid = 0.5 * (np.asarray(seg.p0, float) + np.asarray(seg.p1, float))
    return float(normal @ mid)


def _on_yard_grid(x: float, tol_m: float = 0.05) -> bool:
    return abs(x / YARD_LINE_SPACING_M - round(x / YARD_LINE_SPACING_M)) \
        * YARD_LINE_SPACING_M <= tol_m


def detect_accumulated_lines(votes, *, vote_thresh: float = 0.5,
                             min_len_frac: float = 0.25,
                             merge_tol_px: float = 12.0) -> list[YardLineSeg]:
    """Yard lines from the accumulated votes image — ONE segment per line.

    HoughLinesP returns many collinear fragments per painted line (more so where
    a cable breaks it), so fragments are grouped by orientation + perpendicular
    offset and each group is refit to a single spanning segment. After merging,
    raises if the smallest surviving gap is not comfortably larger than
    merge_tol_px — over-merging (two distinct lines fused into one) is how a
    "missing line" appears, and it silently shifts every later label."""
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

    # Over-merge guard: two physical lines closer together than merge_tol_px
    # get fused into the SAME group by the clustering loop above, before any
    # gap ever exists between separate groups for a between-group check to
    # see — a gap check on the final `merged` list is structurally blind to
    # this (verified empirically: two lines 6 px apart with the default
    # merge_tol_px=12 collapse into one group, leaving only the huge gap to
    # the next, unrelated line for a between-group check to inspect). Instead
    # check the perpendicular residual SPREAD of the raw points *within* each
    # group against its own best-fit line: genuine fragments of one physical
    # line cluster tightly around that fit (observed std ~1.4 px for a 3 px
    # thick painted line broken by a cable gap); two distinct lines fused
    # together leave two offset clusters, roughly doubling that std (observed
    # 3.0 px for two 1-px lines 6 px apart). The threshold is scaled off
    # merge_tol_px so it stays meaningful if the caller changes it.
    merged: list[YardLineSeg] = []
    for g in groups:
        pts = np.array([p for s in g for p in (s.p0, s.p1)], float)
        mean = pts.mean(axis=0)
        direction = np.linalg.svd(pts - mean)[2][0]
        normal = np.array([-direction[1], direction[0]])
        perp = (pts - mean) @ normal
        residual_std = float(perp.std())
        spread_limit = merge_tol_px / 5.0
        if residual_std > spread_limit:
            raise CalibrationError(
                f"endzone field fit: a merged line's points spread "
                f"{residual_std:.1f} px (std) perpendicular to its own best "
                f"fit, above the {spread_limit:.1f} px limit ({merge_tol_px} "
                "merge_tol_px / 5) — two distinct lines were probably fused "
                "into one, which would silently shift every later label. "
                "Lower merge_tol_px or raise vote_thresh.")
        ts = (pts - mean) @ direction
        merged.append(YardLineSeg(tuple(mean + ts.min() * direction),
                                  tuple(mean + ts.max() * direction)))
    if not merged:
        return []
    ref_n = _line_normal(merged[0])
    merged.sort(key=lambda s: _offset(s, ref_n))
    return merged


def label_yard_lines(lines, *, anchors, yard_range_m=None) -> list[float]:
    """World X per detected line, from TWO anchored lines.

    Two anchors pin translation, DIRECTION, and spacing consistency at once.
    One anchor cannot: it leaves the sign free (the identical lines then label
    mirrored), and rank-based labeling silently absorbs a missing line. The
    step implied by the two anchors must come out to exactly +-spacing, which
    is violated precisely when a line is missing, spurious or over-merged."""
    if anchors is None or len(anchors) != 2:
        raise CalibrationError(
            "endzone field fit: need TWO yard-line anchors "
            "(((x1,y1), world_x1), ((x2,y2), world_x2)). One anchor leaves the "
            "labeling direction free and cannot detect a missing line. Add an "
            "'endzone_anchor' block naming two lines (once per game).")
    if len(lines) < 2:
        raise CalibrationError(
            f"endzone field fit: need >= 2 yard lines, got {len(lines)}.")

    for (pt, wx) in anchors:
        if not np.isfinite(np.asarray([pt[0], pt[1], wx], float)).all():
            raise CalibrationError("endzone field fit: anchors must be finite.")
        if not _on_yard_grid(wx):
            raise CalibrationError(
                f"endzone field fit: anchor world_x {wx} is not on the 5-yard "
                f"grid (multiples of {YARD_LINE_SPACING_M:.3f} m) — check the "
                "value read off the mosaic.")

    ref_n = _line_normal(lines[0])
    offs = np.array([_offset(s, ref_n) for s in lines], float)
    order = np.argsort(offs)
    rank_of = {int(idx): pos for pos, idx in enumerate(order)}
    gaps = np.diff(offs[order])
    if np.any(gaps <= 0):
        raise CalibrationError(
            "endzone field fit: coincident line offsets — merging failed.")

    idxs = []
    for (pt, _wx) in anchors:
        d = np.abs(offs - float(ref_n @ np.asarray(pt, float)))
        k = int(np.argmin(d))
        # the match must be clearly nearer than the next line, else a slightly
        # misplaced human click silently yields an off-by-one labeling
        if d[k] > 0.4 * float(gaps.min()):
            raise CalibrationError(
                f"endzone field fit: anchor at {pt} is {d[k]:.1f} px from the "
                f"nearest line, more than 40% of the smallest line gap "
                f"({gaps.min():.1f} px) — click closer to a line.")
        idxs.append(k)
    if idxs[0] == idxs[1]:
        raise CalibrationError(
            "endzone field fit: both anchors matched the SAME line; they must "
            "name two distinct yard lines.")

    r0, r1 = rank_of[idxs[0]], rank_of[idxs[1]]
    x0, x1 = float(anchors[0][1]), float(anchors[1][1])
    step = (x1 - x0) / (r1 - r0)
    if abs(abs(step) - YARD_LINE_SPACING_M) > 0.05:
        raise CalibrationError(
            f"endzone field fit: the two anchors imply a step of {step:.3f} m "
            f"per detected line, not +-{YARD_LINE_SPACING_M:.3f} m — a yard "
            "line is missing, spurious, or two lines were merged into one.")

    xs_by_rank = [x0 + (pos - r0) * step for pos in range(len(lines))]
    worst = max(abs(x) for x in xs_by_rank)
    if worst > GOAL_LINE_X_M + 1e-6:
        raise CalibrationError(
            f"endzone field fit: labeling runs off the painted field (|X| up to "
            f"{worst:.1f} m > {GOAL_LINE_X_M:.1f} m) — check the anchors.")

    if yard_range_m is not None:
        arr = np.asarray(yard_range_m, float)
        if arr.size != 2 or not np.isfinite(arr).all():
            raise CalibrationError(
                "endzone field fit: yard_range_m must be two finite values.")
        lo, hi = float(arr.min()), float(arr.max())
        if max(xs_by_rank) < lo - 15.0 or min(xs_by_rank) > hi + 15.0:
            raise CalibrationError(
                f"endzone field fit: anchored labels {min(xs_by_rank):.1f}.."
                f"{max(xs_by_rank):.1f} m contradict the sideline yard range "
                f"{yard_range_m} — the anchors are probably wrong.")

    out = [0.0] * len(lines)
    for pos, idx in enumerate(order):
        out[int(idx)] = xs_by_rank[pos]
    return out
