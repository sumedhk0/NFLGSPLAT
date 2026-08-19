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

v3: TWO anchors, offsets measured at the segment MIDPOINT. Two anchored lines
pin translation, direction, AND spacing consistency in one stroke — the step
implied by the two anchors must come out to exactly +-YARD_LINE_SPACING_M,
which is violated precisely when a line is missing, spurious, or over-merged.
This is still "one-time per game" for the human (the tripod shares one camera
centre across the half); they name two lines instead of one. The yard-range
prior remains a VALIDATOR only: it may reject a labeling, never choose one.

v3 first shipped with the step check applied only to the SPAN BETWEEN the two
anchors, plus a candidate over-merge guard inside detect_accumulated_lines.
Both were measured unsound:
- A missing/spurious/over-merged line OUTSIDE the anchor span was invisible
  to the between-anchor step check (measured: 5 lines with the 4th absent,
  anchors on the two adjacent inner lines, returned a labeling whose last
  entry was wrong by a full spacing — no error raised).
- Two candidate over-merge guards were tried inside detect_accumulated_lines
  (a between-merged-group gap check, then a within-group residual-spread
  check) and both were measured unsound: the gap check is structurally blind
  to lines closer than merge_tol_px (they fuse into one group during
  clustering before any between-group gap exists to inspect), and the
  residual-spread check has no safe threshold — calibrated at
  merge_tol_px/5 it false-positives on legitimate paint >= 7 px thick
  (measured std 2.46 px) while silently missing over-merges of lines
  <= 4 px apart (measured std 2.00 px), which are exactly the distant,
  compressed lines that actually over-merge in an endzone view.

v3.1 (current): the anchors are now required to be the OUTERMOST two detected
lines, which makes the step check span every line, not just the interval
between the anchors — a missing/spurious/over-merged line anywhere now breaks
it. This also subsumes the over-merge guard: over-merging reduces the line
COUNT by one, so the implied step becomes (N)/(N-1) of a spacing and the
+-0.05 m step check fires. That is a count invariant, independent of paint
thickness and line separation, so detect_accumulated_lines carries no
over-merge check of its own.
"""
from __future__ import annotations

import dataclasses
import itertools

import cv2
import numpy as np

from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.calibration.field_landmarks import (
    GOAL_LINE_X_M,
    HASH_OFFSET_M,
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


def _largest_parallel_cluster(lines: list[YardLineSeg],
                              align_tol: float = 0.9,
                              expect_normal=None) -> list[YardLineSeg]:
    """Partition merged lines into components of mutually near-parallel
    normals (|n_i . n_j| >= align_tol) and return only the largest.

    Everything downstream of this (a single ``ref_n``, ranks, gaps, the
    outermost-span rule) is only meaningful for one pencil of NEAR-PARALLEL
    yard lines. Real endzone footage always shows both sidelines too, which
    detect_accumulated_lines' fragment-merge returns right alongside the yard
    lines regardless of orientation. Two sidelines can even be symmetric
    about the image centre, giving them IDENTICAL offsets under a yard-line
    reference normal (measured: gaps.min() collapses to ~0, which then makes
    the anchor-margin check reject a pixel-perfect anchor click). ``align_tol``
    is looser than the 0.985 used to merge raw Hough fragments into one line,
    to tolerate the perspective fan of a real pencil of parallel field lines.

    With ``expect_normal`` the cluster ALIGNED with that normal is returned
    instead of the biggest one. Size is a good proxy for "these are the yard
    lines" in an accumulated mosaic, where the yard-line pencil dominates
    everything else; it is a bad one in a SINGLE frame, where a tight zoom can
    leave more fragments along the hash columns, the numbers, or a sideline
    than along the two or three yard lines actually in view. Measured on
    play_001 frame 648 -- the very frame the reference camera was solved on --
    size returned eleven near-VERTICAL lines while the yard lines run
    horizontally, so every association failed at ~470 px and the best camera in
    the play could not verify itself. The caller knows where yard lines should
    point, because it has a camera; that is strictly better evidence than
    counting fragments."""
    n = len(lines)
    normals = [_line_normal(s) for s in lines]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(normals[i] @ normals[j])) >= align_tol:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    if expect_normal is not None:
        want = np.asarray(expect_normal, float)
        want = want / max(float(np.hypot(want[0], want[1])), 1e-12)

        def _alignment(idxs):
            """How parallel this cluster is to the expected direction."""
            return max(abs(float(normals[i] @ want)) for i in idxs)

        best = max(clusters.values(), key=lambda idxs: (_alignment(idxs),
                                                        len(idxs)))
        if _alignment(best) < align_tol:
            raise CalibrationError(
                "no detected line family points the way the camera says yard "
                f"lines should ({len(clusters)} family/families, best "
                f"alignment {_alignment(best):.3f} < {align_tol}).")
        return [lines[i] for i in sorted(best)]
    sizes = sorted((len(idxs) for idxs in clusters.values()), reverse=True)
    if sizes[0] < 2:
        raise CalibrationError(
            "endzone field fit: no cluster of >= 2 mutually-parallel "
            "detected lines — the accumulated mosaic has no clean yard-line "
            "pencil (only sidelines/noise survived); sample frames with "
            "more of the field visible.")
    if len(sizes) > 1 and sizes[0] == sizes[1]:
        raise CalibrationError(
            f"endzone field fit: two equally-sized clusters of "
            f"{sizes[0]} mutually-parallel lines were detected — cannot "
            "tell which is the yard-line pencil (e.g. yard lines vs. "
            "sidelines); check the accumulated mosaic.")
    best = max(clusters.values(), key=len)
    return [lines[i] for i in best]


def detect_accumulated_lines(votes, *, vote_thresh: float = 0.5,
                             min_len_frac: float = 0.25,
                             merge_tol_px: float = 12.0,
                             expect_normal=None) -> list[YardLineSeg]:
    """Yard lines from the accumulated votes image — ONE segment per line.

    HoughLinesP returns many collinear fragments per painted line (more so where
    a cable breaks it), so fragments are grouped by orientation + perpendicular
    offset and each group is refit to a single spanning segment.

    Deliberately NO over-merge guard here: two candidates were measured unsound
    (see the module docstring) — a between-group gap check is structurally
    blind to lines closer than merge_tol_px, and a within-group residual-spread
    check has no safe threshold, false-positiving on thick legitimate paint
    while missing over-merges of lines just a few px apart. Over-merging is
    instead caught in label_yard_lines as a line-COUNT violation: the outermost
    two anchors must span exactly len(lines) - 1 steps of YARD_LINE_SPACING_M,
    and a merge changes that count.

    After merging, only the LARGEST mutually-parallel cluster of merged lines
    is returned (see :func:`_largest_parallel_cluster`) — real endzone footage
    always shows both sidelines, which are not part of the near-parallel
    yard-line pencil everything downstream assumes, and can even collide in
    offset with each other under a yard-line reference normal."""
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
    # Keep only the largest mutually-parallel cluster BEFORE any offset/rank/
    # gap math runs (see _largest_parallel_cluster docstring): real footage
    # always shows both sidelines, which are not part of the yard-line
    # pencil the rest of this function (and label_yard_lines) assumes.
    merged = _largest_parallel_cluster(merged,
                                       expect_normal=expect_normal)
    ref_n = _line_normal(merged[0])
    merged.sort(key=lambda s: _offset(s, ref_n))
    return merged


def label_yard_lines(lines, *, anchors, yard_range_m=None,
                     indices=None) -> list[float]:
    """World X per detected line, from TWO anchored lines.

    Two anchors pin translation, DIRECTION, and spacing consistency at once.
    One anchor cannot: it leaves the sign free (the identical lines then label
    mirrored).

    v4 labels by LADDER INDEX, not rank. Ranks assume every yard line in the
    span was detected, which real footage violates -- players stand on the
    paint. Measured on play_001: two of nine lines are masked by the
    line-of-scrimmage crowd, so the seven survivors span eight spacings, and
    the rank form computed 36.576/6 = 6.096 m and rejected a labeling that was
    in fact recoverable. With indices the step check still spans every line and
    still catches a spurious or over-merged line, because those change an
    index, not just a count."""
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
                f"grid (multiples of {YARD_LINE_SPACING_M:.3f} m) - check the "
                "value read off the mosaic.")

    idx = list(indices) if indices is not None else fit_yard_ladder(lines)
    if len(idx) != len(lines):
        raise CalibrationError(
            f"endzone field fit: got {len(idx)} ladder indices for "
            f"{len(lines)} lines.")

    ref_n = _line_normal(lines[0])
    offs = np.array([_offset(s, ref_n) for s in lines], float)
    order = np.argsort(offs)
    gaps = np.diff(offs[order])
    if np.any(gaps <= 0):
        raise CalibrationError(
            "endzone field fit: coincident line offsets - merging failed.")
    rank_of = {int(line_i): pos for pos, line_i in enumerate(order)}

    matched = []
    for (pt, _wx) in anchors:
        d = np.abs(offs - float(ref_n @ np.asarray(pt, float)))
        k = int(np.argmin(d))
        # the match must be clearly nearer than the next line, else a slightly
        # misplaced human click silently yields an off-by-one labeling.
        # Gated on the LOCAL gap at the matched line (the smaller of its
        # neighbouring gaps), not the GLOBAL minimum gap across every detected
        # line: on a full-field render, far/compressed lines can shrink the
        # global minimum well below the gap around the widely-separated
        # outermost lines an operator is actually told to anchor, forcing an
        # unreasonably tight click tolerance even there (measured: 3.4-5.4 px
        # on a 1920-px mosaic).
        pos = rank_of[k]
        local_gaps = [gaps[i] for i in (pos - 1, pos) if 0 <= i < len(gaps)]
        local_gap = float(min(local_gaps))
        if d[k] > 0.4 * local_gap:
            raise CalibrationError(
                f"endzone field fit: anchor at {pt} is {d[k]:.1f} px from the "
                f"nearest line, more than 40% of that line's local gap "
                f"({local_gap:.1f} px) - click closer to a line.")
        matched.append(k)
    if matched[0] == matched[1]:
        raise CalibrationError(
            "endzone field fit: both anchors matched the SAME line; they must "
            "name two distinct yard lines.")

    i0, i1 = idx[matched[0]], idx[matched[1]]
    # The anchors MUST be the outermost detected lines, so the step check below
    # spans every line. Anchoring inner lines would leave a mislabelled line
    # outside that span unchecked.
    if {i0, i1} != {min(idx), max(idx)}:
        raise CalibrationError(
            f"endzone field fit: the two anchors sit at ladder indices {i0} "
            f"and {i1}, but the detected lines span {min(idx)}..{max(idx)}. "
            "Anchor the OUTERMOST two lines, so the spacing check covers every "
            "line.")
    x0, x1 = float(anchors[0][1]), float(anchors[1][1])
    step = (x1 - x0) / (i1 - i0)
    if abs(abs(step) - YARD_LINE_SPACING_M) > 0.05:
        raise CalibrationError(
            f"endzone field fit: the two anchors imply a step of {step:.3f} m "
            f"per yard-grid index, not +-{YARD_LINE_SPACING_M:.3f} m - a line "
            "is spurious, or two lines were merged into one. Detected "
            f"{len(lines)} lines at ladder indices {sorted(idx)} and offsets "
            f"{np.round(offs[order], 1).tolist()} px.")

    xs = [x0 + (k - i0) * step for k in idx]
    worst = max(abs(x) for x in xs)
    if worst > GOAL_LINE_X_M + 1e-6:
        raise CalibrationError(
            f"endzone field fit: labeling runs off the painted field (|X| up to "
            f"{worst:.1f} m > {GOAL_LINE_X_M:.1f} m) - check the anchors.")

    if yard_range_m is not None:
        arr = np.asarray(yard_range_m, float)
        if arr.size != 2 or not np.isfinite(arr).all():
            raise CalibrationError(
                "endzone field fit: yard_range_m must be two finite values.")
        lo, hi = float(arr.min()), float(arr.max())
        if max(xs) < lo - 15.0 or min(xs) > hi + 15.0:
            raise CalibrationError(
                f"endzone field fit: anchored labels {min(xs):.1f}.."
                f"{max(xs):.1f} m contradict the sideline yard range "
                f"{yard_range_m} - the anchors are probably wrong.")
    return xs



# --- ladder + hash rows -----------------------------------------------------
#
# v4 (2026-08-18), forced by the first real run on SEA@AZ play_001.
#
# Two v3.1 assumptions were measured FALSE on real footage:
#
# 1. "every detected line is present" -- label_yard_lines assigned labels by
#    RANK, so its step check computed (x_last - x_first)/(N-1). On play_001 the
#    line-of-scrimmage crowd masks the paint of TWO yard lines, so 7 lines span
#    8 spacings: the check computed 36.576/6 = 6.096 m and rejected data whose
#    labeling was in fact recoverable. Missing lines are NORMAL, not an error --
#    players stand on the paint. fit_yard_ladder recovers the integer INDEX of
#    each line instead of its rank, so gaps are explicit.
#
# 2. "each merged line spans sideline to sideline" -- the driver mapped a
#    line's two ENDPOINTS to world Y = +-HALF_WIDTH_M. On play_001 six of seven
#    lines terminate at x=1919, the IMAGE EDGE, and the endzone camera sees only
#    a ~14 m wide strip (measured: the offensive line spans 860 px ~ 5.6 m), so
#    those endpoints sit at |Y| ~ 4-10 m. Yard lines are mutually parallel and
#    leave 3 DOF of the homography unobservable; the endpoint trick was the only
#    thing supplying them, and it supplied them wrongly.
#
# detect_hash_columns supplies the missing cross-field constraint from the HASH
# ROWS, which are genuinely at a known world Y (+-HASH_OFFSET_M). They are in
# the mosaic but invisible to an ANGLE filter: a hash mark is painted PARALLEL
# to the yard lines, so it lands in the same orientation cluster. LENGTH is what
# separates them (detect_accumulated_lines' min_len_frac correctly keeps them
# out of the long-line set).

HASH_MARK_LEN_M: float = 0.6096          # 2 ft, painted across the field
HASH_MARK_PITCH_M: float = 0.9144        # one per yard along the row

LADDER_MAX_INDEX_GAP = 4
LADDER_MAX_RMS_PX = 3.0
LADDER_MIN_MARGIN = 2.0
LADDER_MAX_LINES = 15


def _fit_projective_1d(idx, off):
    """Least-squares fit of ``off = (a*idx + b) / (c*idx + 1)``.

    Equally-spaced parallel world lines map to a 1-D PROJECTIVE function of
    their integer index, measured along any fixed transversal direction -- so
    the perpendicular offsets used here are a valid parametrisation. Returns
    ``(rms, params)``; ``rms`` is ``inf`` for a degenerate fit."""
    idx = np.asarray(idx, float)
    off = np.asarray(off, float)
    a = np.column_stack([idx, np.ones_like(idx), -idx * off])
    try:
        sol = np.linalg.solve(a.T @ a, a.T @ off)
    except np.linalg.LinAlgError:
        return float("inf"), None
    denom = sol[2] * idx + 1.0
    if np.any(np.abs(denom) < 1e-9):
        return float("inf"), None
    pred = (sol[0] * idx + sol[1]) / denom
    return float(np.sqrt(np.mean((pred - off) ** 2))), sol


def _ladder_search(lines, *, max_index_gap, max_rms_px, min_margin):
    """Shared ladder search. Returns ``(indices, rms, second_rms)``.

    ``indices[i]`` is the yard-grid index of ``lines[i]``; 0 is the line of
    smallest perpendicular offset."""
    n = len(lines)
    if n > LADDER_MAX_LINES:
        raise CalibrationError(
            f"endzone field fit: {n} yard lines detected (max "
            f"{LADDER_MAX_LINES}); the ladder search is exponential in the "
            "line count and this many lines usually means the mosaic is "
            "over-detecting - raise min_len_frac or vote_thresh.")
    ref_n = _line_normal(lines[0])
    offs = np.array([_offset(s, ref_n) for s in lines], float)
    order = np.argsort(offs)
    sorted_offs = offs[order]
    if np.any(np.diff(sorted_offs) <= 0):
        raise CalibrationError(
            "endzone field fit: coincident line offsets - merging failed.")

    out = [0] * n
    if n < 4:
        for pos, line_i in enumerate(order):
            out[int(line_i)] = pos
        return out, 0.0, float("inf")

    scored = []
    for gaps in itertools.product(range(1, max_index_gap + 1), repeat=n - 1):
        idx = np.concatenate([[0.0], np.cumsum(gaps)])
        rms, _ = _fit_projective_1d(idx, sorted_offs)
        if np.isfinite(rms):
            scored.append((rms, sum(gaps), gaps))
    if not scored:
        raise CalibrationError(
            "endzone field fit: no projective ladder could be fitted to the "
            f"{n} detected lines - they are probably not one pencil of yard "
            "lines.")
    best_rms = min(r for r, _sp, _g in scored)
    # Ties go to the SMALLEST span -- i.e. assume no line is missing unless the
    # geometry says one is. Exactly-spaced synthetic lines are fitted perfectly
    # by many index assignments (every candidate scores 0.00 px), so there the
    # margin test below has nothing to measure; Occam is the right tie-break,
    # and it also keeps the gate from firing on clean data.
    tie = max(1e-9, 0.02 * best_rms)
    window = [c for c in scored if c[0] <= best_rms + tie]
    min_span = min(span for _r, span, _g in window)
    finalists = {g for _r, span, g in window if span == min_span}
    if len(finalists) > 1:
        # Distinct gap patterns of the SAME span fit equally well, so where the
        # missing line sits is not recoverable -- e.g. offsets 100/150/200/300
        # are fitted exactly by both (1,1,2) and (2,1,1): the absent line could
        # be at either end. Guessing would shift every label past it by a full
        # spacing, which is the failure this module exists to prevent. (Uniform
        # rescalings like (2,2,2,2) are NOT this case -- they differ in span, so
        # the smallest wins and the anchor step check rejects the rest.)
        raise CalibrationError(
            "endzone field fit: the yard-line spacing is ambiguous - "
            f"{len(finalists)} different index assignments of the same span fit "
            f"the lines equally well ({sorted(finalists)}), so which line is "
            "missing cannot be determined. Detected "
            f"{n} lines at offsets {np.round(sorted_offs, 1).tolist()} px. "
            "More lines, or more perspective spread between them, are needed.")
    best_gaps = finalists.pop()
    # The runner-up must be a genuinely DIFFERENT reading, not the same one
    # rescaled. Multiplying every gap by k describes the identical geometry
    # with the index relabelled -- (2,2,2,...) fits exactly as well as
    # (1,1,1,...) always -- so counting it as the runner-up made the margin
    # gate fire on every clean, evenly-spaced ladder. (Real play_001 slipped
    # through only because its true pattern (1,1,1,3,1,1) rescales to a gap of
    # 6, outside max_index_gap.) Rescalings are already rejected downstream by
    # the anchor step check, which would see S/k instead of S.
    def _shape(gaps):
        from math import gcd
        from functools import reduce
        d = reduce(gcd, gaps)
        return tuple(g // d for g in gaps)

    best_shape = _shape(best_gaps)
    others = [r for r, _sp, g in scored if _shape(g) != best_shape]
    second = min(others) if others else float("inf")
    idx = np.concatenate([[0], np.cumsum(best_gaps)]).astype(int)
    for pos, line_i in enumerate(order):
        out[int(line_i)] = int(idx[pos])
    return out, best_rms, second


def _gate_ladder(rms, second, *, max_rms_px, min_margin, n,
                 exact_px: float = 0.05):
    # Below `exact_px` the fit is essentially perfect and the runner-up is too:
    # there is no noise for a margin to be measured against, so demanding one
    # would reject clean data. The span tie-break in _ladder_search has already
    # chosen the simplest assignment in that case.
    if rms <= exact_px:
        return
    if rms > max_rms_px:
        raise CalibrationError(
            f"endzone field fit: best yard-line ladder fits at {rms:.2f} "
            f"px (limit {max_rms_px:.2f}) - the detected lines do not lie on a "
            "consistent yard grid; check the accumulated mosaic for spurious "
            "or over-merged lines.")
    if second < min_margin * max(rms, 0.1):
        raise CalibrationError(
            "endzone field fit: the yard-line spacing is ambiguous - the best "
            f"index assignment fits at {rms:.2f} px and the next-best at "
            f"{second:.2f} px, short of the {min_margin:.1f}x margin required. "
            "Too few lines survived, or too many are missing, to tell which "
            "yard lines they are.")


def fit_yard_ladder(lines, *, max_index_gap: int = LADDER_MAX_INDEX_GAP,
                    max_rms_px: float = LADDER_MAX_RMS_PX,
                    min_margin: float = LADDER_MIN_MARGIN) -> list[int]:
    """Integer yard-line index per detected line, tolerating MISSING lines.

    Returns ``out[i]`` = the index of ``lines[i]`` on the yard grid, with 0 at
    the line of smallest perpendicular offset. Consecutive indices differ by
    the number of SPACINGS between them, so a line hidden under players leaves
    a gap of 2 rather than shifting every later label by a full spacing (the
    v3.1 rank failure).

    The correct assignment is chosen by fit quality, not by assumption: every
    combination of per-step gaps in ``1..max_index_gap`` is fitted and the best
    must beat the runner-up by ``min_margin``. Measured on play_001: the true
    assignment (gaps 1,1,1,3,1,1) fits at 0.862 px against 5.818 px for the
    next-best, a 6.8x margin, so the gate has real headroom.

    Every input line must belong to the pencil -- one spurious line makes the
    whole set inconsistent (measured on play_001: a single stray line at 1.9
    degrees among lines within 0.4 degrees took the best fit from 0.862 px to
    16.29 px). Use :func:`prune_to_yard_ladder` when the caller cannot
    guarantee that.

    Fewer than 4 lines cannot discriminate -- the fit has 3 parameters, so 3
    points are matched exactly by every candidate -- and fall back to contiguous
    indices, leaving the anchor step check in label_yard_lines as the only
    guard. That is the pre-v4 behaviour, no worse."""
    out, rms, second = _ladder_search(
        lines, max_index_gap=max_index_gap, max_rms_px=max_rms_px,
        min_margin=min_margin)
    _gate_ladder(rms, second, max_rms_px=max_rms_px, min_margin=min_margin,
                 n=len(lines))
    return out


def prune_to_yard_ladder(lines, *, max_outliers: int = 2,
                         max_index_gap: int = LADDER_MAX_INDEX_GAP,
                         max_rms_px: float = LADDER_MAX_RMS_PX,
                         min_margin: float = LADDER_MIN_MARGIN):
    """Largest subset of ``lines`` that forms a consistent yard ladder.

    Returns ``(kept_lines, indices, dropped)``. Real mosaics carry the odd
    spurious line -- a cable shadow, a bench edge, a graphic -- and one is
    enough to make the whole set inconsistent, so calibration should drop it
    rather than fail. Measured on play_001: 8 detected lines fit at best
    16.29 px; dropping the single stray at 1.9 degrees fits at 0.862 px.

    Subsets are tried smallest-drop-first, so a set that is already consistent
    is never pruned, and a real yard line is never discarded merely because
    discarding it also fits. Dropping is capped at ``max_outliers`` because
    every drop weakens the spacing evidence."""
    n = len(lines)
    for drop in range(0, max_outliers + 1):
        best = None
        for combo in itertools.combinations(range(n), drop):
            keep = [i for i in range(n) if i not in combo]
            if len(keep) < 4:
                continue
            sub = [lines[i] for i in keep]
            try:
                idx, rms, second = _ladder_search(
                    sub, max_index_gap=max_index_gap, max_rms_px=max_rms_px,
                    min_margin=min_margin)
                _gate_ladder(rms, second, max_rms_px=max_rms_px,
                             min_margin=min_margin, n=len(sub))
            except CalibrationError:
                continue
            if best is None or rms < best[1]:
                best = ((sub, idx, list(combo)), rms)
        if best is not None:
            return best[0]
    raise CalibrationError(
        f"endzone field fit: no subset of the {n} detected lines (dropping up "
        f"to {max_outliers}) forms a consistent yard ladder. The mosaic's line "
        "detections are too noisy to label; inspect the accumulated mosaic.")


def _dash_components(votes, *, vote_thresh, min_dash_px, max_dash_px,
                     max_dash_thick_px, min_dash_area):
    """Centroids and cross-field lengths of SHORT painted marks."""
    mask = (np.asarray(votes) >= vote_thresh).astype(np.uint8) * 255
    _n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    w = stats[1:, cv2.CC_STAT_WIDTH]
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]
    keep = ((w >= min_dash_px) & (w <= max_dash_px)
            & (h <= max_dash_thick_px) & (area >= min_dash_area))
    return cent[1:][keep], w[keep].astype(float)


def _upper_envelope_len(rows, lens, ref_row, bins: int = 5) -> float:
    """Full-mark length at ``ref_row``, from the UPPER envelope of ``lens``.

    A dash broken by the vote threshold contributes a FRAGMENT, so the mean
    length is biased far low (measured on play_001: mean 63 px against a true
    ~90 px, which alone would misidentify the rows by 48%). Marks also lengthen
    toward the camera, so the envelope is fitted against the row."""
    rows = np.asarray(rows, float)
    lens = np.asarray(lens, float)
    if len(rows) < bins * 2:
        return float(np.percentile(lens, 85))
    edges = np.quantile(rows, np.linspace(0.0, 1.0, bins + 1))
    xs, ys = [], []
    for i in range(bins):
        m = (rows >= edges[i]) & (rows <= edges[i + 1])
        if m.sum() >= 2:
            xs.append(rows[m].mean())
            ys.append(np.percentile(lens[m], 85))
    if len(xs) < 2:
        return float(np.percentile(lens, 85))
    slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(slope * ref_row + intercept)


@dataclasses.dataclass(frozen=True)
class HashRow:
    """One detected hash row: its image line plus the marks that formed it.

    ``line`` is homogeneous and normalised so ``line[:2]`` is a unit normal,
    making ``line @ (u, v, 1)`` a signed pixel distance. ``dashes`` are the mark
    centroids; the driver needs them as a depth ruler, since the marks recur
    every ``HASH_MARK_PITCH_M`` along the row."""
    line: np.ndarray
    dashes: np.ndarray


def _fit_line_tls(pts: np.ndarray) -> np.ndarray:
    """Total-least-squares line through points, as a normalised homogeneous line."""
    mean = pts.mean(axis=0)
    direction = np.linalg.svd(pts - mean)[2][0]
    normal = np.array([-direction[1], direction[0]], float)
    normal = normal / float(np.linalg.norm(normal))
    return np.array([normal[0], normal[1], -float(normal @ mean)], float)


def _ransac_hash_rows(cent, yard_dir, *, tol_px, min_dashes, max_rows=4,
                      min_span_px=300.0, max_align=0.5,
                      min_row_sep_px=100.0):
    """Sequential RANSAC for the down-field dash rows.

    Grouping the marks by their cross-field coordinate was measured
    insufficient: the rows TILT across the frame (measured 59 px of drift over
    1080 rows) and a mark broken by the vote threshold contributes a centroid
    displaced along the row's short axis, so a 1-D clustering fuses a row with
    unrelated paint beside it (measured on play_001: a 53-mark group spanning
    x 771..982, whose least-squares line then kept only 9 marks). RANSAC finds
    the row despite that contamination.

    Candidates are restricted to directions near-PERPENDICULAR to the yard
    lines (``max_align``), which is what a hash row is; without that the best
    consensus is a line running ALONG a yard line through its own marks."""
    cent = np.asarray(cent, float)
    remaining = np.ones(len(cent), bool)
    along = np.array([-yard_dir[1], yard_dir[0]], float)
    rows: list[HashRow] = []
    for _ in range(max_rows):
        pts = cent[remaining]
        if len(pts) < min_dashes:
            break
        homo = np.column_stack([pts, np.ones(len(pts))])
        best_line, best_count = None, 0
        for i in range(len(pts)):
            deltas = pts - pts[i]
            norms = np.linalg.norm(deltas, axis=1)
            ok = norms > 1e-6
            dirs = deltas[ok] / norms[ok, None]
            good = np.abs(dirs @ yard_dir) <= max_align
            for d in dirs[good]:
                normal = np.array([-d[1], d[0]])
                line = np.array([normal[0], normal[1], -float(normal @ pts[i])])
                count = int((np.abs(homo @ line) < tol_px).sum())
                if count > best_count:
                    best_line, best_count = line, count
        if best_line is None or best_count < min_dashes:
            break
        inl = np.abs(homo @ best_line) < tol_px
        sel = pts[inl]
        line = _fit_line_tls(sel)
        inl2 = np.abs(np.column_stack([sel, np.ones(len(sel))]) @ line) < tol_px
        sel = sel[inl2]
        idx = np.where(remaining)[0]
        chosen = idx[np.where(inl)[0]]
        remaining[chosen] = False
        if len(sel) < min_dashes or float(np.ptp(sel @ along)) < min_span_px:
            continue
        rows.append(HashRow(line=_fit_line_tls(sel), dashes=sel))

    # A mark broken by the vote threshold leaves fragment centroids displaced
    # along its own length, and those fragments line up into a PHANTOM row a
    # few tens of px beside the real one (measured on play_001: a 13-mark row
    # 24 px right of the true one, fitting at 1.27 px). The ratio check cannot
    # reject it -- pairing the far row with the phantom gives 9.7 against 9.5
    # for the true pair -- so near-duplicates are removed here, keeping
    # whichever carries more marks. Real hash rows are 2*HASH_OFFSET_M apart
    # and never this close.
    rows.sort(key=lambda r: -len(r.dashes))
    out: list[HashRow] = []
    for r in rows:
        q = r.dashes.mean(axis=0)
        if any(abs(float(k.line @ np.array([q[0], q[1], 1.0]))) < min_row_sep_px
               for k in out):
            continue
        out.append(r)
    return out


def detect_hash_columns(votes, yard_lines, *, vote_thresh: float = 0.5,
                        min_dash_px: int = 15, max_dash_px: int = 160,
                        max_dash_thick_px: int = 14, min_dash_area: int = 20,
                        min_dashes: int = 10, resid_tol_px: float = 3.5,
                        min_span_px: float = 300.0,
                        min_row_sep_px: float = 100.0,
                        ratio_tol: float = 0.35) -> list[HashRow]:
    """The two HASH ROWS of the field, as image lines, from the short marks.

    This is what makes the endzone homography observable. Yard lines are all
    mutually parallel, so they fix only a vanishing point and a 1-D map along
    the field -- 5 constraints for an 8-DOF homography. The hash rows run
    PERPENDICULAR to them at a known world Y (+-HASH_OFFSET_M), supplying the
    missing cross-field direction and scale.

    They cannot be found by orientation: a hash mark is painted parallel to the
    yard lines and lands in the same angular cluster (measured on play_001: 252
    of 255 raw Hough segments fell in the yard-line bins, and only 3 short
    stragglers elsewhere). They are separated by LENGTH, then found by RANSAC.

    The chosen pair is CHECKED, not assumed, against a quantity the search
    never used: the rows are ``2 * HASH_OFFSET_M`` apart and each mark is
    ``HASH_MARK_LEN_M`` long, so their pixel ratio must be about 9.25.
    Measured on play_001 the true pair gives 9.6; reading the pair as a hash
    row plus a sideline tick row instead would demand 24 px marks where 85-98
    px were measured, so the check separates them by a wide margin."""
    if len(yard_lines) < 1:
        raise CalibrationError(
            "endzone field fit: need at least one yard line to orient the "
            "hash-row search.")
    cent, lens = _dash_components(
        votes, vote_thresh=vote_thresh, min_dash_px=min_dash_px,
        max_dash_px=max_dash_px, max_dash_thick_px=max_dash_thick_px,
        min_dash_area=min_dash_area)
    if len(cent) < 2 * min_dashes:
        raise CalibrationError(
            f"endzone field fit: only {len(cent)} short painted marks found in "
            f"the mosaic (need >= {2 * min_dashes} to form two hash rows). "
            "Without the hash rows the yard lines are all parallel and the "
            "cross-field scale is unobservable - accumulate more frames, or "
            "lower vote_thresh.")

    nrm = _line_normal(yard_lines[0])
    yard_dir = np.array([-nrm[1], nrm[0]], float)
    rows = _ransac_hash_rows(cent, yard_dir, tol_px=resid_tol_px,
                             min_dashes=min_dashes, min_span_px=min_span_px,
                             min_row_sep_px=min_row_sep_px)
    if len(rows) < 2:
        raise CalibrationError(
            f"endzone field fit: found {len(rows)} hash row(s) in the mosaic, "
            "need 2. The cross-field scale is unobservable from yard lines "
            "alone (they are mutually parallel), so calibration cannot "
            "proceed - check the accumulated mosaic for hash marks.")

    expected = 2.0 * HASH_OFFSET_M / HASH_MARK_LEN_M
    best, best_err = None, float("inf")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ratio = _pair_ratio(rows[i], rows[j], cent, lens, yard_dir)
            err = abs(ratio - expected) / expected
            if err < best_err:
                best, best_err = (i, j), err
    i, j = best
    if best_err > ratio_tol:
        raise CalibrationError(
            f"endzone field fit: the best pair of candidate hash rows is "
            f"{best_err * 100:.0f}% off the expected separation-to-mark-length "
            f"ratio ({expected:.2f}) - they are probably not the two hash rows "
            "(a hash row paired with a sideline tick row is the usual "
            "confusion). Check the accumulated mosaic.")
    pair = [rows[i], rows[j]]
    pair.sort(key=lambda r: float((r.dashes @ yard_dir).mean()))
    return pair


def _pair_ratio(row_a: HashRow, row_b: HashRow, cent, lens, direction) -> float:
    """Separation / full-mark-length for a candidate pair of hash rows."""
    pts = np.vstack([row_a.dashes, row_b.dashes])
    q = pts.mean(axis=0)
    n_a = row_a.line[:2]
    p_on_a = q - float(row_a.line @ np.array([q[0], q[1], 1.0])) * n_a
    sep = abs(float(row_b.line @ np.array([p_on_a[0], p_on_a[1], 1.0])))
    along = np.array([-direction[1], direction[0]], float)
    mark = _upper_envelope_len(cent @ along, lens, float(q @ along))
    if mark <= 1e-6:
        return float("inf")
    return sep / mark


def yard_x_mapper(lines, indices, xs):
    """Callable mapping any image point to its continuous world X.

    The labelled lines give world X at a handful of rows; the hash marks lie
    BETWEEN them and need X too. Index and offset are related by the same 1-D
    projective map the ladder fitted, and world X is affine in the index, so
    inverting one and applying the other interpolates correctly under
    perspective (linear interpolation in pixels would not)."""
    ref_n = _line_normal(lines[0])
    offs = np.array([_offset(seg, ref_n) for seg in lines], float)
    idx = np.asarray(indices, float)
    _rms, sol = _fit_projective_1d(idx, offs)
    if sol is None:
        raise CalibrationError(
            "endzone field fit: could not invert the yard ladder to map "
            "hash-mark positions to world X.")
    a, b, c = sol
    slope, intercept = np.polyfit(idx, np.asarray(xs, float), 1)

    def to_x(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        o = pts @ ref_n
        denom = c * o - a
        if np.any(np.abs(denom) < 1e-9):
            raise CalibrationError(
                "endzone field fit: degenerate ladder inversion.")
        return slope * ((b - o) / denom) + intercept

    return to_x
