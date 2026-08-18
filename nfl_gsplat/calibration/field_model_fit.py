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


def _largest_parallel_cluster(lines: list[YardLineSeg],
                              align_tol: float = 0.9) -> list[YardLineSeg]:
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
    to tolerate the perspective fan of a real pencil of parallel field lines."""
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
                             merge_tol_px: float = 12.0) -> list[YardLineSeg]:
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
    merged = _largest_parallel_cluster(merged)
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
                f"({local_gap:.1f} px) — click closer to a line.")
        idxs.append(k)
    if idxs[0] == idxs[1]:
        raise CalibrationError(
            "endzone field fit: both anchors matched the SAME line; they must "
            "name two distinct yard lines.")

    r0, r1 = rank_of[idxs[0]], rank_of[idxs[1]]
    # The anchors MUST be the outermost detected lines. The step check below
    # only constrains the span BETWEEN them, so anchoring inner lines leaves a
    # missing/spurious/over-merged line outside that span silently mislabelled
    # by a full spacing. Spanning every line makes the check global — and makes
    # a separate over-merge guard unnecessary, since a merge changes the count.
    if abs(r1 - r0) != len(lines) - 1:
        raise CalibrationError(
            f"endzone field fit: the two anchors span {abs(r1 - r0) + 1} of "
            f"{len(lines)} detected lines. Anchor the OUTERMOST two lines, so "
            "the spacing check covers every line; otherwise a missing or "
            "spurious line outside the span is silently mislabelled.")
    x0, x1 = float(anchors[0][1]), float(anchors[1][1])
    step = (x1 - x0) / (r1 - r0)
    if abs(abs(step) - YARD_LINE_SPACING_M) > 0.05:
        raise CalibrationError(
            f"endzone field fit: the two anchors imply a step of {step:.3f} m "
            f"per detected line, not +-{YARD_LINE_SPACING_M:.3f} m — a yard "
            "line is missing, spurious, or two lines were merged into one. "
            f"Detected {len(lines)} lines at offsets "
            f"{np.round(offs[order], 1).tolist()} px (anchors ranked {r0} "
            f"and {r1}) — a wrong count there is the usual cause.")

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
