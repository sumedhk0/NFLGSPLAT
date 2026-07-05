"""Fuse pretrained-model identity with classical geometry.

Model keypoints are coarse (±3-30 px) but carry yard identity; classical
slanted lines are precise but anonymous. Each model keypoint votes for the
classical line nearest in x at the keypoint's row; identified lines are
intersected with the RANSAC hash rows to produce precise labeled
correspondences. Ambiguity -> drop (frame gap), never guess.
"""
from __future__ import annotations

from collections import Counter

from nfl_gsplat.calibration.field_identify import fit_hash_rows, line_x_at
from nfl_gsplat.calibration.field_landmarks import (
    _yardline_x_m, yardline_name_from_x_m, YARD_LINE_SPACING_M,
)
from nfl_gsplat.calibration.roboflow_kps import to_nfl_name, yard_base

_SNAP_TOL_M = 0.35 * YARD_LINE_SPACING_M     # accept interpolated X within ~1.6 m


def identify_lines(yard_lines, model_kps, *, territory,
                   max_assign_px: float = 60.0, min_margin: float = 2.0):
    """Vote yard identity onto classical lines. Returns {line_idx: base_name}.

    Gates (fail toward fewer, correct lines):
      - a keypoint only votes if its nearest line is within ``max_assign_px``
        AND ``min_margin``x closer than the runner-up;
      - conflicting votes on one line -> per-line majority; a tie for the top
        vote count drops that line only (the frame survives on its other
        lines);
      - identified world-X must be strictly monotone in image x, else {}.
    Unvoted lines strictly between the leftmost and rightmost voted image-x
    get identity by world-X interpolation snapped via
    ``yardline_name_from_x_m`` (round-trip-safe); lines outside that range
    are left unidentified rather than clamped-extrapolated. Any base name
    that ends up assigned to more than one line is ambiguous and every line
    carrying it is dropped.
    """
    if not yard_lines or not model_kps:
        return {}
    votes: dict[int, list[str]] = {}
    for (cls, u, v, _conf) in model_kps:
        base = yard_base(cls, territory)
        if base is None:
            continue
        dists = sorted((abs(line_x_at(s, v) - u), i)
                       for i, s in enumerate(yard_lines))
        d0, i0 = dists[0]
        if d0 > max_assign_px:
            continue
        if len(dists) > 1 and dists[1][0] < min_margin * d0:
            continue                                  # ambiguous between two lines
        votes.setdefault(i0, []).append(base)

    ident: dict[int, str] = {}
    for i, bases in votes.items():
        counts = Counter(bases)
        top_count = max(counts.values())
        top_bases = [b for b, c in counts.items() if c == top_count]
        if len(top_bases) != 1:
            continue                                  # tie: drop this line only
        ident[i] = top_bases[0]
    if not ident:
        return {}

    # order all lines by x at a common row; voted world-X must be monotone
    ref_y = sum(p[1] for s in yard_lines for p in (s.p0, s.p1)) / (2 * len(yard_lines))
    order = sorted(range(len(yard_lines)), key=lambda i: line_x_at(yard_lines[i], ref_y))
    xs_img = [line_x_at(yard_lines[i], ref_y) for i in order]
    voted = [(k, ident[order[k]]) for k in range(len(order)) if order[k] in ident]
    Xw = [_yardline_x_m(b) for (_k, b) in voted]
    if len(voted) >= 2:
        inc = all(b > a for a, b in zip(Xw, Xw[1:]))
        dec = all(b < a for a, b in zip(Xw, Xw[1:]))
        if not (inc or dec):
            return {}                                 # inconsistent left-right ordering

    # fill unvoted lines by piecewise-linear world-X interpolation + snap,
    # but only strictly between the voted image-x extremes -- np.interp
    # clamps outside that range, which would assign an out-of-range line
    # the same identity as the nearest endpoint (a duplicate base).
    if len(voted) >= 2:
        import numpy as np
        vk = [k for (k, _b) in voted]
        vX = np.array(Xw, float)
        vx = np.array([xs_img[k] for k in vk], float)
        x_lo, x_hi = float(min(vx)), float(max(vx))
        if vx[0] > vx[-1]:
            vx, vX = vx[::-1], vX[::-1]
        for k in range(len(order)):
            if order[k] in ident:
                continue
            xk = xs_img[k]
            if xk < x_lo or xk > x_hi:
                continue                              # outside voted range: no fill
            X_est = float(np.interp(xk, vx, vX))
            try:
                name = yardline_name_from_x_m(X_est, tol_m=_SNAP_TOL_M)
            except ValueError:
                continue                              # off-grid: leave unidentified
            if abs(_yardline_x_m(name) - X_est) <= _SNAP_TOL_M:
                ident[order[k]] = name

    # a base assigned to more than one line is ambiguous: never guess which
    # is correct -> drop every line carrying that base.
    base_counts = Counter(ident.values())
    dupes = {b for b, c in base_counts.items() if c > 1}
    if dupes:
        ident = {i: b for i, b in ident.items() if b not in dupes}
    return ident


def fuse_frame(yard_lines, hashes, model_kps, *, territory, image_size,
               max_assign_px: float = 60.0, min_margin: float = 2.0):
    """One frame -> labeled correspondences [(landmark_name, (u, v))]."""
    ident = identify_lines(yard_lines, model_kps, territory=territory,
                           max_assign_px=max_assign_px, min_margin=min_margin)
    if not ident:
        return []
    corrs: list[tuple[str, tuple[float, float]]] = []

    rows = fit_hash_rows(hashes, image_width=image_size[0])
    # rows are sorted upper-first; upper row = +Y = *_left_hash (convention
    # validated on real footage 2026-06). With only one row we cannot tell
    # upper from lower -> skip hash intersections (fail toward less, correct).
    if len(rows) == 2:
        for lr, row in (("left", rows[0]), ("right", rows[1])):
            for i, base in ident.items():
                seg = yard_lines[i]
                uv = _intersect(seg, row)
                if uv is not None:
                    corrs.append((f"{base}_{lr}_hash", uv))

    for (cls, u, v, _conf) in model_kps:
        name = to_nfl_name(cls, territory)
        if name is not None and name.endswith("_number"):
            corrs.append((name, (float(u), float(v))))
    return corrs


def _intersect(seg_a, seg_b):
    """Intersection of two YardLineSeg treated as infinite lines; None if parallel."""
    (x1, y1), (x2, y2) = seg_a.p0, seg_a.p1
    (x3, y3), (x4, y4) = seg_b.p0, seg_b.p1
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return (float((a * (x3 - x4) - (x1 - x2) * b) / d),
            float((a * (y3 - y4) - (y1 - y2) * b) / d))
