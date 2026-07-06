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
    Unvoted lines are NOT filled here: linear world-X interpolation/
    extrapolation misassigns identities under perspective (measured on real
    footage: the away_40 line snapped to away_35 with votes only on the 20
    and 30). Unvoted-line completion is done downstream in ``fuse_frame``
    via an exact homography fit from the voted lines. Any base name that
    ends up assigned to more than one line is ambiguous and every line
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
    voted = [(k, ident[order[k]]) for k in range(len(order)) if order[k] in ident]
    Xw = [_yardline_x_m(b) for (_k, b) in voted]
    if len(voted) >= 2:
        inc = all(b > a for a, b in zip(Xw, Xw[1:]))
        dec = all(b < a for a, b in zip(Xw, Xw[1:]))
        if not (inc or dec):
            return {}                                 # inconsistent left-right ordering

    return _drop_duplicate_bases(ident)


def _drop_duplicate_bases(ident: dict[int, str]) -> dict[int, str]:
    """A base name assigned to more than one line is ambiguous: never guess
    which is correct -> drop every line carrying that base."""
    base_counts = Counter(ident.values())
    dupes = {b for b, c in base_counts.items() if c > 1}
    if dupes:
        return {i: b for i, b in ident.items() if b not in dupes}
    return ident


def _complete_identities_via_homography(ident, yard_lines, rows):
    """Identify unvoted lines by mapping their hash-row intersections through
    the homography implied by the voted lines. Projectively exact — linear
    interpolation misassigns identities under perspective (measured: the 40
    line snapped to away_35 on real footage). Needs >=2 voted lines and both
    hash rows (4+ point correspondences => exact plane homography)."""
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M

    if len(ident) < 2 or len(rows) != 2:
        return ident
    world, img = [], []
    for i, base in ident.items():
        X = _yardline_x_m(base)
        for Y, row in ((+HASH_OFFSET_M, rows[0]), (-HASH_OFFSET_M, rows[1])):
            uv = _intersect(yard_lines[i], row)
            if uv is not None:
                world.append([X, Y]); img.append(uv)
    if len(world) < 4:
        return ident
    Hm, _ = cv2.findHomography(np.array(world, np.float64), np.array(img, np.float64), 0)
    if Hm is None:
        return ident
    Hinv = np.linalg.inv(Hm)
    out = dict(ident)
    for i, seg in enumerate(yard_lines):
        if i in out:
            continue
        Xs = []
        for row in rows:
            uv = _intersect(seg, row)
            if uv is None:
                continue
            w = Hinv @ np.array([uv[0], uv[1], 1.0])
            if abs(w[2]) > 1e-9:
                Xs.append(w[0] / w[2])
        if not Xs:
            continue
        X_est = float(np.mean(Xs))
        try:
            out[i] = yardline_name_from_x_m(X_est, tol_m=_SNAP_TOL_M)
        except ValueError:
            continue
    return out


def correspondences_from_identities(ident, yard_lines, rows, *, gate: bool = True):
    """Identified line x hash-row intersections -> plane-consistent correspondences.

    Requires both hash rows (upper row = +Y = ``*_left_hash``, the convention
    validated on real footage 2026-06); with fewer than two rows we cannot
    tell upper from lower, so returns []. With ``gate`` (default), filters the
    raw intersections through :func:`_ransac_consistent` so only points on a
    single plane homography survive. :func:`fuse_frame` passes ``gate=False``
    and gates ONCE over intersections + number keypoints together — gating
    twice rejected valid points on real footage (a 6-corr frame shrank to 5,
    below the solver minimum).
    """
    if len(rows) != 2:
        return []
    corrs: list[tuple[str, tuple[float, float]]] = []
    for lr, row in (("left", rows[0]), ("right", rows[1])):
        for i, base in ident.items():
            seg = yard_lines[i]
            uv = _intersect(seg, row)
            if uv is not None:
                corrs.append((f"{base}_{lr}_hash", uv))
    return _ransac_consistent(corrs) if gate else corrs


def predict_identities(yard_lines, rows, H_plane, *, tol_m: float = 1.0):
    """Identify classical lines by mapping their hash-row intersections through
    a KNOWN plane homography (from a solved neighboring frame).

    ``H_plane`` maps world plane points ``(X, Y, 1)`` (Z=0) to image points
    ``(u, v, w)`` up to scale — exactly ``K @ [R[:,0], R[:,1], t]`` for a
    solved frame's intrinsics/pose. Chain-propagated frame to frame in the
    calling sweep, so ``H_plane`` is at most a few frames stale (slow pan
    footage: adjacent-frame homographies differ by only a few px). Returns
    ``{line_idx: base_name}``; duplicate base names are dropped (never guess
    which line is correct).
    """
    import numpy as np

    if len(rows) != 2 or H_plane is None:
        return {}
    Hinv = np.linalg.inv(H_plane)
    ident: dict[int, str] = {}
    for i, seg in enumerate(yard_lines):
        Xs = []
        for row in rows:
            uv = _intersect(seg, row)
            if uv is None:
                continue
            w = Hinv @ np.array([uv[0], uv[1], 1.0])
            if abs(w[2]) > 1e-9:
                Xs.append(w[0] / w[2])
        if not Xs:
            continue
        try:
            ident[i] = yardline_name_from_x_m(float(np.mean(Xs)), tol_m=tol_m)
        except ValueError:
            continue
    return _drop_duplicate_bases(ident)


def fuse_frame(yard_lines, hashes, model_kps, *, territory, image_size,
               max_assign_px: float = 60.0, min_margin: float = 2.0):
    """One frame -> labeled correspondences [(landmark_name, (u, v))]."""
    rows = fit_hash_rows(hashes, image_width=image_size[0])
    ident = identify_lines(yard_lines, model_kps, territory=territory,
                           max_assign_px=max_assign_px, min_margin=min_margin)
    if not ident:
        return []
    corrs: list[tuple[str, tuple[float, float]]] = []

    # rows are sorted upper-first; upper row = +Y = *_left_hash (convention
    # validated on real footage 2026-06). With only one row we cannot tell
    # upper from lower -> skip hash intersections (fail toward less, correct).
    if len(rows) == 2:
        ident = _complete_identities_via_homography(ident, yard_lines, rows)
        ident = _drop_duplicate_bases(ident)
        # gate=False: the single RANSAC gate below covers intersections AND
        # number keypoints together (double-gating starved real frames)
        corrs.extend(correspondences_from_identities(ident, yard_lines, rows, gate=False))

    for (cls, u, v, _conf) in model_kps:
        name = to_nfl_name(cls, territory)
        if name is not None and name.endswith("_number"):
            corrs.append((name, (float(u), float(v))))
    return _ransac_consistent(corrs)


def _ransac_consistent(corrs, reproj_px: float = 8.0):
    """Keep only correspondences consistent with a single plane homography.

    Coarse model numbers and occasional misidentified intersections get
    dropped here so the downstream PnP (which has no outlier rejection and
    gates on all-point RMS) sees only clean points.
    """
    if len(corrs) < 4:
        return []
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    world = np.array([NFL_LANDMARKS[n][:2] for (n, _uv) in corrs], np.float64)
    uv = np.array([p for (_n, p) in corrs], np.float64)
    Hm, mask = cv2.findHomography(world, uv, cv2.RANSAC, reproj_px)
    if Hm is None:
        return []
    keep = mask.ravel().astype(bool)
    return [c for c, k in zip(corrs, keep) if k]


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
