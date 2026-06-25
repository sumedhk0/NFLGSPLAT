"""Identify detected yard lines (assign absolute yardage) + emit correspondences.

Pure geometry. Strategy:
1. Order detected yard lines left→right by their x at image mid-height
   (diagonal-safe; broadcast yard lines slant, so mean-x is unreliable).
2. Seed identity from a ``CalibHint`` (ref_frame/ref_x/yard/side/increasing)
   via ``seed_state_from_hint``; propagate to neighbours using the constant
   index spacing (adjacent detected lines are 5 yd apart).
3. In subsequent frames reuse ``prior`` by matching current lines to the
   previous lines by nearest image-x (lines move little frame-to-frame).
4. For each yardage-identified line, intersect with the two fitted hash-row lines
   (and sidelines when present) and emit ``(landmark_name, uv)`` correspondences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class IdentityState:
    line_yardage: dict[float, tuple[str, int]] = field(default_factory=dict)
    homography: "np.ndarray | None" = None
    anchor_label: "tuple[str, int] | None" = None
    anchor_x: "float | None" = None
    direction: int = 0


def line_x_at(seg, y: float) -> float:
    """Image-x of a (near-vertical) line segment at height ``y``.

    Yard lines are near-vertical but slanted in broadcast views, so x varies
    with y; their mean-x is unreliable for ordering/matching. Interpolates along
    the segment's direction. Degenerates to the mean-x for a perfectly
    horizontal segment (|dy| ~ 0), which shouldn't occur for yard lines."""
    (x0, y0), (x1, y1) = seg.p0, seg.p1
    dy = y1 - y0
    if abs(dy) < 1e-6:
        return 0.5 * (x0 + x1)
    t = (float(y) - y0) / dy
    return x0 + t * (x1 - x0)


def _merge_lines(lines, tol: float, ref_y: float):
    """Merge yard-line segments whose x at ``ref_y`` are within ``tol`` (the same
    physical line detected as multiple segments). Returns lines sorted by x@ref_y,
    one representative per cluster (the one spanning the largest y-range)."""
    items = sorted(lines, key=lambda s: line_x_at(s, ref_y))
    merged = []
    for seg in items:
        x = line_x_at(seg, ref_y)
        if merged and abs(line_x_at(merged[-1], ref_y) - x) <= tol:
            prev = merged[-1]
            if abs(seg.p1[1] - seg.p0[1]) > abs(prev.p1[1] - prev.p0[1]):
                merged[-1] = seg
        else:
            merged.append(seg)
    return merged


def _seg_intersection(a, b) -> tuple[float, float] | None:
    (x1, y1), (x2, y2) = a.p0, a.p1
    (x3, y3), (x4, y4) = b.p0, b.p1
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return (px, py)


def _yard_step(side: str, yard: int, step: int) -> tuple[str, int]:
    """Move ``step`` yard-LINES (x5 yd) from (side, yard) toward the home goal,
    folding across midfield. ``step`` may be negative. Returns ("",0) if off-field.

    Field position in yard-line units: away goal=0, away_5=1 .. away_45=9,
    mid_50=10, home_45=11 .. home_5=19, home goal=20.
    """
    if side == "mid":
        pos = 10
    elif side == "away":
        pos = yard // 5
    else:  # home
        pos = 20 - yard // 5
    pos += step
    if pos < 1 or pos > 19:
        return ("", 0)
    if pos == 10:
        return ("mid", 50)
    if pos < 10:
        return ("away", pos * 5)
    return ("home", (20 - pos) * 5)


def _yardline_name(side: str, yard: int) -> str:
    return "mid_50" if side == "mid" else f"{side}_{yard}"


def seed_state_from_hint(feats, hint) -> IdentityState:
    """Ref-frame seed: record the hint anchor (side/yard + image-x) and direction.
    The actual labeling happens in identify_correspondences (consensus)."""
    from nfl_gsplat.calibration.field_features import yardline_label
    side, yard = yardline_label(hint.side, hint.yard)
    direction = 1 if hint.increasing == "right" else -1
    return IdentityState(anchor_label=(side, yard), anchor_x=float(hint.ref_x),
                         direction=direction)


def _ransac_line(pts, *, inlier_px: float, iters: int, rng):
    """Best-fit line over 2D points by RANSAC. Returns (inlier_mask, (a, b)) for
    y = a*x + b, or (None, None) if degenerate. Hash rows are near-horizontal so
    y = a*x + b is well-conditioned."""
    import numpy as np
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    best_mask, best_count = None, -1
    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        x0, y0 = pts[i]; x1, y1 = pts[j]
        if abs(x1 - x0) < 1e-6:
            continue
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        resid = np.abs(pts[:, 1] - (a * pts[:, 0] + b))
        mask = resid <= inlier_px
        if mask.sum() > best_count:
            best_count, best_mask = int(mask.sum()), mask
    if best_mask is None:
        return None, None
    xin, yin = pts[best_mask, 0], pts[best_mask, 1]
    A = np.vstack([xin, np.ones_like(xin)]).T
    a, b = np.linalg.lstsq(A, yin, rcond=None)[0]
    return best_mask, (float(a), float(b))


def fit_hash_rows(hashes, *, image_width: int, inlier_px: float = 6.0,
                  min_inliers: int = 6, iters: int = 200):
    """Fit up to two hash-ROW lines from raw tick points via RANSAC, returning
    each as a width-spanning ``YardLineSeg``. Averages out the dense 1-yard ticks
    and noise. Returns [] / [one] / [two] sorted by row height (upper first)."""
    import numpy as np

    from nfl_gsplat.calibration.field_features import YardLineSeg
    pts = list(hashes)
    if len(pts) < min_inliers:
        return []
    rng = np.random.default_rng(12345)
    rows = []
    remaining = np.asarray(pts, dtype=np.float64)
    for _ in range(2):
        if len(remaining) < min_inliers:
            break
        mask, line = _ransac_line(remaining, inlier_px=inlier_px, iters=iters, rng=rng)
        if mask is None or int(mask.sum()) < min_inliers:
            break
        a, b = line
        rows.append(YardLineSeg((0.0, b), (float(image_width), a * image_width + b)))
        remaining = remaining[~mask]
    rows.sort(key=lambda r: 0.5 * (r.p0[1] + r.p1[1]))
    return rows


def identify_correspondences(feats, prior):
    """Label this frame's yard lines by homography consensus (anchored by the hint
    at the ref frame, by prior.homography afterwards) and emit hash correspondences.
    Returns (correspondences, IdentityState carrying the homography)."""
    import cv2
    import numpy as np

    from nfl_gsplat.calibration.field_homography import label_lines_by_consensus
    from nfl_gsplat.calibration.field_landmarks import _yardline_x_m

    mid = feats.image_size[1] / 2.0
    lines = _merge_lines(feats.yard_lines, tol=25.0, ref_y=mid)
    rows = fit_hash_rows(feats.hashes, image_width=feats.image_size[0])
    if (len(lines) < 2 or len(rows) < 2 or prior is None
            or prior.anchor_label is None):
        return [], IdentityState()

    side, yard = prior.anchor_label
    anchor_world_x = _yardline_x_m(_yardline_name(side, yard))
    if prior.homography is not None:
        p = cv2.perspectiveTransform(
            np.array([[[anchor_world_x, 0.0]]], np.float64), prior.homography).reshape(2)
        pred_x = float(p[0])
    else:
        pred_x = float(prior.anchor_x)
    anchor_idx = min(range(len(lines)), key=lambda i: abs(line_x_at(lines[i], mid) - pred_x))

    res = label_lines_by_consensus(
        lines, rows, anchor_idx=anchor_idx, anchor_world_x=anchor_world_x,
        anchor_side=side, anchor_yard=yard, direction=prior.direction or 1,
        image_size=feats.image_size)
    if res.homography is None or res.inlier_count < 2:
        return [], IdentityState()
    new_anchor_x = line_x_at(lines[anchor_idx], mid)
    state = IdentityState(homography=res.homography, anchor_label=(side, yard),
                          anchor_x=new_anchor_x, direction=prior.direction or 1)
    return res.correspondences, state
