"""Identify detected yard lines (assign absolute yardage) + emit correspondences.

Pure geometry. Strategy:
1. Order detected yard lines left→right by their mean image x.
2. If OCR numbers are present, snap each to the nearest yard line and seed that
   line's yardage; propagate to neighbours using the constant index spacing
   (adjacent detected lines are 5 yd apart). Direction (toward home vs away) is
   resolved from the order of two seeded numbers; a single number defaults the
   higher-x direction toward the 50 then home (documented; the bundle-adjusted
   PnP + RMS gate reject a wrong guess, and two numbers remove the ambiguity).
3. If no numbers this frame, reuse ``prior`` by matching current lines to the
   previous lines by nearest image-x (lines move little frame-to-frame).
4. For each yardage-identified line, intersect with detected sidelines/hash rows
   and emit ``(landmark_name, uv)`` correspondences.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nfl_gsplat.calibration.field_features import DetectedFeatures, landmark_name


@dataclass(frozen=True)
class IdentityState:
    line_yardage: dict[float, tuple[str, int]] = field(default_factory=dict)


def _line_x(seg) -> float:
    return 0.5 * (seg.p0[0] + seg.p1[0])


def _seg_intersection(a, b) -> tuple[float, float] | None:
    (x1, y1), (x2, y2) = a.p0, a.p1
    (x3, y3), (x4, y4) = b.p0, b.p1
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return (px, py)


def _assign_from_numbers(lines_sorted, numbers) -> dict[int, tuple[str, int]]:
    if not numbers:
        return {}
    line_xs = np.array([_line_x(s) for s in lines_sorted])
    seeds: dict[int, int] = {}
    for num in numbers:
        idx = int(np.argmin(np.abs(line_xs - num.center[0])))
        seeds[idx] = num.value
    if len(seeds) >= 2:
        items = sorted(seeds.items())
        (i0, y0), (i1, y1) = items[0], items[-1]
        inc = (y1 - y0) / max(i1 - i0, 1)
    else:
        inc = 5.0
    i_seed, y_seed = next(iter(seeds.items()))
    out: dict[int, tuple[str, int]] = {}
    for i in range(len(lines_sorted)):
        yd_signed = y_seed + inc * (i - i_seed)
        v = int(round(yd_signed))
        if v == 50:
            out[i] = ("mid", 50)
        elif 10 <= v <= 45:
            out[i] = ("away", v) if inc > 0 else ("home", v)
        elif v > 50:
            folded = 100 - v
            if folded == 50 or folded in range(5, 50, 5):
                out[i] = ("home", folded)
    return out


def identify_correspondences(
    feats: DetectedFeatures, prior: IdentityState | None,
) -> tuple[list[tuple[str, tuple[float, float]]], IdentityState]:
    lines = sorted(feats.yard_lines, key=_line_x)
    if not lines:
        return [], IdentityState()

    idx_yardage = _assign_from_numbers(lines, feats.numbers)

    if not idx_yardage and prior is not None and prior.line_yardage:
        prior_xs = np.array(list(prior.line_yardage.keys()))
        prior_vals = list(prior.line_yardage.values())
        for i, seg in enumerate(lines):
            j = int(np.argmin(np.abs(prior_xs - _line_x(seg))))
            if abs(prior_xs[j] - _line_x(seg)) < 60.0:
                idx_yardage[i] = prior_vals[j]

    corrs: list[tuple[str, tuple[float, float]]] = []
    state_map: dict[float, tuple[str, int]] = {}
    for i, seg in enumerate(lines):
        if i not in idx_yardage:
            continue
        side, yd = idx_yardage[i]
        state_map[_line_x(seg)] = (side, yd)
        for sl in feats.sidelines:
            pt = _seg_intersection(seg, sl)
            if pt is None:
                continue
            lr = "left" if pt[1] < feats.image_size[1] / 2 else "right"
            corrs.append((landmark_name(side, yd, lr, "sideline"), pt))
        for hx, hy in feats.hashes:
            if abs(hx - _line_x(seg)) < 25.0:
                lr = "left" if hy < feats.image_size[1] / 2 else "right"
                corrs.append((landmark_name(side, yd, lr, "hash"), (float(hx), float(hy))))

    seen: set[str] = set()
    deduped: list[tuple[str, tuple[float, float]]] = []
    for name, uv in corrs:
        if name not in seen:
            seen.add(name)
            deduped.append((name, uv))
    return deduped, IdentityState(line_yardage=state_map)
