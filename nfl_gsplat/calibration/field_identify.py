"""Identify detected yard lines (assign absolute yardage) + emit correspondences.

Pure geometry. Strategy:
1. Order detected yard lines left→right by their mean image x.
2. Seed identity from a ``CalibHint`` (ref_frame/ref_x/yard/side/increasing)
   via ``seed_state_from_hint``; propagate to neighbours using the constant
   index spacing (adjacent detected lines are 5 yd apart).
3. In subsequent frames reuse ``prior`` by matching current lines to the
   previous lines by nearest image-x (lines move little frame-to-frame).
4. For each yardage-identified line, intersect with detected sidelines/hash rows
   and emit ``(landmark_name, uv)`` correspondences.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


def seed_state_from_hint(feats, hint) -> IdentityState:
    """Initial IdentityState for hint.ref_frame: snap ref_x to the nearest yard
    line, label it (side, yard), label the rest by 5-yd index spacing. ``increasing``
    = image direction yards grow: 'right' => +1 yard-line per +1 line index."""
    lines = sorted(feats.yard_lines, key=_line_x)
    if not lines:
        return IdentityState()
    xs = [_line_x(s) for s in lines]
    seed_idx = min(range(len(xs)), key=lambda i: abs(xs[i] - hint.ref_x))
    step_per_index = 1 if hint.increasing == "right" else -1
    out: dict[float, tuple[str, int]] = {}
    for i, s in enumerate(lines):
        side, yard = _yard_step(hint.side, hint.yard, step_per_index * (i - seed_idx))
        if side:
            out[_line_x(s)] = (side, yard)
    return IdentityState(line_yardage=out)


def identify_correspondences(
    feats: DetectedFeatures, prior: IdentityState | None,
) -> tuple[list[tuple[str, tuple[float, float]]], IdentityState]:
    """Propagate yard-line identity from ``prior`` to this frame's lines (nearest
    image-x) and emit [(landmark_name, (u,v))] at hash/sideline intersections.
    With no prior, returns ([], empty) — identity is seeded by a hint."""
    import numpy as np

    lines = sorted(feats.yard_lines, key=_line_x)
    if not lines or prior is None or not prior.line_yardage:
        return [], IdentityState()
    prior_xs = np.array(list(prior.line_yardage.keys()))
    prior_vals = list(prior.line_yardage.values())
    corrs: list[tuple[str, tuple[float, float]]] = []
    state_map: dict[float, tuple[str, int]] = {}
    for seg in lines:
        x = _line_x(seg)
        j = int(np.argmin(np.abs(prior_xs - x)))
        if abs(prior_xs[j] - x) > 60.0:
            continue
        side, yd = prior_vals[j]
        state_map[x] = (side, yd)
        for sl in feats.sidelines:
            pt = _seg_intersection(seg, sl)
            if pt is None:
                continue
            # Assumes the standard broadcast camera side (image-top = world +Y = 'left').
            # For a camera on the opposite sideline this is mirrored; resolved/validated at bring-up.
            lr = "left" if pt[1] < feats.image_size[1] / 2 else "right"
            corrs.append((landmark_name(side, yd, lr, "sideline"), pt))
        for hx, hy in feats.hashes:
            if abs(hx - x) < 25.0:
                # Assumes the standard broadcast camera side (image-top = world +Y = 'left').
                # For a camera on the opposite sideline this is mirrored; resolved/validated at bring-up.
                lr = "left" if hy < feats.image_size[1] / 2 else "right"
                corrs.append((landmark_name(side, yd, lr, "hash"), (float(hx), float(hy))))
    seen: set[str] = set()
    deduped = []
    for name, uv in corrs:
        if name not in seen:
            seen.add(name)
            deduped.append((name, uv))
    return deduped, IdentityState(line_yardage=state_map)
