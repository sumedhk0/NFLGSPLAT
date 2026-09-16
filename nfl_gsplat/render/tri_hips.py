"""Ground points from the two cameras' hip keypoints, triangulated.

WHY. A body both cameras see stands, in the timeline, on the sideline's foot point slid along the
sideline ray to the endzone's foot point (render.depth_snap). Measured on play 1 (2026-09-16, 1541
two-view live frames whose id has hip keypoints in both cameras, 90 % of such frames): the hip centre
triangulated from the two hip rays reprojects onto the hips at 5.3 px (sideline) / 8.4 px (endzone)
and sits at 0.77 m (p10 0.60, p90 0.92 -- crouching to standing), while the timeline's point is 0.20 m
from it at the median and 0.58 at p90, 0.33-0.38 m one way along the sideline's depth on the
defence's side of the line. On the render that is a man drawn a stride from where he is.

WHAT. For every (frame, id) with a confident hip pair in both cameras, the closest point between the
two hip-centre rays. Kept only when the rays pass within ``max_gap_m`` of each other and the point
sits at a hip's height (``z_min``..``z_max``): a mispaired id (play 1's 9: its endzone rows are another
man) fails both. The point's ground xy replaces the timeline's ground point on those frames; the
foot-point machinery stands everywhere else.
"""
from __future__ import annotations

import numpy as np

HIP_CONF: float = 0.3
MAX_RAY_GAP_M: float = 0.5
HIP_Z_MIN_M: float = 0.5
HIP_Z_MAX_M: float = 1.4


def closest_point(c1, d1, c2, d2):
    """``(midpoint, gap)`` of the closest points on two lines ``c + s d``; None when parallel."""
    c1, d1, c2, d2 = (np.asarray(v, float) for v in (c1, d1, c2, d2))
    w = c1 - c2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-12:
        return None
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1, p2 = c1 + s * d1, c2 + t * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


def pixel_ray(K, R, t, uv):
    """``(centre, unit direction)`` in world coordinates of the ray through pixel ``uv``."""
    K, R, t = (np.asarray(v, float) for v in (K, R, t))
    centre = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]))
    return centre, d / np.linalg.norm(d)


def hip_centres(kdf, *, min_conf: float = HIP_CONF) -> dict:
    """``{(cam, frame, pid): uv}`` -- the mean of a confident COCO hip pair (11, 12) per camera."""
    k = kdf[(kdf["joint"].isin([11, 12])) & (kdf["conf"] >= min_conf)]
    out = {}
    for (cam, f, pid), g in k.groupby(["cam", "frame", "global_player_id"]):
        if len(g) == 2:
            out[(str(cam), int(f), int(pid))] = g[["x", "y"]].to_numpy(float).mean(axis=0)
    return out


def triangulated_hips(kdf, tracks, *, frame_shift=None, min_conf: float = HIP_CONF,
                      max_gap_m: float = MAX_RAY_GAP_M, z_min: float = HIP_Z_MIN_M, z_max: float = HIP_Z_MAX_M,
                      cams=("sideline", "endzone")) -> dict:
    """``{(frame, pid): (xy, z, gap)}`` in the first camera's frame numbering for every id with a hip
    pair in both ``cams`` on a frame both cameras have a pose for, passing the gap and height gates.
    ``frame_shift`` ``{cam: n}``: that camera's keypoint rows were moved by n from its clip frames."""
    a, b = cams
    if a not in tracks or b not in tracks:
        return {}
    hips = hip_centres(kdf, min_conf=min_conf)
    shift = frame_shift or {}
    out = {}
    for (cam, f, pid), uv in hips.items():
        if cam != a:
            continue
        uv2 = hips.get((b, f, pid))
        if uv2 is None:
            continue
        fa, fb = f + int(shift.get(a, 0)), f + int(shift.get(b, 0))
        ta, tb = tracks[a], tracks[b]
        if not (0 <= fa < len(ta.conf) and 0 <= fb < len(tb.conf)) or ta.conf[fa] <= 0 or tb.conf[fb] <= 0:
            continue
        c1, d1 = pixel_ray(ta.K[fa], ta.R[fa], ta.t[fa], uv)
        c2, d2 = pixel_ray(tb.K[fb], tb.R[fb], tb.t[fb], uv2)
        res = closest_point(c1, d1, c2, d2)
        if res is None:
            continue
        X, gap = res
        if gap > max_gap_m or not (z_min <= X[2] <= z_max):
            continue
        out[(int(f), int(pid))] = (X[:2].copy(), float(X[2]), gap)
    return out


def place_on_triangulated_hips(ground: dict, tri: dict) -> tuple[dict, list]:
    """``(ground, moves)``: ``ground`` (frame -> {pid: xy}) with every (frame, pid) present in ``tri``
    set to the triangulated ground xy; ``moves`` lists the metres each moved. Frames the timeline does
    not draw are not added."""
    out = {f: dict(d) for f, d in ground.items()}
    moves = []
    for (f, pid), (xy, _z, _gap) in tri.items():
        if f in out and pid in out[f]:
            moves.append(float(np.linalg.norm(np.asarray(out[f][pid], float) - xy)))
            out[f][pid] = np.asarray(xy, float)
    return out, moves


ANCHOR_WINDOW: int = 15
ANCHOR_MIN_SUPPORT: int = 3


def anchor_ground_to_tri(ground: dict, tri: dict, *, window: int = ANCHOR_WINDOW,
                         min_support: int = ANCHOR_MIN_SUPPORT) -> tuple[dict, list]:
    """``(ground, shifts)``: every drawn (frame, pid) moved by the median of (triangulated - drawn)
    over the id's triangulated frames within ``window`` frames, needing ``min_support`` of them.

    Why an offset and not the points: swapping the triangulated hip in on the frames that have one
    and leaving the refit's transl elsewhere put two placement sources 0.2-0.6 m apart next to each
    other frame by frame (play 1, 2026-09-16: root jitter p90 0.0116 -> 0.0217, live hops 4 -> 7,
    the endzone offset unmoved at 44 px). The correction a body needs varies slowly, so it is
    estimated as a windowed median and applied to every frame, seen or not -- the ankle anchor
    (play_timeline.anchor_boxes_to_ankles) does the same for box points."""
    by_pid: dict[int, dict] = {}
    for (f, pid), (xy, _z, _gap) in tri.items():
        if f in ground and pid in ground[f]:
            by_pid.setdefault(int(pid), {})[int(f)] = np.asarray(xy, float) - np.asarray(ground[f][pid], float)
    out = {f: dict(d) for f, d in ground.items()}
    shifts = []
    for f, d in ground.items():
        for pid, xy in d.items():
            offs = by_pid.get(int(pid))
            if not offs:
                continue
            near = [v for g, v in offs.items() if abs(g - f) <= window]
            if len(near) < min_support:
                continue
            delta = np.median(np.stack(near), axis=0)
            out[f][pid] = np.asarray(xy, float) + delta
            shifts.append(float(np.linalg.norm(delta)))
    return out, shifts
