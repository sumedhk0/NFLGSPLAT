"""Hip centres triangulated from both cameras' hip keypoints -- a placement RULER.

The timeline places a paired body on the sideline's foot point slid along the sideline ray to the
endzone's foot point (render.depth_snap). This module triangulates the hip centre from the two hip
rays instead, gated by ray gap and hip height, so the placement can be scored against a point that
owes nothing to foot points: on play 1's live play (2026-09-16, 1883 two-view frames with a hip
pair in both cameras, all passing the gates) the rays meet within 0.06 m at the median, the point
sits at 0.84 m (p10 0.64, p90 0.96) and reprojects at 2.6 / 4.1 px, and the timeline's placement
is 0.04 m (x) / 0.02 m (y) from it at the median, 0.08 / 0.05 at p90. The paired placement is right;
nothing here needs to move a body.

History, so nobody rebuilds it: the same evening a placement built on these points was measured
three ways (swap in, keep through the refit placement, windowed-median anchor) against a ruler that
indexed the endzone keypoints half a second off (timeline frame - offset instead of + offset), which
made moving men look 0.2-0.6 m misplaced; all three lost on jitter and hops, and the "defect" was
the instrument. See HANDOFF 2026-09-16 20:35 and the population-and-units memory.
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
