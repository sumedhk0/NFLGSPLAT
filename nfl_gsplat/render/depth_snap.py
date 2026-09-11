"""Depth for a body only one camera sees, from the other camera's detections.

WHY. A sideline body stands where its ankle ray meets the turf. That point slides
ALONG the ray with a few pixels of error, and the sideline camera is nearly
horizontal, so a small pixel error is a large error in depth -- the axis the
sideline cannot see. Only about seven of twenty-two players a frame are paired
across the cameras, so most bodies carry that error into the render, where it
shows as a body standing between two real players in the endzone view.

WHAT. The endzone camera sees that axis. For each sideline body: walk along its
own sideline ray and take the endzone body whose ground point lies nearest the
ray, subject to gates -- same team, within LATERAL_M of the ray, no further than
MAX_MOVE_M from where the sideline put it, and clearly nearer the ray than the
runner-up (MARGIN_M). Then slide the body along its ray to that point. The
sideline image is unchanged by construction (the body stays on its own ray); only
its depth moves.

This needs no track pairing -- it matches per frame, so it helps exactly the
bodies the pairing could not match, which is the point.

MEASURED (play 1, 2026-09-11, 3479 body-frames of PAIRED players with their
two-camera answer held out as truth): it fires on 69 % of frames, picks the right
man on 97 % of those, and the distance to truth falls from 0.20 m to 0.05 m at
the median and 0.43 m to 0.15 m at p90. It is worse on 8 % of the frames it
touches. Without the team gate the right-man rate is 91 %.
"""
from __future__ import annotations

import numpy as np

LATERAL_M: float = 0.6      # an endzone body this close to the sideline ray may be this man
MARGIN_M: float = 0.5       # and the runner-up must be this much further off the ray
MAX_MOVE_M: float = 2.5     # never slide a body further than this
MIN_RAY_M: float = 1.0      # a body this close to the camera has no usable ray direction


def camera_ground_centre(track, f) -> np.ndarray:
    """The camera's own position on the ground plane at frame ``f``."""
    R = np.asarray(track.R[f], float)
    t = np.asarray(track.t[f], float)
    return (-R.T @ t)[:2]


def snap_one(xy, centre, others, *, lateral_m: float = LATERAL_M, margin_m: float = MARGIN_M,
             max_move_m: float = MAX_MOVE_M):
    """``(new xy, id snapped to)`` or ``(xy, None)``. ``others``: ``{id: xy}`` from the other
    camera, already filtered to the bodies this one could be (the same team)."""
    xy = np.asarray(xy, float)
    centre = np.asarray(centre, float)
    u = xy - centre
    n = float(np.linalg.norm(u))
    if n < MIN_RAY_M:
        return xy, None
    u = u / n
    cands = []
    for pid, e in others.items():
        e = np.asarray(e, float)
        along = float((e - centre) @ u)
        lateral = float(np.linalg.norm((e - centre) - along * u))
        if lateral <= lateral_m and abs(along - n) <= max_move_m:
            cands.append((lateral, int(pid), centre + along * u))
    if not cands:
        return xy, None
    cands.sort(key=lambda c: c[0])
    if len(cands) > 1 and cands[1][0] - cands[0][0] < margin_m:
        return xy, None                      # two bodies on the ray: which one is not knowable
    return cands[0][2], cands[0][1]


def snap_ground(ground_side: dict, ground_other: dict, track, *, teams=None, frame_shift: int = 0,
                lateral_m: float = LATERAL_M, margin_m: float = MARGIN_M, max_move_m: float = MAX_MOVE_M):
    """``(ground, n_snapped)``: ``ground_side`` (frame -> {pid: xy}) with each body sliding along
    its own ray to the nearest body of ``ground_other`` (keyed by that camera's own frames, i.e.
    ``frame + frame_shift``). ``teams`` ``{pid: team}`` gates the match; an id whose team is
    unknown may match any."""
    out: dict = {}
    n_snap = 0
    for f, bodies in ground_side.items():
        others_all = ground_other.get(int(f) + int(frame_shift))
        if not others_all or int(f) >= len(track.conf) or track.conf[int(f)] <= 0:
            out[f] = dict(bodies)
            continue
        centre = camera_ground_centre(track, int(f))
        new: dict = {}
        for pid, xy in bodies.items():
            side = None if teams is None else teams.get(int(pid))
            others = ({q: e for q, e in others_all.items() if teams.get(int(q)) == side}
                      if (teams is not None and side is not None) else dict(others_all))
            moved, took = snap_one(xy, centre, others, lateral_m=lateral_m, margin_m=margin_m,
                                   max_move_m=max_move_m)
            new[pid] = moved
            n_snap += int(took is not None)
        out[f] = new
    return out, n_snap
