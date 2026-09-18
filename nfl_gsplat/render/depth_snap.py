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
# The 3 % of snaps that pick the wrong man are the ones that disagree with the same body's own
# neighbouring snaps: play 1's id 1 slid 2.0 m along its ray at frame 420 with no endzone row of its
# own (a same-team body lay on the ray) and the smoother spread it into a 12-frame ramp -- one of the
# live-play hops the user sees. The correction a body needs along its ray varies slowly, so a snap
# more than VETO_M from the median of its own snaps within VETO_WINDOW frames (2+ of them), or alone
# in its window and larger than VETO_M outright, is dropped. Measured on play 1 (2026-09-15): 110 of
# 4635 snaps vetoed; live steps > 0.25 m/frame 19 -> 15, whole clip 196 -> 186 and the last step over
# 0.6 m gone, root jitter p90 0.0286 -> 0.0278, census and the endzone reprojection unchanged.
# Replacing every correction by the windowed MEDIAN instead was measured and REJECTED (live 19 -> 28):
# the piecewise medians step where the window's membership changes, and filling frames the gates had
# left unsnapped moved bodies that were right where they stood.
VETO_WINDOW: int = 6
VETO_M: float = 1.0


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


EXCLUSIVE: bool = False     # one endzone body snaps at most one sideline body per frame (the nearer ray keeps it;
                            # measured 2026-09-18: 116 of 4337 live snaps had two claimants)


def _lateral(xy, centre, e):
    xy = np.asarray(xy, float); centre = np.asarray(centre, float); e = np.asarray(e, float)
    u = xy - centre
    n = float(np.linalg.norm(u))
    if n < MIN_RAY_M:
        return float("inf")
    u = u / n
    along = float((e - centre) @ u)
    return float(np.linalg.norm((e - centre) - along * u))


def veto_outlier_snaps(deltas: dict, *, window: int = VETO_WINDOW, veto_m: float = VETO_M) -> set:
    """``{(frame, pid)}`` of the snaps to drop from ``deltas`` ``{pid: {frame: along-ray correction}}``:
    a correction more than ``veto_m`` from the median of the same body's other snaps within
    ``window`` frames when there are at least two of them, or, with fewer, larger than ``veto_m``
    outright. The per-frame gates cannot see this -- they judge one frame -- and the wrong-man
    snaps are exactly the ones that disagree with their neighbours."""
    bad = set()
    for pid, byf in deltas.items():
        fs = np.asarray(sorted(byf))
        ds = np.asarray([byf[f] for f in fs])
        for i, f in enumerate(fs):
            near = (np.abs(fs - f) <= window) & (fs != f)
            if near.sum() >= 2:
                if abs(ds[i] - float(np.median(ds[near]))) > veto_m:
                    bad.add((int(f), int(pid)))
            elif abs(ds[i]) > veto_m:
                bad.add((int(f), int(pid)))
    return bad


def snap_ground(ground_side: dict, ground_other: dict, track, *, teams=None, frame_shift: int = 0,
                lateral_m: float = LATERAL_M, margin_m: float = MARGIN_M, max_move_m: float = MAX_MOVE_M,
                veto_window: int = VETO_WINDOW, veto_m: float = VETO_M, exclusive: bool | None = None):
    """``(ground, n_snapped)``: ``ground_side`` (frame -> {pid: xy}) with each body sliding along
    its own ray to the nearest body of ``ground_other`` (keyed by that camera's own frames, i.e.
    ``frame + frame_shift``). ``teams`` ``{pid: team}`` gates the match; an id whose team is
    unknown may match any. A snap that disagrees with the same body's neighbouring snaps is then
    undone (veto_outlier_snaps; ``veto_window`` 0 turns that off)."""
    out: dict = {}
    deltas: dict = {}
    for f, bodies in ground_side.items():
        others_all = ground_other.get(int(f) + int(frame_shift))
        if not others_all or int(f) >= len(track.conf) or track.conf[int(f)] <= 0:
            out[f] = dict(bodies)
            continue
        centre = camera_ground_centre(track, int(f))
        new: dict = {}
        took_by: dict = {}
        for pid, xy in bodies.items():
            side = None if teams is None else teams.get(int(pid))
            others = ({q: e for q, e in others_all.items() if teams.get(int(q)) == side}
                      if (teams is not None and side is not None) else dict(others_all))
            moved, took = snap_one(xy, centre, others, lateral_m=lateral_m, margin_m=margin_m,
                                   max_move_m=max_move_m)
            new[pid] = moved
            if took is not None:
                took_by.setdefault(int(took), []).append((_lateral(xy, centre, others_all[took]), int(pid)))
        if (EXCLUSIVE if exclusive is None else exclusive):
            # one endzone body, one sideline body: the nearer ray keeps it, the others stay put
            for q, claimants in took_by.items():
                if len(claimants) > 1:
                    claimants.sort()
                    for _lat, pid in claimants[1:]:
                        new[pid] = np.asarray(bodies[pid], float)
                    took_by[q] = claimants[:1]
        for q, claimants in took_by.items():
            for _lat, pid in claimants:
                moved, xy = new[pid], bodies[pid]
                # the signed slide along the ray: positive away from the camera
                deltas.setdefault(int(pid), {})[int(f)] = (float(np.linalg.norm(np.asarray(moved, float) - centre))
                                                           - float(np.linalg.norm(np.asarray(xy, float) - centre)))
        out[f] = new
    bad = veto_outlier_snaps(deltas, window=veto_window, veto_m=veto_m) if veto_window > 0 else set()
    for f, pid in bad:
        out[f][pid] = ground_side[f][pid]
    n_snap = sum(len(v) for v in deltas.values()) - len(bad)
    return out, n_snap
