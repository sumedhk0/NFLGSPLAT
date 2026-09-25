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
# A run of snaps onto the WRONG man agree with each other and pass the outlier veto; what they do is
# make the body jump from where it stood the frame before (veto_jumps). Measured on play 1's play window
# 393-639 (2026-09-18): steps > 0.25 m/frame 23 -> 11, hops 2 -> 0, root jitter p90 0.0301 -> 0.0284 and
# p99 0.0908 -> 0.0721, census 1.397 -> 1.409 (BAL +0.04); at 0.6 m steps 10 but census 1.47 (KC +0.10:
# right snaps refused). The sideline reprojection cannot see a slide along the ray, and the vetoed frames
# have no endzone row of their own, so no reprojection ruler moves.
JUMP_M: float | None = 1.0
JUMP_REACH: int = 3         # the previous drawn frame must lie within this many frames to judge a jump
                            # (8 and 15 measured 2026-09-18: steps 11 -> 15 for census 1.409 -> 1.405 -- a body
                            # returning after a hole is judged against a stale point; 3 keeps the step ruler)


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


# The body's OWN endzone detection first. The positional snap below takes the nearest same-team endzone body within
# LATERAL_M of the sideline ray and refuses when a runner-up lies within MARGIN_M of it -- two bodies on one ray. Men
# side by side across the field share a sideline ray, so on play 1 the refusal left #76 and the quarterback a metre off
# in depth at 580-606 (endzone film: drawn on empty turf beside their own keypoints) while the endzone boxed each
# under his own id. With SNAP_OWN_ID, an id whose own endzone point lies within OWN_LATERAL_M of its ray (and within
# MAX_MOVE_M along it) slides onto it with no margin test; otherwise the positional snap. The outlier and jump vetoes
# still apply. Read at call time.
# ADOPTED 2026-09-25 (v114) with play 1's id 186 untangled in the tables (the twin that rode Thuney would otherwise
# have been pulled onto the quarterback's endzone rows under the same id and drawn): endzone film, the quarterback
# 585-606 and #76 580-601 on their keypoints (a metre off before); census live 0.13 unchanged, full 1.40 -> 1.39;
# steps full 37 -> 36; hops 0.
SNAP_OWN_ID: bool = True
OWN_LATERAL_M: float = 0.8


def snap_own(xy, centre, own, *, lateral_m: float = None, max_move_m: float = MAX_MOVE_M):
    """``xy`` slid along its ray from ``centre`` onto ``own`` (the id's own point in the other camera), or None when
    ``own`` lies more than ``lateral_m`` (default OWN_LATERAL_M) off the ray or more than ``max_move_m`` along it."""
    lateral_m = OWN_LATERAL_M if lateral_m is None else float(lateral_m)
    xy = np.asarray(xy, float)
    centre = np.asarray(centre, float)
    u = xy - centre
    n = float(np.linalg.norm(u))
    if n < MIN_RAY_M:
        return None
    u = u / n
    e = np.asarray(own, float)
    along = float((e - centre) @ u)
    lateral = float(np.linalg.norm((e - centre) - along * u))
    if lateral > lateral_m or abs(along - n) > max_move_m:
        return None
    return centre + along * u


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
                veto_window: int = VETO_WINDOW, veto_m: float = VETO_M, exclusive: bool | None = None,
                jump_m: float | None | bool = None, own_id: bool | None = None, own_lateral_m: float | None = None):
    """``(ground, n_snapped)``: ``ground_side`` (frame -> {pid: xy}) with each body sliding along
    its own ray to the nearest body of ``ground_other`` (keyed by that camera's own frames, i.e.
    ``frame + frame_shift``). ``teams`` ``{pid: team}`` gates the match; an id whose team is
    unknown may match any. A snap that disagrees with the same body's neighbouring snaps is then
    undone (veto_outlier_snaps; ``veto_window`` 0 turns that off)."""
    out: dict = {}
    deltas: dict = {}
    own_on = SNAP_OWN_ID if own_id is None else bool(own_id)
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
            moved = took = None
            if own_on:
                own = others_all.get(pid, others_all.get(int(pid)))
                if own is not None:
                    moved = snap_own(xy, centre, own, lateral_m=own_lateral_m, max_move_m=max_move_m)
                    if moved is not None:
                        took = pid if pid in others_all else int(pid)
            if moved is None:
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
    jump_m = JUMP_M if jump_m is None else jump_m          # None = the module's setting; False = off
    if jump_m:
        bad |= veto_jumps(out, ground_side, deltas, bad, jump_m=float(jump_m))
    n_snap = sum(len(v) for v in deltas.values()) - len(bad)
    return out, n_snap


def veto_jumps(out: dict, ground_side: dict, deltas: dict, already: set, *, jump_m: float, reach: int | None = None) -> set:
    """Undo (in ``out``) every snap that makes a body JUMP: its snapped point more than ``jump_m``
    from the same body's point on the previous drawn frame (within ``reach`` frames) while its
    unsnapped point is within ``jump_m`` of it. Returns the ``{(frame, pid)}`` undone. The outlier
    veto compares a snap with the body's other snaps; a run of snaps onto the wrong man agree with
    each other and pass it (play 1 id 28 at 588-594: +2.0, +1.8, +1.7, +1.2 m along its ray, onto a
    teammate 2 m deeper, with no endzone row of its own). Frames are walked in order, so a vetoed
    frame's raw point is what the next frame is judged against."""
    reach = JUMP_REACH if reach is None else int(reach)
    by_pid: dict = {}
    for f, bodies in out.items():
        for pid, xy in bodies.items():
            by_pid.setdefault(int(pid), {})[int(f)] = xy
    undone = set()
    for pid, byf in by_pid.items():
        prev = None
        for f in sorted(byf):
            xy = np.asarray(byf[f], float)
            if prev is not None and f - prev[0] <= reach and f in deltas.get(pid, {}) and (f, pid) not in already:
                raw = np.asarray(ground_side[f][pid], float)
                if np.linalg.norm(xy - prev[1]) > jump_m >= np.linalg.norm(raw - prev[1]):
                    out[f][pid] = ground_side[f][pid]
                    xy = raw
                    undone.add((int(f), int(pid)))
            prev = (f, xy)
    return undone
