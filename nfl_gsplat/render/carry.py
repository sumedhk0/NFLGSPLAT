"""The hands on the ball: the arms of whoever holds the football (08y's ``ball.json`` names him per
frame) are posed to carry it, and the ball is drawn between his hands.

WHY. The ball path puts the ball 0.3 m in front of the holder's pelvis at chest height; the fitted
arms hang or swing wherever the keypoints put them, so on v62 the ball floated a hand's width ahead
of the body (scratchpad/review_v62/strips/render_ball_f600.png). A viewer reads a ball nobody holds
as a mistake before anything else.

WHAT. Per holder the shoulder and elbow rows of the fitted body_pose are turned toward a carry pose
(both hands in front of the chest, a ball's width apart) by a weight that ramps over BLEND frames
where the holding starts or stops, so the arms of the receiver close on the ball over a few frames
after the catch and the passer's arms go back to the fit when the ball leaves. The wrists are
straightened by the same weight. The ball is then placed at the midpoint of the posed wrists, pushed
HANDS_FWD_M along the body's forward (the palms are that far beyond the wrist joints), so the ball
sits in the hands whatever the body's height or crouch.

The carry rotations were solved on the real SMPL-X model (2026-09-18, scratch probe under smplx312):
rest is a T-pose with +x the body's LEFT, +y up, +z forward; a shoulder turned about -y swings the
arm forward, about -z drops it (mirrored signs on the right); the fit put the wrists at
(+-0.11, -0.17, 0.33) from the model origin, 5 cm from the target (+-0.10, -0.22, 0.30) which is a
ball 0.3 m in front of the pelvis and a hand above it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.render.gait import blend_weights

BLEND: int = 4                 # frames to close the hands on the ball or let it go
HANDS_FWD_M: float = 0.06      # the ball's centre this far beyond the wrists' midpoint along the body's forward
MIRROR = np.array([1.0, -1.0, -1.0])     # a rotation vector mirrored across the body's sagittal plane
SHOULDER_L = np.array([0.0, -0.906, -0.725])
ELBOW_L = np.array([0.0, -1.087, -0.362])
CARRY_ROWS = {15: SHOULDER_L, 16: SHOULDER_L * MIRROR, 17: ELBOW_L, 18: ELBOW_L * MIRROR}
WRIST_ROWS = (19, 20)


def load_holders(play_dir) -> dict[int, int]:
    """``{frame: pid}`` of the body holding the ball, from ``<play-dir>/ball.json`` (frames with no
    holder -- the flight, the ball on the ground -- are absent). {} without a ball path."""
    f = Path(play_dir) / "ball.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {int(k): int(r["pid"]) for k, r in d["frames"].items() if r.get("pid") is not None}


def holder_weights(frames_of_pid, held_frames, *, blend: int = BLEND) -> dict[int, float]:
    """``{frame: weight}`` for one id: 1 on the frames it holds the ball, ramping over ``blend``
    frames at each edge of a holding stretch (gait.blend_weights), 0 elsewhere. ``frames_of_pid``
    are the frames the id is drawn on, in any order."""
    fs = sorted(int(f) for f in frames_of_pid)
    if not fs:
        return {}
    on = [f in held_frames for f in fs]
    w = blend_weights(on, blend=blend)
    return {f: float(x) for f, x in zip(fs, w) if x > 0}


def carry_body_pose(body_pose, w: float):
    """A copy of ``body_pose [21, 3]`` with the shoulder and elbow rows turned toward the carry pose
    by ``w`` in 0..1 (a slerp per joint) and the wrist rows scaled toward zero by the same weight."""
    bp = np.array(body_pose, float).reshape(21, 3).copy()
    w = float(np.clip(w, 0.0, 1.0))
    if w <= 0:
        return bp
    for row, target in CARRY_ROWS.items():
        a = Rotation.from_rotvec(bp[row]); b = Rotation.from_rotvec(target)
        # slerp: a * (a^-1 b)^w
        step = (a.inv() * b).as_rotvec() * w
        bp[row] = (a * Rotation.from_rotvec(step)).as_rotvec()
    for row in WRIST_ROWS:
        bp[row] *= (1.0 - w)
    return bp


def ball_between_hands(joints_world, yaw: float, *, fwd_m: float = HANDS_FWD_M) -> np.ndarray:
    """The ball's centre from a body's posed world joints ``[22+, 3]``: the wrists' midpoint pushed
    ``fwd_m`` along the body's forward on the ground (``yaw``)."""
    j = np.asarray(joints_world, float)
    mid = 0.5 * (j[20] + j[21])
    return mid + fwd_m * np.array([np.cos(yaw), np.sin(yaw), 0.0])


# ---- the throw -----------------------------------------------------------------------------------
# Two key poses of a right-handed throw, solved on the real SMPL-X model (scratchpad
# solve_throw_pose.py, 2026-09-18): COCKED = the ball beside the ear, the elbow out, the left arm
# pointing at the target (wrists at L (0.29, 0.08, 0.46), R (-0.41, 0.36, -0.14) from the model
# origin); RELEASE = the throwing arm extended up and forward, the left arm dropping across
# (L (0.28, -0.20, 0.25), R (-0.25, 0.53, 0.23)). A left-handed passer mirrors every row.
THROW_HAND: str = "R"
THROW_COCK: int = 6            # frames over which the arms go from the carry to the cocked pose ...
THROW_SWING: int = 8           # ... then from cocked to the release pose (the ball leaves on the last)
THROW_FOLLOW: int = 8          # frames after the release over which the arms go back to the fit
PALM_M: float = 0.08           # the ball's centre this far beyond the throwing wrist along the forearm
COCKED_ROWS = {15: np.array([-0.404, -1.266, -0.261]), 17: np.array([0.0, -0.193, 0.118]),
               16: np.array([0.0, -0.263, -0.237]), 18: np.array([0.0, 0.0, -1.313])}
RELEASE_ROWS = {15: np.array([0.229, -0.457, -0.762]), 17: np.array([-0.105, -1.143, 0.0]),
                16: np.array([0.229, 0.61, -1.294]), 18: np.array([0.0, 0.0, -0.131])}
# The way from the carry to the cocked pose: the shoulder's straight slerp from arm-forward-down to
# arm-up-back passes through zero rotation, the T-pose, and v64/v66 drew the ball at arm's length out to
# the side for two frames (516-518). LIFT is a waypoint at phase 0.5 -- the arm raised in FRONT, the ball
# coming up past the shoulder -- checked on the model: the throwing wrist rises from (-0.11, -0.18, 0.34)
# through (-0.42, 0.09, 0.36) and (-0.36, 0.42, 0.15) to the cocked (-0.41, 0.36, -0.14), never out at
# the body's side, the shoulder never under 19 degrees from rest, the elbow 51-71 degrees flexed.
LIFT_ROWS = {15: np.array([0.1, -1.0, -0.3]), 17: np.array([0.0, -0.9, 0.0]),
             16: np.array([0.0, 0.35, -0.55]), 18: np.array([0.0, 0.5, -1.0])}


def _mirror_rows(rows: dict) -> dict:
    """The same arm pose for the other hand: left and right rows swapped, each mirrored."""
    swap = {15: 16, 16: 15, 17: 18, 18: 17}
    return {swap[r]: np.asarray(v, float) * MIRROR for r, v in rows.items()}


def _slerp_rows(a: dict, b: dict, u: float) -> dict:
    u = float(np.clip(u, 0.0, 1.0))
    out = {}
    for r in a:
        ra = Rotation.from_rotvec(a[r]); rb = Rotation.from_rotvec(b[r])
        out[r] = (ra * Rotation.from_rotvec((ra.inv() * rb).as_rotvec() * u)).as_rotvec()
    return out


def throw_rows(phase: float, hand: str = THROW_HAND) -> dict:
    """The four arm rows at ``phase``: 0 = the carry pose, 0.5 = the lift, 1 = cocked, 2 = the
    release, slerped between; ``hand`` "R" or "L"."""
    carry, lift, cocked, release = CARRY_ROWS, LIFT_ROWS, COCKED_ROWS, RELEASE_ROWS
    if hand == "L":
        lift, cocked, release = _mirror_rows(lift), _mirror_rows(cocked), _mirror_rows(release)
    if phase <= 0.5:
        return _slerp_rows(carry, lift, 2.0 * phase)
    if phase <= 1.0:
        return _slerp_rows(lift, cocked, 2.0 * (phase - 0.5))
    return _slerp_rows(cocked, release, phase - 1.0)


def throw_body_pose(body_pose, phase: float, w: float, hand: str = THROW_HAND):
    """A copy of ``body_pose`` with the arm rows turned toward throw_rows(phase) by ``w``."""
    bp = np.array(body_pose, float).reshape(21, 3).copy()
    w = float(np.clip(w, 0.0, 1.0))
    if w <= 0:
        return bp
    for row, target in throw_rows(phase, hand).items():
        a = Rotation.from_rotvec(bp[row]); b = Rotation.from_rotvec(target)
        bp[row] = (a * Rotation.from_rotvec((a.inv() * b).as_rotvec() * w)).as_rotvec()
    for row in WRIST_ROWS:
        bp[row] *= (1.0 - w)
    return bp


def throw_schedule(release: int, *, cock: int = THROW_COCK, swing: int = THROW_SWING,
                   follow: int = THROW_FOLLOW) -> dict[int, tuple[float, float]]:
    """``{frame: (phase, weight)}`` around ``release``: the arms cock over ``cock`` frames, swing
    over ``swing`` (phase 2 on the release frame itself, when the ball is already in flight), then
    the release pose fades over ``follow`` frames. The weight is 1 through the throw."""
    out = {}
    start = release - swing - cock
    for f in range(start, release + follow + 1):
        if f < release - swing:
            phase = (f - start + 1) / float(cock)
            w = 1.0
        elif f <= release:
            phase = 1.0 + (f - (release - swing) + 1) / float(swing + 1)     # 2.0 on the release frame
            w = 1.0
        else:
            phase = 2.0
            w = 1.0 - (f - release) / float(follow)
        if w > 0:
            out[f] = (float(phase), float(w))
    return out


def ball_at_hand(joints_world, hand: str = THROW_HAND, *, palm_m: float = PALM_M) -> np.ndarray:
    """The ball's centre in one hand: the wrist pushed ``palm_m`` along the forearm."""
    j = np.asarray(joints_world, float)
    elbow, wrist = (j[19], j[21]) if hand == "R" else (j[18], j[20])
    d = wrist - elbow
    n = float(np.linalg.norm(d))
    return wrist + (palm_m * d / n if n > 1e-6 else 0.0)


# ---- the catch -----------------------------------------------------------------------------------
# The receiver's hands go up to a ball arriving at head height over the last CATCH_REACH frames of the
# flight (the REACH pose: both arms forward and up, solved on the model, scratchpad solve_reach_pose.py:
# wrists at L (0.22, 0.21, 0.41), R (-0.17, 0.34, 0.36) from the model origin), then settle into the carry
# over CATCH_SETTLE frames with the ball between the hands. Without it (v64) the ball landed on the chest
# of a man whose arms hung at his sides.
CATCH_REACH: int = 8
CATCH_SETTLE: int = 6
REACH_ROWS = {15: np.array([0.2, -0.999, 0.699]), 17: np.array([0.0, -0.779, 0.2]),
              16: np.array([-0.23, 0.999, -0.699]), 18: np.array([0.0, 0.899, -0.2])}


def catch_schedule(catch: int, *, reach: int = CATCH_REACH, settle: int = CATCH_SETTLE) -> dict[int, tuple[float, float]]:
    """``{frame: (phase, weight)}`` around ``catch``: the arms rise to the reach pose over the ``reach``
    frames before the catch (phase 0, the weight ramping to 1 on the catch frame), then turn into the
    carry pose over ``settle`` frames (phase 0 -> 1, weight 1). After that the carry rule takes over."""
    out = {}
    for f in range(catch - reach + 1, catch + settle + 1):
        if f <= catch:
            out[f] = (0.0, (f - (catch - reach)) / float(reach))
        else:
            out[f] = (min(1.0, (f - catch) / float(settle)), 1.0)
    return out


def catch_body_pose(body_pose, phase: float, w: float):
    """A copy of ``body_pose`` with the arm rows turned by ``w`` toward the reach pose (phase 0) slerped
    into the carry pose (phase 1)."""
    bp = np.array(body_pose, float).reshape(21, 3).copy()
    w = float(np.clip(w, 0.0, 1.0))
    if w <= 0:
        return bp
    for row, target in _slerp_rows(REACH_ROWS, CARRY_ROWS, phase).items():
        a = Rotation.from_rotvec(bp[row]); b = Rotation.from_rotvec(target)
        bp[row] = (a * Rotation.from_rotvec((a.inv() * b).as_rotvec() * w)).as_rotvec()
    for row in WRIST_ROWS:
        bp[row] *= (1.0 - w)
    return bp


def load_ball_meta(play_dir) -> dict:
    """``{"release", "catch", "passer", "receiver"}`` from ``<play-dir>/ball.json`` (values may be
    None), {} without a ball path. The passer is the holder on the frame before the release."""
    f = Path(play_dir) / "ball.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    holders = {int(k): r.get("pid") for k, r in d["frames"].items()}
    rel = d.get("release")
    return {"release": rel, "catch": d.get("catch"), "receiver": d.get("receiver"),
            "passer": holders.get(int(rel) - 1) if rel is not None else None}
