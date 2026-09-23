"""The tackled carrier goes to the ground: a synthesised fall ending on the down frame.

WHY. The detector loses a man the moment he is under the tackle (play 1's receiver: last box 602, down
on the film at 607), so the renderer holds his last STANDING state through the tail and the pile is
drawn upright while the film shows three men on the turf (v72, diag/catch/catch_check_v72.png at 606
and 614). The carrier is the man the viewer watches; his fall is the play's last event.

WHAT. Over FALL_FRAMES frames ending on the down frame (08y's ``down`` in ball.json, the frame 08x ends
the play on) the carrier's orientation pitches forward about the horizontal axis to his left -- the
body's up turns into its forward, so he lies face down along the way he was running, the way a man
wrapped from behind falls -- and the hips, knees and spine curl toward TACKLED_ROWS. The arm rows are
left alone: the carry pose (render.carry) keeps the ball between his wrists, and 05k draws it there.
placed_body puts the lowest vertex on the turf, so the pitched body lies on it without any height rule.
Only the carrier falls; the tacklers keep their fitted or held poses (open item).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.render.timeline import yaw_of

FALL_FRAMES: int = 8            # frames over which the carrier goes from upright to the turf
FALL_SETTLE: int = 3            # ... ending this many frames AFTER the down frame: "down" (08y --down, read off the film as
                                # the knee or the body first touching) comes mid-fall; v74 against the film (diag/catch/
                                # tackle_check_v74.png) had the pile flat at 604 where the film's is flat at 608-610
FALL_PITCH: float = np.pi / 2   # radians forward at the end of the fall: face down
TACKLED_ROWS = {0: np.array([-0.60, 0.0, 0.0]),   # L hip: flexed
                1: np.array([-0.60, 0.0, 0.0]),   # R hip
                3: np.array([0.90, 0.0, 0.0]),    # L knee: bent
                4: np.array([0.90, 0.0, 0.0]),    # R knee
                2: np.array([0.25, 0.0, 0.0]),    # spine1: curled forward over the ball ...
                5: np.array([0.20, 0.0, 0.0]),    # spine2
                8: np.array([0.10, 0.0, 0.0]),    # spine3
                11: np.array([-0.30, 0.0, 0.0])}  # neck: the head up off the turf


def fall_schedule(down: int, *, frames: int = FALL_FRAMES, last: int | None = None, settle: int = FALL_SETTLE) -> dict[int, float]:
    """``{frame: phase}`` for the carrier: the phase rises from 1/frames to 1 over the ``frames`` frames
    ending ``settle`` frames after ``down`` and stays 1 through ``last`` (the clip's last frame; the end
    of the fall alone when None)."""
    end = int(down) + int(settle)
    out = {}
    for i in range(int(frames)):
        out[end - int(frames) + 1 + i] = (i + 1) / float(frames)
    for f in range(end + 1, int(last if last is not None else end) + 1):
        out[f] = 1.0
    return out


def _ease(phase: float) -> float:
    p = float(np.clip(phase, 0.0, 1.0))
    return p * p * (3.0 - 2.0 * p)


def fall_orient(global_orient, phase: float, *, pitch: float = FALL_PITCH, direction=None) -> np.ndarray:
    """The body's orientation pitched by ``ease(phase) * pitch`` toward ``direction`` (a ground vector;
    its facing when None) about the horizontal axis perpendicular to it, so at phase 1 the up of the
    body points along that direction: the carrier falls along the way he ran, a tackler onto him."""
    go = np.asarray(global_orient, float).reshape(3)
    p = _ease(phase)
    if p <= 0:
        return go.copy()
    if direction is None:
        yaw = yaw_of(go)
        d = np.array([np.cos(yaw), np.sin(yaw)])
    else:
        d = np.asarray(direction, float)[:2]
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-9 else np.array([np.cos(yaw_of(go)), np.sin(yaw_of(go))])
    axis = np.array([-d[1], d[0], 0.0])                          # z x direction
    return (Rotation.from_rotvec(axis * p * pitch) * Rotation.from_rotvec(go)).as_rotvec()


TACKLER_M: float = 1.5          # an other-team body this close to the carrier on the down frame is in the tackle and falls onto him


# A tackler the detector loses BEFORE the down (he is under the pile with the carrier) is not on the down frame for
# tacklers() to find: play 1 v97, Roquan Smith (171) closes from 1.9 m to 0.5 m of the carrier by 598, his track ends
# at 600, the down is 607 -- the film has two Ravens on the pile, the render one. An other-team body whose last drawn
# frame is within LOST_FRAMES before the down and whose last spot is within LOST_M of the carrier's on that frame is a
# lost tackler: 05k holds him from his last frame (timeline.hold_to_end's ``always``) so the pile keeps him.
LOST_FRAMES: int = 12
LOST_M: float = 1.5


def lost_tacklers(states_by_frame: dict, carrier: int, team_of: dict, down: int, *, lost_frames: int = LOST_FRAMES,
                  within_m: float = LOST_M) -> dict[int, int]:
    """``{pid: last_frame}`` of the other-team bodies whose last drawn frame lies in ``[down - lost_frames, down)`` and
    whose spot on that frame is within ``within_m`` of the carrier's. ``states_by_frame``: frame -> [PlayerState]."""
    my_team = team_of.get(int(carrier))
    last: dict[int, tuple[int, np.ndarray]] = {}
    carrier_xy: dict[int, np.ndarray] = {}
    for f, states in states_by_frame.items():
        for s in states:
            pid = int(s.pid)
            if pid == int(carrier):
                carrier_xy[int(f)] = np.asarray(s.xy, float)
            if pid not in last or int(f) > last[pid][0]:
                last[pid] = (int(f), np.asarray(s.xy, float))
    out = {}
    for pid, (f_last, xy) in last.items():
        if pid == int(carrier) or not (down - lost_frames <= f_last < down):
            continue
        if my_team is not None and team_of.get(pid) == my_team:
            continue
        c = carrier_xy.get(f_last)
        if c is None:
            continue
        if float(np.linalg.norm(np.asarray(xy)[:2] - c[:2])) <= within_m:
            out[pid] = f_last
    return out


def tacklers(states, carrier: int, team_of: dict, *, within_m: float = TACKLER_M) -> dict[int, np.ndarray]:
    """``{pid: direction}`` for the other-team bodies within ``within_m`` of the carrier among ``states``
    (one frame's PlayerStates): the direction is from each toward the carrier on the ground."""
    c = next((s for s in states if int(s.pid) == int(carrier)), None)
    if c is None:
        return {}
    cxy = np.asarray(c.xy[:2], float)
    team = team_of.get(int(carrier))
    out = {}
    for s in states:
        pid = int(s.pid)
        if pid == int(carrier) or team_of.get(pid) is None or team_of.get(pid) == team:
            continue
        d = cxy - np.asarray(s.xy[:2], float)
        if float(np.linalg.norm(d)) <= within_m:
            out[pid] = d
    return out


def tackled_body_pose(body_pose, phase: float, rows=TACKLED_ROWS) -> np.ndarray:
    """A copy of ``body_pose [21, 3]`` with the leg and spine rows turned toward the tackled pose by
    ``ease(phase)``; every other row (the arms holding the ball) is untouched."""
    bp = np.array(body_pose, float).reshape(21, 3).copy()
    w = _ease(phase)
    if w <= 0:
        return bp
    for row, target in rows.items():
        a = Rotation.from_rotvec(bp[row]); b = Rotation.from_rotvec(np.asarray(target, float))
        bp[row] = (a * Rotation.from_rotvec((a.inv() * b).as_rotvec() * w)).as_rotvec()
    return bp
