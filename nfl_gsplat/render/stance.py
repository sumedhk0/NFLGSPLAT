"""The quarterback under centre: a canonical stance for the pre-snap frames the tracker never saw him on.

WHY. play_timeline holds the quarterback at the spot he steps back from (endzone_only_rule.qb_hold) for
the whole pre-snap -- 164 frames on play 1, 40 % of the clip -- and draws him there with the first pose
his track was fitted with: a crouch from the frame he steps back on, which reads as one more lineman
bent over the ball. Under centre a quarterback stands with the knees soft, the torso bent forward
about thirty degrees and both arms down and forward, the hands between the centre's legs.

WHAT. UNDER_CENTRE is a full 21-row body_pose (SMPL-X order, axis-angle) of that stance, checked on
the real model (2026-09-18, scratch probe under smplx312: +x on a spine row bends the torso forward, a
hip about -x and a knee about +x fold the leg, a shoulder about -y (left) / +y (right) swings the arm
forward and about -z / +z drops it; with these rows the head sits 0.25 m forward of the pelvis line,
the wrists 0.47 m below the model origin and 0.23 m in front). stance_body_pose turns every row of a
fitted pose toward it by a weight, so the held frames wear the stance and the last few before the
track begins ease into the fit.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

BLEND: int = 6                 # frames before the track begins over which the stance eases into the fit
UNDER_CENTRE = np.zeros((21, 3))
UNDER_CENTRE[0] = [-0.50, 0.0, 0.0]          # L hip: flexed forward
UNDER_CENTRE[1] = [-0.50, 0.0, 0.0]          # R hip
UNDER_CENTRE[3] = [0.75, 0.0, 0.0]           # L knee: bent
UNDER_CENTRE[4] = [0.75, 0.0, 0.0]           # R knee
UNDER_CENTRE[2] = [0.30, 0.0, 0.0]           # spine1: torso forward ...
UNDER_CENTRE[5] = [0.30, 0.0, 0.0]           # spine2
UNDER_CENTRE[8] = [0.15, 0.0, 0.0]           # spine3
UNDER_CENTRE[11] = [-0.35, 0.0, 0.0]         # neck: the head comes up to look over the line
UNDER_CENTRE[15] = [0.0, -1.15, -0.80]       # L shoulder: the arm forward and down
UNDER_CENTRE[16] = [0.0, 1.15, 0.80]         # R shoulder (mirrored)
UNDER_CENTRE[17] = [0.0, -0.30, 0.0]         # L elbow: a little bent
UNDER_CENTRE[18] = [0.0, 0.30, 0.0]          # R elbow


def stance_body_pose(body_pose, w: float, stance=UNDER_CENTRE):
    """A copy of ``body_pose [21, 3]`` with every row turned toward ``stance`` by ``w`` in 0..1."""
    bp = np.array(body_pose, float).reshape(21, 3).copy()
    w = float(np.clip(w, 0.0, 1.0))
    if w <= 0:
        return bp
    if w >= 1:
        return np.array(stance, float).copy()
    for row in range(21):
        a = Rotation.from_rotvec(bp[row]); b = Rotation.from_rotvec(stance[row])
        bp[row] = (a * Rotation.from_rotvec((a.inv() * b).as_rotvec() * w)).as_rotvec()
    return bp


def stance_weights(held_frames, *, blend: int = BLEND) -> dict[int, float]:
    """``{frame: weight}`` for one held id: 1 on its held frames except the last ``blend`` before the
    hold ends, which ramp down toward the fitted pose that follows."""
    fs = sorted(int(f) for f in held_frames)
    if not fs:
        return {}
    n = len(fs)
    out = {}
    for i, f in enumerate(fs):
        k = n - 1 - i                                     # frames left before the hold ends
        out[f] = 1.0 if k >= blend else (k + 1) / float(blend + 1)
    return out
