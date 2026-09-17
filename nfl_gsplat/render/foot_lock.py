"""Foot lock: a planted foot stands still on the turf while the body passes over it.

WHY. A fitted body's feet skate: on play 1's live play (2026-09-16, 244 body-frames moving faster
than 0.1 m/frame) the slower ankle moves at 0.94 of the pelvis speed and is planted (under 0.3 of
it) on 1 % of frames, where a runner plants one foot about half the time. The 2D keypoints carry no
stance either (the runner's ankle keypoints jump 0.2-15 px between frames on a 100-px body), so the
fit cannot have one and the smoothers neither add nor remove one. It is the "moonwalk" of fitted
motion, and a viewer sees it without a ruler.

WHAT. Per id, per leg: the ankle's world speed from the drawn body (the pelvis at the state's xy,
the legs under the fitted pose). Frames where it dips under ``stance_ratio`` of the pelvis speed
while the body moves are a stance; consecutive ones form a segment of at least ``min_stance``
frames and at most ``max_stance``. Each segment's foot is pinned to one ground point (the median
of the segment's ankle xy, so the correction is smallest at the middle and grows to half the skate
at the ends), and the hip and knee rotations of that leg are re-solved per frame so the ankle
reaches the pin at its fitted height, with a pull toward the fitted rotations so the leg bends
the way the fit had it. The pelvis, the other leg and everything above the hips are untouched.
Numpy FK (pose.forward_kinematics) on the id's own rest skeleton; no torch.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from nfl_gsplat.pose.forward_kinematics import (SMPLX_BODY_PARENTS, load_smplx_skeleton, pose_params_to_rotmats,
                                                posed_joint_positions)

MOVING_M: float = 0.06         # a pelvis slower than this per frame is standing: no stance to find
STANCE_RATIO: float = 0.5      # an ankle slower than this share of the pelvis is on the ground
MIN_STANCE: int = 2            # frames a stance must last
MAX_STANCE: int = 8            # ... and at most (a longer dip is a standing man, not a stride)
PULL: float = 0.05             # weight of the pull toward the fitted hip / knee rotations (rad per m); it only
                               # settles the null space, a stronger one made the foot miss its pin by centimetres

LEG = {"L": (0, 3, 7), "R": (1, 4, 8)}      # body_pose rows of hip and knee (joints 1/2, 4/5), ankle joint


def relative_joints(body_pose, global_orient, rest, parents=SMPLX_BODY_PARENTS):
    """Joints ``[22, 3]`` with the pelvis at the origin, under the pose (the renderer's frame
    before the pelvis is put over the state's xy)."""
    R = pose_params_to_rotmats(np.asarray(global_orient, float), np.asarray(body_pose, float).reshape(21, 3))
    J = posed_joint_positions(rest, parents, R)
    return J - J[0]


def ankle_world_xy(seq, rest, parents=SMPLX_BODY_PARENTS):
    """``(pelvis_xy [T, 2], ankles_xy [T, 2, 2])`` (L, R) for a per-frame sequence of
    ``(xy, body_pose, global_orient)``."""
    pel = np.array([np.asarray(s[0], float) for s in seq])
    ank = np.zeros((len(seq), 2, 2))
    for t, (xy, bp, go) in enumerate(seq):
        J = relative_joints(bp, go, rest, parents)
        ank[t, 0] = pel[t] + J[7, :2]
        ank[t, 1] = pel[t] + J[8, :2]
    return pel, ank


def _speeds(xy):
    """Central-difference speed per frame, edges one-sided."""
    v = np.zeros(len(xy))
    if len(xy) < 2:
        return v
    v[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) / 2.0
    v[0] = np.linalg.norm(xy[1] - xy[0])
    v[-1] = np.linalg.norm(xy[-1] - xy[-2])
    return v


def stance_segments(pel_xy, ankle_xy, *, moving_m: float = MOVING_M, stance_ratio: float = STANCE_RATIO,
                    min_stance: int = MIN_STANCE, max_stance: int = MAX_STANCE) -> list:
    """``[(first, last)]`` index runs where the ankle's speed is under ``stance_ratio`` of the pelvis
    speed while the pelvis moves faster than ``moving_m``; runs shorter than ``min_stance`` are
    dropped, longer than ``max_stance`` are cut to their slowest ``max_stance`` frames."""
    vp, va = _speeds(np.asarray(pel_xy, float)), _speeds(np.asarray(ankle_xy, float))
    on = (vp > moving_m) & (va < stance_ratio * vp)
    out = []
    t = 0
    while t < len(on):
        if not on[t]:
            t += 1
            continue
        a = t
        while t + 1 < len(on) and on[t + 1]:
            t += 1
        b = t
        t += 1
        if b - a + 1 < min_stance:
            continue
        if b - a + 1 > max_stance:
            k = int(np.argmin(va[a:b + 1])) + a
            a0, b0 = a, b
            a = max(a0, k - max_stance // 2)
            b = min(b0, a + max_stance - 1)
            a = max(a0, b - max_stance + 1)
        out.append((a, b))
    return out


def solve_leg(body_pose, global_orient, rest, side: str, target_xy, pelvis_xy, *, pull: float = PULL,
              parents=SMPLX_BODY_PARENTS):
    """``(body_pose, miss_m)``: the hip and knee rows of ``side`` re-solved so the ankle's world xy lands on
    ``target_xy`` with the pelvis at ``pelvis_xy``; the rest of the pose untouched."""
    hip, knee, ankle = LEG[side]
    bp0 = np.asarray(body_pose, float).reshape(21, 3).copy()
    go = np.asarray(global_orient, float)
    x0 = np.concatenate([bp0[hip], bp0[knee]])
    tgt = np.asarray(target_xy, float) - np.asarray(pelvis_xy, float)

    def resid(x):
        bp = bp0.copy()
        bp[hip], bp[knee] = x[:3], x[3:]
        J = relative_joints(bp, go, rest, parents)
        return np.concatenate([J[ankle, :2] - tgt, pull * (x - x0)])

    sol = least_squares(resid, x0, method="lm", max_nfev=60)
    bp = bp0.copy()
    bp[hip], bp[knee] = sol.x[:3], sol.x[3:]
    miss = float(np.linalg.norm(resid(sol.x)[:2]))
    return bp, miss


def foot_lock_sequence(seq, betas, body_models_dir, *, moving_m: float = MOVING_M, stance_ratio: float = STANCE_RATIO,
                       min_stance: int = MIN_STANCE, max_stance: int = MAX_STANCE, pull: float = PULL):
    """``(body_poses [T, 21, 3], report)`` for one id's per-frame ``(xy, body_pose, global_orient)``
    sequence (consecutive frames). ``report``: segments per leg, the metres each pinned foot was
    moved (the skate removed), and the solver misses."""
    rest, parents = load_smplx_skeleton(body_models_dir, betas=np.asarray(betas, float)[:10])
    pel, ank = ankle_world_xy(seq, rest, parents)
    out = np.array([np.asarray(s[1], float).reshape(21, 3) for s in seq])
    report = {"segments": 0, "frames": 0, "moved_m": [], "miss_m": []}
    for li, side in enumerate(("L", "R")):
        for a, b in stance_segments(pel, ank[:, li], moving_m=moving_m, stance_ratio=stance_ratio,
                                    min_stance=min_stance, max_stance=max_stance):
            pin = np.median(ank[a:b + 1, li], axis=0)
            report["segments"] += 1
            for t in range(a, b + 1):
                bp, miss = solve_leg(out[t], seq[t][2], rest, side, pin, pel[t], pull=pull, parents=parents)
                out[t] = bp
                report["frames"] += 1
                report["moved_m"].append(float(np.linalg.norm(ank[t, li] - pin)))
                report["miss_m"].append(miss)
    return out, report


def foot_lock_timeline(tl, body_models_dir, *, lo=None, hi=None, **kw):
    """Apply :func:`foot_lock_sequence` to every id of a Timeline in place (frames lo..hi, all when
    None); returns ``{pid: report}``. Runs of consecutive drawn frames are locked separately."""
    by: dict = {}
    for f, states in tl.states.items():
        if (lo is not None and f < lo) or (hi is not None and f > hi):
            continue
        for s in states:
            by.setdefault(int(s.pid), {})[int(f)] = s
    reports = {}
    for pid, byf in by.items():
        fs = sorted(byf)
        runs, run = [], [fs[0]]
        for f in fs[1:]:
            if f == run[-1] + 1:
                run.append(f)
            else:
                runs.append(run)
                run = [f]
        runs.append(run)
        rep = {"segments": 0, "frames": 0, "moved_m": [], "miss_m": []}
        for run in runs:
            if len(run) < 3:
                continue
            seq = [(byf[f].xy, byf[f].body_pose, byf[f].global_orient) for f in run]
            bps, r = foot_lock_sequence(seq, byf[run[0]].betas, body_models_dir, **kw)
            for f, bp in zip(run, bps):
                byf[f].body_pose = bp
            for k in ("segments", "frames"):
                rep[k] += r[k]
            rep["moved_m"] += r["moved_m"]
            rep["miss_m"] += r["miss_m"]
        reports[pid] = rep
    return reports
