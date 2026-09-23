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

# ---- the rhythm mode (2026-09-23) -------------------------------------------------------------------------------
# The dips mode above found nothing on the sprinting band (the fitted legs hardly cycle there; the gait replaces
# them). On the JOGGING band (2.5-4.8 m/s, 39 % of play 1's body-frames) the fitted legs DO cycle, with the
# keypoints (sideline ankle residual p50 6 px; the fitted ankle's sweep 13.9 px std against the keypoints' 15.1,
# correlation 0.85-0.99 on every jogger), yet the slower ankle still moves at 0.76 of the pelvis and is planted on
# 9 % of frames: the sweep is a sinusoid, which plants a foot for an instant per cycle, where a stance holds the
# foot for 40 % of the cycle. So: a strike is a local maximum of the fitted hip's forward flexion in the plane of
# the motion (gait.leg_yaw); from it, for DUTY of the cycle to the next strike, the ankle is pinned to its world
# xy at the strike and the leg re-solved (solve_leg) -- the fit's phase and extremes kept, the stance flattened
# onto the turf -- blended at both edges; a stance is released early when the pin falls farther than MAX_BACK_M
# behind the hip. Only frames whose pelvis speed is in the band are touched.
MODE: str = "rhythm"            # "off" | "dips" | "rhythm": what the scripts apply after the gait (read at call time);
                                # rhythm since v103 (2026-09-23), film-checked on play 1
JOG_M: float = 0.042            # pelvis speed (m per frame) above which a body jogs: 2.5 m/s at 59.94 fps
RUN_M_LOCK: float = 0.08        # ... and below which the lock applies (the gait's RUN_M: faster legs are the gait's)
DUTY: float = 0.40              # stance share of a jog cycle
EDGE: int = 1                   # frames to ease a stance in and out INSIDE the window (1 = none: the pin IS the fitted
                                # ankle at the strike, so the start is continuous by construction)
RELEASE: int = 6                # frames AFTER a stance over which the foot swings from the pin back to the fit's own
                                # ankle (smoothstep), so the release is a swing, not a snap: the first A/B (EDGE 3, no
                                # release) raised joint jitter p90 27 % with the foot jumping 0.3 m on the frame after
FLEX_SIGMA: float = 1.5         # Gaussian (frames) on the fitted flexion before its maxima are read
MIN_CYCLE: int = 8              # a strike-to-strike interval outside this range is not a leg cycle (no lock)
MAX_CYCLE: int = 60
MIN_SWEEP: float = 0.15         # rad: a flexion maximum must drop this far before the next maximum to be a strike
MAX_BACK_M: float = 0.45        # the pin farther than this behind the hip along the motion: the foot lets go
WARM_START: bool = True         # start each locked frame's leg solve from the previous locked frame's solution (read at
                                # call time); the pull stays toward the fit, this only picks the nearest of the equal legs
ENGAGED_M: float | None = 1.0   # a stance whose strike frame has an other-team body within this (m) is not locked: an
                                # engaged man drives, chops and pushes -- his feet do slide (v101 on the film: the lock
                                # compressed a rusher's stride against his blocker where the fit matched the film)
ENGAGED_IOU: float | None = 0.05  # ... or whose SIDELINE box an other-team box overlaps at this IoU on ANY frame of the
                                # stance (the film's own contact signal: the placement drew a rusher 1.4-1.6 m from the
                                # blocker he was leaning on; side by side their boxes overlap 0.07-0.19, and his strike
                                # frame read 0.136 under a 0.15 threshold while 459-465 were over it)
ENGAGED_CAM: str = "sideline"
BAND_RULE: str = "strike"       # "window": every frame of a stance must be in the speed band; "strike": the strike frame
                                # must be, and no frame of the stance may reach the gait's speed (a man slowing through
                                # 2.5 m/s mid-stance still plants)


def motion_flexion(hip_rotvec, yaw: float) -> float:
    """The fit's hip flexion (rad, forward positive) in a leg plane turned by ``yaw`` about the pelvis's up axis
    (gait.leg_yaw's convention; ``yaw`` 0 = gait.fit_hip_flexion): the thigh's rest direction (-y) rotated by the
    hip, un-turned by the yaw, its forward component against its down component."""
    from scipy.spatial.transform import Rotation
    d = Rotation.from_rotvec(np.asarray(hip_rotvec, float)).apply([0.0, -1.0, 0.0])
    d = Rotation.from_rotvec([0.0, -float(yaw), 0.0]).apply(d)
    return float(np.arctan2(d[2], -d[1]))


def strikes(flex, *, sigma=None, min_cycle=None, min_sweep=None) -> list:
    """Foot strikes of one leg: local maxima of the Gaussian-smoothed forward flexion ``flex [T]`` that fall by at
    least ``min_sweep`` before the next maximum, at least ``min_cycle`` frames apart (the later one yields). None = the
    module's knobs, read here."""
    sigma = FLEX_SIGMA if sigma is None else float(sigma)
    min_cycle = MIN_CYCLE if min_cycle is None else int(min_cycle)
    min_sweep = MIN_SWEEP if min_sweep is None else float(min_sweep)
    x = np.asarray(flex, float)
    if len(x) < 3:
        return []
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        x = gaussian_filter1d(x, sigma, mode="nearest")
    peaks = [t for t in range(1, len(x) - 1) if x[t] >= x[t - 1] and x[t] > x[t + 1]]
    out = []
    for i, t in enumerate(peaks):
        end = peaks[i + 1] if i + 1 < len(peaks) else len(x)
        if x[t] - x[t:end].min() < min_sweep:
            continue
        if out and t - out[-1] < min_cycle:               # two maxima inside a cycle: the higher one is the strike
            if x[t] > x[out[-1]]:
                out[-1] = int(t)
            continue
        out.append(int(t))
    return out


def stance_windows(strike_frames: list, T: int, *, duty=None, min_cycle=None, max_cycle=None) -> list:
    """``[(t0, t1)]`` per strike: the stance runs from the strike for ``duty`` of the cycle to the next strike (the
    last strike takes the median cycle; a lone strike takes none). Cycles outside min..max are not cycles."""
    duty = DUTY if duty is None else float(duty)
    min_cycle = MIN_CYCLE if min_cycle is None else int(min_cycle)
    max_cycle = MAX_CYCLE if max_cycle is None else int(max_cycle)
    if len(strike_frames) < 2:
        return []
    cycles = np.diff(strike_frames)
    med = float(np.median(cycles))
    out = []
    for i, t0 in enumerate(strike_frames):
        cyc = float(cycles[i]) if i < len(cycles) else med
        if cyc < min_cycle or cyc > max_cycle:
            continue
        t1 = min(T - 1, int(t0) + int(round(duty * cyc)))
        if t1 > t0:
            out.append((int(t0), int(t1)))
    return out


def rhythm_stances(seq, rest, parents=SMPLX_BODY_PARENTS, *, jog_m=None, run_m=None, duty=None, sigma=None,
                   min_cycle=None, max_cycle=None, min_sweep=None, max_back_m=None, band_rule=None,
                   engaged=None) -> list:
    """``[(side, t0, t1, pin_xy)]`` for a per-frame ``(xy, body_pose, global_orient)`` sequence: each jogging leg
    cycle's stance, read off the fit's own flexion rhythm in the plane of the motion, pinned to the ankle's world
    xy at the strike. A window that leaves the speed band is dropped (``band_rule`` "window") or only one whose
    strike is out of the band or that reaches the gait's speed ("strike"); one whose pin falls ``max_back_m``
    behind the hip along the motion ends there. ``engaged [T]`` (bool, optional): frames on which an other-team body
    stands within ENGAGED_M of the man or his box is on an opponent's -- a stance with any such frame is not locked."""
    from nfl_gsplat.render.gait import HIP_ROW, forward_on_ground, leg_yaw
    jog_m = JOG_M if jog_m is None else float(jog_m)
    run_m = RUN_M_LOCK if run_m is None else float(run_m)
    max_back_m = MAX_BACK_M if max_back_m is None else float(max_back_m)
    band_rule = BAND_RULE if band_rule is None else str(band_rule)
    if band_rule not in ("window", "strike"):
        raise ValueError(f"band rule {band_rule!r}: window or strike")
    T = len(seq)
    if T < 3:
        return []
    pel, ank = ankle_world_xy(seq, rest, parents)
    vel = np.zeros((T, 2))
    vel[1:-1] = (pel[2:] - pel[:-2]) / 2.0
    vel[0], vel[-1] = pel[1] - pel[0], pel[-1] - pel[-2]
    speed = np.linalg.norm(vel, axis=1)
    band = (speed >= jog_m) & (speed < run_m)
    if not band.any():
        return []
    yaw = np.zeros(T)
    for t in range(T):
        f = forward_on_ground(seq[t][2])
        if f is not None:
            yaw[t], _adv = leg_yaw(f, vel[t])
    out = []
    for li, side in enumerate(("L", "R")):
        flex = [motion_flexion(np.asarray(seq[t][1], float).reshape(21, 3)[HIP_ROW[side]], yaw[t]) for t in range(T)]
        st = strikes(flex, sigma=sigma, min_cycle=min_cycle, min_sweep=min_sweep)
        for t0, t1 in stance_windows(st, T, duty=duty, min_cycle=min_cycle, max_cycle=max_cycle):
            if engaged is not None and bool(np.asarray(engaged)[t0:t1 + 1].any()):
                continue
            if band_rule == "window" and not band[t0:t1 + 1].all():
                continue
            if band_rule == "strike" and (not band[t0] or (speed[t0:t1 + 1] >= run_m).any()):
                continue
            pin = ank[t0, li].copy()
            end = t1
            for t in range(t0, t1 + 1):                        # release when the pin is too far behind the hip
                u = vel[t] / max(float(np.linalg.norm(vel[t])), 1e-9)
                if float((pin - pel[t]) @ u) < -max_back_m:
                    end = t - 1
                    break
            if end > t0:
                out.append((side, int(t0), int(end), pin))
    return out


def lock_stances(seq, rest, parents, stances, *, edge=None, pull=None, release=None, run_m=None):
    """``(body_poses [T, 21, 3], report)``: each ``(side, t0, t1, pin_xy)`` stance's leg re-solved per frame so the
    ankle lands on the pin (the target eased from the fitted ankle over ``edge`` frames inside the window), then
    over the ``release`` frames after the stance the target swings from the pin back to the fitted ankle on a
    smoothstep. A release frame that belongs to another stance of the same leg, or runs at the gait's speed, is
    left to them."""
    edge = EDGE if edge is None else int(edge)
    pull = PULL if pull is None else float(pull)
    release = RELEASE if release is None else int(release)
    run_m = RUN_M_LOCK if run_m is None else float(run_m)
    T = len(seq)
    out = np.array([np.asarray(s[1], float).reshape(21, 3) for s in seq])
    pel, ank = ankle_world_xy(seq, rest, parents)
    speed = _speeds(pel)
    owned = {"L": set(), "R": set()}
    for side, t0, t1, _pin in stances:
        owned[side].update(range(t0, t1 + 1))
    report = {"segments": 0, "frames": 0, "released": 0, "moved_m": [], "miss_m": []}
    prev = {"L": (None, None), "R": (None, None)}              # the last solved (frame, x) per leg: the warm start

    def solve(t, side, li, target):
        hip, knee, _ankle = LEG[side]
        pf, px = prev[side]
        x_init = px if (WARM_START and pf is not None and t == pf + 1) else None
        bp, miss = solve_leg(out[t], seq[t][2], rest, side, target, pel[t], pull=pull, parents=parents, x_init=x_init)
        out[t] = bp
        prev[side] = (t, np.concatenate([bp[hip], bp[knee]]))
        report["frames"] += 1
        report["moved_m"].append(float(np.linalg.norm(ank[t, li] - target)))
        report["miss_m"].append(miss)

    for side, t0, t1, pin in sorted(stances, key=lambda st: (st[1], st[0])):
        li = 0 if side == "L" else 1
        pin = np.asarray(pin, float)
        report["segments"] += 1
        for t in range(t0, t1 + 1):
            w = min(1.0, (t - t0 + 1) / max(edge, 1), (t1 - t + 1) / max(edge, 1)) if edge > 1 else 1.0
            solve(t, side, li, (1 - w) * ank[t, li] + w * pin)
        for k in range(1, release + 1):
            t = t1 + k
            if t >= T or t in owned[side] or speed[t] >= run_m:
                break
            w = 1.0 - k / (release + 1)
            w = w * w * (3.0 - 2.0 * w)
            solve(t, side, li, (1 - w) * ank[t, li] + w * pin)
            report["released"] += 1
    return out, report


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
              parents=SMPLX_BODY_PARENTS, x_init=None):
    """``(body_pose, miss_m)``: the hip and knee rows of ``side`` re-solved so the ankle's world xy lands on
    ``target_xy`` with the pelvis at ``pelvis_xy``; the rest of the pose untouched. The pull is toward the FITTED
    hip and knee; ``x_init`` (6: hip, knee rotvecs) starts the solve elsewhere -- the previous locked frame's
    solution, so consecutive frames settle in the same minimum of the null space instead of jittering between
    equivalent legs."""
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

    sol = least_squares(resid, x0 if x_init is None else np.asarray(x_init, float), method="lm", max_nfev=60)
    bp = bp0.copy()
    bp[hip], bp[knee] = sol.x[:3], sol.x[3:]
    miss = float(np.linalg.norm(resid(sol.x)[:2]))
    return bp, miss


def foot_lock_sequence(seq, betas, body_models_dir, *, moving_m: float = MOVING_M, stance_ratio: float = STANCE_RATIO,
                       min_stance: int = MIN_STANCE, max_stance: int = MAX_STANCE, pull: float = PULL, mode=None,
                       **rhythm_kw):
    """``(body_poses [T, 21, 3], report)`` for one id's per-frame ``(xy, body_pose, global_orient)``
    sequence (consecutive frames). ``report``: segments per leg, the metres each pinned foot was
    moved (the skate removed), and the solver misses. ``mode`` None = the module's MODE at call time: "dips"
    (the ankle-speed stances below) or "rhythm" (rhythm_stances, the jogging band); "off" returns the input."""
    mode = MODE if mode is None else str(mode)
    out0 = np.array([np.asarray(s[1], float).reshape(21, 3) for s in seq])
    if mode == "off" or len(seq) < 3:
        return out0, {"segments": 0, "frames": 0, "released": 0, "moved_m": [], "miss_m": []}
    rest, parents = load_smplx_skeleton(body_models_dir, betas=np.asarray(betas, float)[:10])
    if mode == "rhythm":
        st = rhythm_stances(seq, rest, parents, **{k: v for k, v in rhythm_kw.items() if k not in ("edge", "release")})
        return lock_stances(seq, rest, parents, st, pull=pull, edge=rhythm_kw.get("edge"),
                            release=rhythm_kw.get("release"), run_m=rhythm_kw.get("run_m"))
    if mode != "dips":
        raise ValueError(f"foot lock mode {mode!r}: off, dips or rhythm")
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


def engaged_flags(states_by_frame, team_of, *, engaged_m=None) -> dict:
    """``{(pid, frame): True}`` where an other-team body stands within ``engaged_m`` of the man (None = ENGAGED_M;
    an id whose team is unknown is never engaged). Empty when the gate is off (ENGAGED_M None)."""
    engaged_m = ENGAGED_M if engaged_m is None else engaged_m
    out: dict = {}
    if engaged_m is None or not team_of:
        return out
    for f, states in states_by_frame.items():
        rows = [(int(s.pid), team_of.get(int(s.pid)), np.asarray(s.xy, float)) for s in states]
        for pid, team, xy in rows:
            if team is None:
                continue
            for q, tq, xq in rows:
                if tq is not None and tq != team and float(np.linalg.norm(xy - xq)) <= float(engaged_m):
                    out[(pid, int(f))] = True
                    break
    return out


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    if i <= 0:
        return 0.0
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def engaged_flags_from_boxes(df, team_of, *, iou=None, cam=None) -> dict:
    """``{(pid, frame): True}`` where an other-team box in ``cam`` overlaps the man's box at IoU >= ``iou`` (None =
    ENGAGED_IOU / ENGAGED_CAM; empty when ENGAGED_IOU is None). ``df``: the tracks table (frame, cam,
    global_player_id, bbox_x1..y2); frames are that camera's own frames -- the sideline's are the play's."""
    iou = ENGAGED_IOU if iou is None else iou
    cam = ENGAGED_CAM if cam is None else str(cam)
    out: dict = {}
    if iou is None or df is None or not team_of:
        return out
    sub = df[df["cam"] == cam]
    cols = ["global_player_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    for f, g in sub.groupby("frame"):
        rows = [(int(r[0]), team_of.get(int(r[0])), tuple(float(v) for v in r[1:])) for r in g[cols].to_numpy()]
        for pid, team, box in rows:
            if team is None:
                continue
            for q, tq, bq in rows:
                if tq is not None and tq != team and _iou(box, bq) >= float(iou):
                    out[(pid, int(f))] = True
                    break
    return out


def foot_lock_timeline(tl, body_models_dir, *, lo=None, hi=None, team_of=None, engaged_m=None, boxes_df=None, **kw):
    """Apply :func:`foot_lock_sequence` to every id of a Timeline in place (frames lo..hi, all when
    None); returns ``{pid: report}``. Runs of consecutive drawn frames are locked separately. ``team_of``
    ``{pid: team}`` turns the engagement gate on (rhythm mode: no stance struck with an opponent within
    ENGAGED_M, nor -- with ``boxes_df``, the tracks table -- with an other-team box on the man's at ENGAGED_IOU)."""
    by: dict = {}
    for f, states in tl.states.items():
        if (lo is not None and f < lo) or (hi is not None and f > hi):
            continue
        for s in states:
            by.setdefault(int(s.pid), {})[int(f)] = s
    eng = engaged_flags(tl.states, team_of, engaged_m=engaged_m) if team_of else {}
    if team_of and boxes_df is not None:
        eng.update(engaged_flags_from_boxes(boxes_df, team_of))
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
        rep = {"segments": 0, "frames": 0, "released": 0, "moved_m": [], "miss_m": []}
        for run in runs:
            if len(run) < 3:
                continue
            seq = [(byf[f].xy, byf[f].body_pose, byf[f].global_orient) for f in run]
            run_kw = dict(kw)
            if eng:
                run_kw["engaged"] = np.array([eng.get((pid, f), False) for f in run])
            bps, r = foot_lock_sequence(seq, byf[run[0]].betas, body_models_dir, **run_kw)
            for f, bp in zip(run, bps):
                byf[f].body_pose = bp
            for k in ("segments", "frames", "released"):
                rep[k] += r.get(k, 0)
            rep["moved_m"] += r["moved_m"]
            rep["miss_m"] += r["miss_m"]
        reports[pid] = rep
    return reports
