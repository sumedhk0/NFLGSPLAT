"""A running gait for the legs of a moving body, phase-locked to the distance it covers.

WHY. Fitted legs skate: on play 1 (2026-09-16) the slower ankle of a moving body travels at 0.94 of
the pelvis speed and is planted on 1 % of moving frames, where a runner plants one foot about half
the time. The ankle keypoints carry no stance either (0.2-15 px jumps on a 100-px sprinter), so no
fit to them can plant a foot. This module synthesises the legs of a body that is running from the
one thing the footage does give well -- where the pelvis goes -- and leaves the fit everything
above the hips.

WHAT. For each id the pelvis path gives a speed per frame. Where the body moves faster than
``run_m`` per frame the gait is on: a phase advances by 2 pi per stride length ``L(v)`` of distance
travelled along the body's forward axis (a backpedal runs the cycle in reverse), the left leg at
phase ``phi`` and the right at ``phi + pi``. Per leg the cycle is: foot strike at phase 0 with the
hip flexed forward by the stance half-sweep ``A``; stance for a duty share ``d`` of the cycle, the
hip extending linearly to ``-A`` (the foot then stays where it struck, by construction, because
the foot's forward reach ``l sin(hip)`` runs linearly from ``+l sin A`` to ``-l sin A`` over ``d L``
metres of travel at leg length ``l``, so ``sin A = d L / (2 l)`` plants it exactly in the sagittal
plane); then swing, the hip returning to ``+A`` on a cosine while the knee flexes up to
``knee_swing`` at mid-swing; the knee holds ``knee_stance`` through stance. The hip and knee rows
of the fitted body_pose are replaced by the gait's, blended in and out over ``blend`` frames where
the gait switches on or off, so a man slowing to a stop hands his legs back to the fit. The ankle
rows level the foot with the turf in stance and drop the toes a little in swing.

Conventions (SMPL-X, verified on the real model 2026-09-16 with the hinge bounds): a knee flexes
about +x; a hip flexes forward about -x (the leg swings toward the body's +z, its forward).
"""
from __future__ import annotations

import numpy as np

RUN_M: float = 0.08            # pelvis speed (m per timeline frame) above which the legs run; 0.08 = 4.8 m/s at 60 fps
# Frames to cross-fade the gait in or out. 6 for its first four days; measured 2026-09-22 on play 1 v96 (09d over the
# live play, stride-2 sampling as rendered, the confidence gate off): hinge-angle jerk events over 25 deg/frame^2 --
# BLEND 6: 57 (legs 50); 16: 11 (legs 4); 30: 11 (legs 4); 16 with a smoothstep ramp: 13. The legs' sideways-bend share
# fell with it (L 7.9 -> 7.0 %, R 5.0 -> 3.5 %); the arms' 7 events are the fit's (the throw, 80 at 532). But the ramp
# is time the feet still skate (07l, planted share of moving frames: BLEND 6 22.6 %, 10 14 %, 16 6.7 %), so the default
# is 10 with MIN_ON 12 below (leg events 7, planted 14 %, joint jitter p99 1.06 -> 0.21). The first A/B of this knob
# measured two identical arms: it was bound as a default argument of gait_sequence -- every knob here is read at call
# time now.
BLEND: int = 10
# The on/off decision flickers where a runner's speed sits about RUN_M (the jogging band, 4-6 m/s, is where
# most of a pass play's men run), and every flicker freezes the phase and ramps the blend down and up again:
# the knees SNAP -- on a SYNTHETIC run. Measured 2026-09-22 (09d, stride-2 sampling as rendered): a synthetic
# run at a constant speed peaks at 16 deg/frame^2 of knee-angle jerk, the same run with the speed flickering
# +-0.006 about RUN_M peaks at 49 with 10 of 58 frames over 25; on play 1 v96 the live play counts 50 leg
# jerk events with the gait and 2 without. Hysteresis (off only below RUN_M * OFF_SHARE) and a minimum run
# (a run of either state shorter than MIN_RUN frames between longer runs takes its neighbours' state, the
# off gaps first) are here for that mechanism -- but on the PLAY OFF_SHARE 0.75 + MIN_RUN 12 measured leg
# jerk 50 -> 51 and the legs bent SIDEWAYS on 11.8 % of frames against 7.9 % (09d): a 3.6 m/s off-threshold
# re-admits the jogging band, where men shuffle and backpedal and the gait's legs swing in the motion plane
# across the body. So the defaults are the plain threshold (1.0 / 1); the play's flicker is not its jerk.
OFF_SHARE: float = 1.0
MIN_RUN: int = 1
# Where the play's leg jerk really is (2026-09-22, joints_v96a, 09d): 41 of the 50 events over 25 deg/frame^2
# sit within 8 frames of a 4.8 m/s crossing -- a man accelerates through RUN_M, the gait blends in from the
# fit's legs to its own phase-zero legs, he slows and it blends out again. A linear ramp over BLEND frames
# has a corner at each end whose jerk is the leg gap over the ramp's samples (a 60 deg knee gap over 3
# rendered samples = 20 deg/sample^2). BLEND_SHAPE "smooth" = a smoothstep ramp (C1 at both ends, 1.5x the
# slope mid-ramp); "linear" = the ramp as before. Read at call time by gait_sequence.
BLEND_SHAPE: str = "linear"
# A Gaussian over the gait's own rows (hips, knees, ankles) where the gait is on or blending, this many frames of
# sigma (0 = off): rounds the ramps' corners and the stance/swing joins without lengthening the ramp, which is time
# the feet still skate (BLEND 6 -> 16 halved the planted share, 07l). Read at call time by gait_sequence.
GAIT_SMOOTH_SIGMA: float = 0.0
# The ramp's jerk is the GAP between the fit's legs and the gait's at the switch (measured 2026-09-22: stretching the
# ramp (BLEND 16) or blurring it (GAIT_SMOOTH_SIGMA 3) removes the jerk and hands back the planting, 22.6 % -> 6.7 / 4 %).
# Two knobs on the gap itself, read at call time: MIN_ON drops an on-run shorter than this many frames (a crossing
# that is nothing but ramp), and only on-runs -- filling OFF gaps (MIN_RUN) put the gait on shufflers and lost;
# PHASE_MATCH starts every on-run at the phase whose hip flexions are nearest the fit's at the switch-on frame.
# Measured (09d leg events / 07l planted %): MIN_ON 12 at BLEND 6 -> 10 / 14; 18 -> 8 / --; with BLEND 10 -> 7 / 14, the
# default. PHASE_MATCH alone 41 (the hips match, the knees do not; the fit's legs at a crossing are not gait-like) and
# 14 with MIN_ON 12: off. A Gaussian over the gait rows (GAIT_SMOOTH_SIGMA 3) took the leg events to 0 and the planted
# share to 4 %: it rounds the stance sweep that plants the foot -- off.
MIN_ON: int = 12
PHASE_MATCH: bool = False
PHASE_GRID: int = 48
DUTY: float = 0.38             # a fixed duty for leg_angles' tests; gait_sequence uses duty_share(v)
LEG_M: float = 0.88            # hip-to-ankle, metres, for the stance sweep (a mean SMPL-X leg)
KNEE_STANCE: float = 0.30      # rad, the knee's flexion through stance
KNEE_SWING: float = 1.30       # rad, the knee's peak flexion in swing
HIP_ROW, KNEE_ROW, ANKLE_ROW = {"L": 0, "R": 1}, {"L": 3, "R": 4}, {"L": 6, "R": 7}
FOOT_SWING: float = 0.6        # share of the levelling ankle rotation kept in swing (the toes drop a little)
# The arms of a runner pump opposite the legs; the fitted arms of a sprinter flail (2026-09-18: joint
# jitter p90 on the running ids 0.19-0.24 m/frame^2 with the legs already synthesised). Where the gait is
# on, the shoulder and elbow rows go the same way: each arm hangs ARM_DOWN from the T-pose and swings
# ARM_SWING about the body's lateral axis in antiphase with its own leg, the elbow bent ELBOW_RUN.
# Conventions checked on the model (scratch probe): the drop is a rotation about z (-z left, +z right)
# and the swing a rotation about x applied AFTER it (R_x(swing) R_z(drop)); negative swings the arm
# forward; the elbow flexes forward about -y (left) / +y (right).
ARMS: bool = False             # opt-in until a render strip shows it: 07l's joint jitter does not move with the arms
                               # (p90 0.122 / p99 0.356 either way on play 1's play window, 2026-09-18) -- that ruler
                               # is orientation and leg-phase jitter, so the arms are judged on the footage
ARM_DOWN: float = 1.3
ARM_SWING: float = 0.55
ELBOW_RUN: float = 1.4
SHOULDER_ROW, ELBOW_ROW = {"L": 15, "R": 16}, {"L": 17, "R": 18}


def arm_rotvecs(phase: float, side: str, *, down: float = ARM_DOWN, swing: float = ARM_SWING, elbow: float = ELBOW_RUN):
    """``(shoulder, elbow)`` rotation vectors for one arm at its own leg's ``phase`` (0 = that foot's
    strike): the arm is back when its leg is forward and forward when the leg is back."""
    from scipy.spatial.transform import Rotation

    sgn = -1.0 if side == "L" else 1.0
    sw = float(swing) * float(np.cos(phase))                  # + = back (the leg is forward at phase 0)
    sh = (Rotation.from_rotvec([sw, 0.0, 0.0]) * Rotation.from_rotvec([0.0, 0.0, sgn * down])).as_rotvec()
    return sh, np.array([0.0, sgn * float(elbow), 0.0])


def stride_length(v: float) -> float:
    """Metres per full cycle of one leg (TWO steps) at ``v`` m per frame (60 fps): a walk of 0.02
    (1.2 m/s) cycles every 1.3 m, a sprint of 0.15 (9 m/s) every 4.6 m (steps of 2.3 m); linear
    between, clamped. The first cut used the step length here and ran the runner at 2.8 cycles a
    second (7 cycles in 148 frames), twice a sprinter's cadence."""
    return float(np.clip(1.3 + (v - 0.02) * (3.3 / 0.13), 1.0, 5.0))


def duty_share(v: float) -> float:
    """Share of the cycle a foot is on the ground at ``v`` m per frame: 0.6 walking, 0.4 jogging,
    0.22 sprinting (the flight phase grows with speed); linear, clamped."""
    return float(np.clip(0.6 - (v - 0.02) * (0.38 / 0.13), 0.22, 0.6))


def leg_angles(phase: float, amp: float, *, duty: float = DUTY, knee_stance: float = KNEE_STANCE,
               knee_swing: float = KNEE_SWING) -> tuple[float, float]:
    """``(hip_flex, knee_flex)`` in radians at ``phase`` (0 = foot strike, 2 pi = the next); hip
    positive = forward."""
    p = float(phase) % (2 * np.pi)
    stance_end = 2 * np.pi * duty
    if p < stance_end:                                        # stance: the foot stays put, so the hip's
        reach = np.sin(amp)                                   # forward reach (leg lengths) runs linearly
        hip = float(np.arcsin(reach - 2 * reach * (p / stance_end)))   # from +reach to -reach
        knee = knee_stance
    else:                                                     # swing: back to the front on a cosine
        s = (p - stance_end) / (2 * np.pi - stance_end)       # 0..1
        hip = -amp + 2 * amp * (0.5 - 0.5 * np.cos(np.pi * s))
        knee = knee_stance + (knee_swing - knee_stance) * np.sin(np.pi * s)
    return float(hip), float(knee)


def drop_short_on_runs(on, min_on: int):
    """``on`` with every run of True shorter than ``min_on`` frames, bounded by False on both sides, set False."""
    on = np.array(on, bool, copy=True)
    if min_on <= 1 or len(on) < 3:
        return on
    t = 0
    while t < len(on):
        if not on[t]:
            t += 1
            continue
        a = t
        while t < len(on) and on[t]:
            t += 1
        if t - a < min_on and a > 0 and t < len(on):
            on[a:t] = False
    return on


def on_flags(speed, *, run_m: float = RUN_M, off_share=None, min_run=None, min_on=None):
    """``on [T]``: the body runs. On above ``run_m``, off only below ``run_m * off_share`` (hysteresis), and a
    run of either state shorter than ``min_run`` frames with longer runs either side takes their state. None
    = the module's OFF_SHARE / MIN_RUN at call time; ``off_share`` 1 and ``min_run`` 1 = the plain threshold."""
    off_share = OFF_SHARE if off_share is None else float(off_share)
    min_run = MIN_RUN if min_run is None else int(min_run)
    spd = np.asarray(speed, float)
    on = np.zeros(len(spd), bool)
    cur = False
    lo = run_m * off_share
    for t in range(len(spd)):
        cur = (spd[t] > run_m) if not cur else (spd[t] > lo)
        on[t] = cur
    if min_run > 1 and len(on) > 2:
        for target in (False, True):
            t = 0
            while t < len(on):
                if on[t] != target:
                    t += 1
                    continue
                a = t
                while t < len(on) and on[t] == target:
                    t += 1
                if t - a < min_run and a > 0 and t < len(on):
                    on[a:t] = not target
    min_on = MIN_ON if min_on is None else int(min_on)
    return drop_short_on_runs(on, min_on)


def phases(forward_advance, speed, *, run_m: float = RUN_M, off_share=None, min_run=None, min_on=None, on=None,
           phi0=None):
    """``(phi [T], on [T])``: the gait phase per frame, advancing by 2 pi per stride of forward
    travel while the body runs (on_flags, or ``on`` given); held where it does not. ``phi0``: {frame: phase} --
    an on-run starting at that frame starts at that phase (phase matching) instead of carrying the held one."""
    adv = np.asarray(forward_advance, float)
    spd = np.asarray(speed, float)
    if on is None:
        on = on_flags(spd, run_m=run_m, off_share=off_share, min_run=min_run, min_on=min_on)
    on = np.asarray(on, bool)
    phi0 = phi0 or {}
    phi = np.zeros(len(adv))
    if on[0] and 0 in phi0:
        phi[0] = phi0[0]
    for t in range(1, len(adv)):
        if on[t] and not on[t - 1] and t in phi0:
            phi[t] = phi0[t]
        else:
            phi[t] = phi[t - 1] + (2 * np.pi * adv[t] / stride_length(spd[t]) if on[t] else 0.0)
    return phi, on


def fit_hip_flexion(hip_rotvec) -> float:
    """The fit's hip flexion, radians, positive forward: the thigh's rest direction (-y) rotated by the hip's
    axis-angle, its forward (+z) component against its down component. The gait's hip_rotvec puts flexion
    about -x in the plane of the motion; a fitted hip may also abduct and twist, which this ignores."""
    from scipy.spatial.transform import Rotation
    d = Rotation.from_rotvec(np.asarray(hip_rotvec, float)).apply([0.0, -1.0, 0.0])
    return float(np.arctan2(d[2], -d[1]))


def phase_match(flex_l: float, flex_r: float, amp: float, *, duty: float, grid: int = None) -> float:
    """The phase in [0, 2 pi) whose leg_angles hip flexions (left at phase, right at phase + pi) are nearest
    the fit's ``(flex_l, flex_r)`` in the least-squares sense over a grid of ``grid`` phases."""
    grid = PHASE_GRID if grid is None else int(grid)
    best, best_err = 0.0, None
    for k in range(grid):
        p = 2 * np.pi * k / grid
        hl, _ = leg_angles(p, amp, duty=duty)
        hr, _ = leg_angles(p + np.pi, amp, duty=duty)
        err = (hl - flex_l) ** 2 + (hr - flex_r) ** 2
        if best_err is None or err < best_err:
            best, best_err = p, err
    return best


def blend_weights(on, *, blend: int = BLEND, shape=None):
    """``w [T]`` in 0..1: 1 where the gait is on, ramping over ``blend`` frames at each edge -- linearly, or
    (``shape`` "smooth") along a smoothstep so the ramp meets 0 and 1 without a corner. None = BLEND_SHAPE."""
    shape = BLEND_SHAPE if shape is None else str(shape)
    on = np.asarray(on, bool)
    w = on.astype(float)
    if blend <= 1 or len(on) < 2:
        return w
    for t in range(1, len(on)):
        if on[t] and not on[t - 1]:                           # switching on: ramp up from t
            for k in range(blend):
                if t + k < len(on) and on[t + k]:
                    w[t + k] = min(w[t + k], (k + 1) / blend)
        if on[t - 1] and not on[t]:                           # switching off: ramp down before t
            for k in range(blend):
                if t - 1 - k >= 0 and on[t - 1 - k]:
                    w[t - 1 - k] = min(w[t - 1 - k], (k + 1) / blend)
    if shape == "smooth":
        w = w * w * (3.0 - 2.0 * w)
    elif shape != "linear":
        raise ValueError(f"blend shape {shape!r}: linear or smooth")
    return w


def forward_on_ground(global_orient):
    """The body's forward (+z of the pelvis) projected onto the ground, unit length (or None)."""
    from scipy.spatial.transform import Rotation

    f = Rotation.from_rotvec(np.asarray(global_orient, float)).apply([0.0, 0.0, 1.0])[:2]
    n = float(np.linalg.norm(f))
    return f / n if n > 1e-6 else None


def leg_yaw(forward, velocity):
    """``(yaw, advance)``: the signed angle (rad, positive toward the body's left) that turns the
    legs' plane from the body's forward onto the direction of motion, and the speed along that
    turned plane (negative when the body moves backwards along it). The legs run where the body
    goes -- on play 1's runner the fitted facing sat 45-90 deg off the velocity on 39 % of his moving
    frames and the legs swung sideways to his motion, planting nothing; a torso twisted off the
    line of running is what a cut looks like, but the feet still go where the man goes."""
    f = np.asarray(forward, float)
    v = np.asarray(velocity, float)
    sp = float(np.linalg.norm(v))
    if sp < 1e-9:
        return 0.0, 0.0
    left = np.array([-f[1], f[0]])
    yaw = float(np.arctan2(v @ left, v @ f))
    if abs(yaw) <= np.pi / 2:
        return yaw, sp                                      # legs turned onto the motion, running forward
    yaw = yaw - np.pi if yaw > 0 else yaw + np.pi            # moving backwards: legs face away, cycle backwards
    return yaw, -sp


def hip_rotvec(hip_flex: float, yaw: float):
    """Axis-angle of a hip flexed forward by ``hip_flex`` (about -x) in a leg plane turned by ``yaw``
    about the pelvis's up axis (+y)."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_rotvec([0.0, yaw, 0.0]) * Rotation.from_rotvec([-hip_flex, 0.0, 0.0])
    return r.as_rotvec()


def gait_sequence(seq, *, run_m=None, blend=None, duty=None, leg_m: float = LEG_M,
                  knee_stance: float = KNEE_STANCE, knee_swing: float = KNEE_SWING, off_share=None, min_run=None,
                  blend_shape=None, smooth_sigma=None, min_on=None, phase_match_on=None):
    """``(body_poses [T, 21, 3], report)`` for one id's consecutive ``(xy, body_pose, global_orient)``
    frames: the hip and knee rows replaced by the gait where the body runs, blended at the edges.
    ``report``: frames on, the phase advanced, the stride length range. ``run_m`` / ``blend`` None = the
    module's RUN_M / BLEND read HERE, at call time (a default argument binds the value at import, and a
    2026-09-22 A/B that set gait.BLEND measured two identical arms because of it)."""
    run_m = RUN_M if run_m is None else float(run_m)
    blend = BLEND if blend is None else int(blend)
    smooth_sigma = GAIT_SMOOTH_SIGMA if smooth_sigma is None else float(smooth_sigma)
    phase_match_on = PHASE_MATCH if phase_match_on is None else bool(phase_match_on)
    T = len(seq)
    xy = np.array([np.asarray(s[0], float) for s in seq])
    out = np.array([np.asarray(s[1], float).reshape(21, 3).copy() for s in seq])
    if T < 3:
        return out, {"on": 0, "cycles": 0.0}
    vel = np.zeros((T, 2))
    vel[1:-1] = (xy[2:] - xy[:-2]) / 2.0
    vel[0], vel[-1] = xy[1] - xy[0], xy[-1] - xy[-2]
    speed = np.linalg.norm(vel, axis=1)
    adv = np.zeros(T)
    yaw = np.zeros(T)
    for t in range(T):
        f = forward_on_ground(seq[t][2])
        if f is None:
            adv[t] = speed[t]
        else:
            yaw[t], adv[t] = leg_yaw(f, vel[t])
    on = on_flags(speed, run_m=run_m, off_share=off_share, min_run=min_run, min_on=min_on)
    phi0 = {}
    if phase_match_on:
        for t in range(T):
            if on[t] and (t == 0 or not on[t - 1]):
                L = stride_length(speed[t])
                d = duty_share(speed[t]) if duty is None else duty
                amp = float(np.arcsin(np.clip(d * L / (2.0 * leg_m), 0.0, 0.95)))
                phi0[t] = phase_match(fit_hip_flexion(out[t, HIP_ROW["L"]]), fit_hip_flexion(out[t, HIP_ROW["R"]]),
                                      amp, duty=d)
    phi, on = phases(adv, speed, run_m=run_m, on=on, phi0=phi0)
    w = blend_weights(on, blend=blend, shape=blend_shape)
    for t in range(T):
        if w[t] <= 0:
            continue
        L = stride_length(speed[t])
        d = duty_share(speed[t]) if duty is None else duty
        amp = float(np.arcsin(np.clip(d * L / (2.0 * leg_m), 0.0, 0.95)))
        for side, off in (("L", 0.0), ("R", np.pi)):
            hip, knee = leg_angles(phi[t] + off, amp, duty=d, knee_stance=knee_stance, knee_swing=knee_swing)
            g_hip = hip_rotvec(hip, yaw[t])                   # forward flexion about -x, in the plane of the motion
            g_knee = np.array([knee, 0.0, 0.0])
            # the foot: level with the turf in stance (the shin pitches by -hip + knee about x, the ankle
            # undoes it), most of the way there in swing so the toes drop a little
            in_stance = ((phi[t] + off) % (2 * np.pi)) < 2 * np.pi * d
            g_ankle = np.array([(hip - knee) * (1.0 if in_stance else FOOT_SWING), 0.0, 0.0])
            out[t, HIP_ROW[side]] = (1 - w[t]) * out[t, HIP_ROW[side]] + w[t] * g_hip
            out[t, KNEE_ROW[side]] = (1 - w[t]) * out[t, KNEE_ROW[side]] + w[t] * g_knee
            out[t, ANKLE_ROW[side]] = (1 - w[t]) * out[t, ANKLE_ROW[side]] + w[t] * g_ankle
            if ARMS:
                g_sh, g_el = arm_rotvecs(phi[t] + off, side)
                out[t, SHOULDER_ROW[side]] = (1 - w[t]) * out[t, SHOULDER_ROW[side]] + w[t] * g_sh
                out[t, ELBOW_ROW[side]] = (1 - w[t]) * out[t, ELBOW_ROW[side]] + w[t] * g_el
    if smooth_sigma > 0 and (w > 0).any() and T >= 3:
        from scipy.ndimage import binary_dilation, gaussian_filter1d
        reach = int(np.ceil(2 * smooth_sigma))
        touched = binary_dilation(w > 0, iterations=reach)          # (np.convolve "same" grows past a short sequence)
        rows = [r for s_ in ("L", "R") for r in (HIP_ROW[s_], KNEE_ROW[s_], ANKLE_ROW[s_])]
        sm = gaussian_filter1d(out[:, rows], smooth_sigma, axis=0, mode="nearest")
        out[np.ix_(touched, rows)] = sm[touched]
    return out, {"on": int(on.sum()), "cycles": float(abs(phi[-1] - phi[0]) / (2 * np.pi)),
                 "stride_m": (float(stride_length(speed[on].min())) if on.any() else 0.0,
                              float(stride_length(speed[on].max())) if on.any() else 0.0)}


def gait_timeline(tl, *, lo=None, hi=None, **kw):
    """Apply :func:`gait_sequence` to every id of a Timeline in place (frames lo..hi, all when None),
    one consecutive run of drawn frames at a time; returns ``{pid: report}``."""
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
        rep = {"on": 0, "cycles": 0.0}
        for run in runs:
            if len(run) < 3:
                continue
            seq = [(byf[f].xy, byf[f].body_pose, byf[f].global_orient) for f in run]
            bps, r = gait_sequence(seq, **kw)
            for f, bp in zip(run, bps):
                byf[f].body_pose = bp
            rep["on"] += r["on"]
            rep["cycles"] += r["cycles"]
        reports[pid] = rep
    return reports
