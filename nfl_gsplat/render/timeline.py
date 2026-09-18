"""A body for every detected player on every frame.

WHY. The world-mode render drew only players fused in both views with four
or more fused frames and a 21-frame track floor, from poses every sixth
frame: on play 1 that was a median of 14 bodies where 21-23 players were
detected, jumping in and out with the track fragments, at 10 frames per
second, and a spine tilt of 39 degrees median (falling over). This module
builds, per frame, the state of EVERY detected player -- ground position,
pose parameters, yaw -- from what the pipeline already has:

  position   both views' feet through their cameras, fused per frame, then
             zero-phase smoothed per player and gaps up to a limit filled
  pose       the fused refit (world) where it exists; else the sideline
             per-view pose turned into the world; else the play's MEDIAN
             pose (a real stance from the data, not a T-pose), facing the
             player's direction of travel. Interpolated to every frame:
             per-joint SLERP between posed frames, held at the ends
  upright    the world orientation is split into yaw about the vertical and
             a tilt off it; the tilt is clamped to MAX_TILT_DEG, the yaw kept

Fragments are not stitched here: a player whose id changes keeps a body
either side of the change; appearance continuity is a later problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

MAX_TILT_DEG: float = 35.0       # single-view poses: a lineman's stance; nobody stands past this
# Two-view poses (fused, or triangulated from keypoints) are measured, not
# guessed: on play 1 the triangulated refit tilts p50 16 deg, p90 38, none
# past 60 -- bent linemen and lunges, not bodies falling over. The 35 deg
# clamp built for monocular garbage trimmed 13 % of them.
MAX_TILT_TWO_VIEW_DEG: float = 60.0
# A man on the ground is not clamped upright: the sideline box wider than this share of its
# height says he lies (play 1's tackle at 640: ids 184 and 55 under boxes 0.46-0.58 as tall as
# wide, fitted standing at 24 deg and clamped there until 2026-09-17). 0.9 also caught crouched
# linemen pre-snap (ids 4 and 17 on 110 frames each, boxes 0.80); linemen's p5 is 0.87 and none
# go under 0.7, the tackled men all do.
LYING_ASPECT: float = 0.7
LYING_SIGMA_MULT: float | None = 3.0    # pose Gaussian sigma multiplier on lying frames (2026-09-18: lying-frame joint acceleration
                                        # p90 0.19 -> 0.08, the diving tackler 0.42 -> 0.09, one isolated case 0.08 -> 0.15)
LYING_BLEND: int = 3                    # frames the wider smoothing ramps in over, either side of a lying run
DUPLICATE_M: float = 0.9         # two ids closer than this on one frame are one player
# An id the endzone alone sees this frame is its unreconciled detection,
# poor along its depth axis (x). Within these distances of a kept state it
# is the endzone's copy of that player: play 2 drew six red bodies strung
# along x from one group of players. (The sideline's own detections are
# never deduped -- see dedupe_frames.)
ONE_VIEW_DEPTH_M: float = 4.0
# 1.0 was tried on play 1 evidence and REVERTED the same day: it recovers the linemen the sideline
# merges (Kansas City at the snap 8.9 -> 10.8, census 2.82 -> 1.20) but readmits play 2's ghosts.
# Measured on play 2: of 123 states 1.0 keeps that 1.5 drops, |dx| to the killer is p50 2.87 m
# (p90 3.52) -- id 2 on 53 frames at 3.46, id 34 at 2.90, id 1 at 2.87 -- which is the endzone's
# copy strung along its own depth axis, the exact failure this radius exists to prevent.
#
# The two populations do separate, just not on THIS axis. In the marginal band (|dy| 1.0-1.5):
#   play 2 ghosts        |dx| 2.4-3.5 m   the endzone's copy, displaced in depth
#   play 1 merged linemen |dx| 0.38-0.79 m  a different man standing beside him, same depth
# That suggested a depth-aware exception -- keep a state near in depth but clear across -- and it
# was measured on both plays and REJECTED: nothing on play 1 (snap 2.82 at near_depth 0.8/1.0/1.5,
# id 74 drawn on 3 frames against 66 under the blanket radius) and 54 states newly kept on play 2.
# The premise was a population error: 0.38-0.79 m is id 74's distance to the nearest SIDELINE
# BODY, not to the kept state this box actually compares against. Three replacements for this rule
# are now measured and rejected (same-body gate, blanket radius, depth-aware exception); measure
# the separation to the KILLER on both plays before proposing a fourth. See HANDOFF, 2026-09-13.
ONE_VIEW_ACROSS_M: float = 1.5
MAX_GAP_FRAMES: int = 30         # half a second of missing detections is bridged
# How long a detection gap the drawn body glides across. 30 until 2026-09-16: two bodies in play 1's
# pile glided 3.15 m (id 153, 435-454) and 2.13 m (id 4, 444-457) with nobody seeing them. Measured
# on the live play: bridge 10 draws 46 fewer body-frames, 40 fewer of them unseen by either camera
# (93 -> 53), census 1.50 -> 1.32, root jitter p90 0.0120 -> 0.0116, live hops 4 -> 4, steps 5 -> 5,
# pops (a body vanishing and reappearing) 34 -> 32; bridge 4 costs hops 6 and steps 12. The span
# and anchoring rules keep MAX_GAP_FRAMES.
FILL_GAP_FRAMES: int = 10
# A body interpolated through a detection gap is anchored (its player was
# seen within MAX_GAP_FRAMES) -- unless it stands on top of a body the
# sideline detects in that frame: then it is the same player under a second
# fragment id. Measured on play 1 v23 (visible vanishes / double bodies per
# rendered frame-pair): the plain rule 155 / 108, anchoring alone 39 / 360,
# anchoring with this radius 46 / 123 at 0.4 m, 55 / 121 at 0.5.
INTERP_DUP_M: float = 0.4
MIN_FRAMES: int = 6              # shorter fragments are noise
# A SHORT fragment that spends most of its life within arm's reach of another body of its own team
# is that body's second copy: play 1's id 162 (2026-09-16), 19 drawn frames of track 19's tail
# after the 08t cut, whose own detections flip between a Chiefs lineman and the Baltimore man beside
# him (the footage strip shows the keypoints on each in turn) -- a hop under 08t's 1.5 m floor, and
# too few ankle frames for the twin test. Measured on play 1's live play: with these thresholds only
# 162 qualifies (53 % of its frames within 0.6 m of id 11 / 165; the next fragments 0-38 %), and
# dropping it costs the census nothing. A rule with one instance so far; the thresholds are here to
# be re-measured on the next play, not tuned to this one.
RIDER_MAX_FRAMES: int = 40
RIDER_M: float = 0.6
RIDER_SHARE: float = 0.5
VEL_WINDOW: int = 12             # frames over which yaw follows the travel direction
# Poses as rendered still shake: hands and feet move 27 cm between rendered
# frames at the p90 in second differences (play 1 v23, fused and one-view
# alike) where a sprinting limb's real second difference is a few cm. A
# zero-phase moving average over the interpolated axis-angles, this many
# source frames wide (0 = off); measured 2026-09-08, see HANDOFF v24.
POSE_SMOOTH_FRAMES: int = 7          # a moving median (see smooth_axis_angles); 9 was the mean's window
# Measured on play 1 with the sole-on-turf fits (limbs' reprojection p50 / jitter p50):
# runner median-7 17.1 px / 3.9 m/s, adaptive 14.7 / 4.4; lineman 2.3 px / 0.70 either way.
POSE_SMOOTH_RANGE_RAD: float = 0.5
# The body_pose smoother is a GAUSSIAN now, this many frames of sigma (0 = off), applied to the
# interpolated axis-angles with no range gate; the median above still smooths the orientation.
# Measured 2026-09-15 on play 1's live play (30 ids, the renderer's own forward pass): the raw
# stride-2 fits carry the jitter themselves (wrists' second difference p90 0.12 m/frame^2 at the
# keyframes, 0.12 drawn), and a median cannot remove noise that is on every frame -- median 7
# gated (the previous smoother) read jitter p90 0.231 against raw 0.261, median 15 ungated 0.159.
# Gaussian sigma 2: p50 0.035 -> 0.019, p90 0.261 -> 0.109, p99 1.38 -> 0.69 (max over joints,
# pelvis-relative), for +0.4 px on the limbs' reprojection p50 in the sideline (8.6 -> 9.0, inside
# the detector's noise) and +1.9 px on the endzone p90 (12.3 -> 14.2). Sigma 4 halves the jitter
# again (p90 0.078) but costs +1.3 px p50 and +4.4 px endzone p90: that is smear. See HANDOFF.
POSE_SMOOTH_SIGMA: float = 2.0
# The orientation kept the 7-frame median after the limbs got their Gaussian, unmeasured. Measured
# 2026-09-16 on play 1's live play (all drawn ids, joint jitter max over joints, yaw second difference,
# limbs reprojected in both cameras): median 7 -> gauss sigma 4 takes joint jitter p90 0.090 -> 0.058
# and p99 0.44 -> 0.27, yaw jitter p90 2.7 -> 0.5 deg/frame^2, for +1.0 px on the sideline limbs' p90
# and +1.1 px on the endzone's (p50s unchanged, steps and census unchanged); sigma 2 buys 0.062 / 0.41
# at +0.6 / +0.8 px. The tail is where the twitching lives, so sigma 4. 0 = the median as before.
ORIENT_SMOOTH_SIGMA: float = 4.0
# Joint limits on the four hinges, applied to the interpolated axis-angles BEFORE the Gaussian so
# the smoother rounds the kinks. body_pose rows (joint - 1), the hinge axis, and the sign that makes
# flexion positive in SMPL-X's rest pose: knees flex about +x; elbows about y, right +, left -.
# Measured on play 1's drawn live play (2026-09-15, 3676 body-frames): knees hyperextended past -15 deg
# on 3 % of frames (min -104), elbows on 4-6 % (min -136); knees bent SIDEWAYS past 35 deg on 12 %,
# elbows on 14 % -- on the ids whose limbs jitter most, i.e. where the fit is garbage. The clamp
# (flexion to [-5, 150], off-axis to 25 deg) takes every one to zero for +0.2 px on the limbs'
# sideline reprojection p90 and +1.7 px on the endzone's, p50 unchanged. Clamping the COLLARS too
# (they turn 150-176 deg at the p99) was measured and REJECTED: +2.8 / +5.2 px at the p90 -- the
# regressor places the whole arm through the clavicle, and the limit moves the arm off the keypoints.
HINGES: dict = {"L_knee": (3, 0, +1.0), "R_knee": (4, 0, +1.0), "L_elbow": (17, 1, -1.0), "R_elbow": (18, 1, +1.0)}
HINGE_FLEX_MIN_DEG: float = -5.0
HINGE_FLEX_MAX_DEG: float = 150.0
HINGE_OFF_AXIS_MAX_DEG: float = 25.0
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class PlayerState:
    pid: int
    xy: np.ndarray               # [2] ground, metres
    body_pose: np.ndarray        # [21, 3]
    global_orient: np.ndarray    # [3] world axis-angle (upright-clamped)
    betas: np.ndarray            # [10]
    source: str                  # fused | sideline | default
    clamped: bool = False
    views: tuple = ("sideline", "endzone")   # which cameras saw this id on this frame


@dataclass
class Timeline:
    frames: list[int]
    states: dict[int, list[PlayerState]] = field(default_factory=dict)
    n_clamped: int = 0
    n_default: int = 0
    n_duplicates: int = 0
    members: dict = field(default_factory=dict)   # player id -> member ids (after stitching)
    held: set = field(default_factory=set)        # (pid, frame) drawn at a held spot with no fit of its own (the quarterback under centre)


# ---- orientation helpers ---------------------------------------------------

def body_up(global_orient) -> np.ndarray:
    """World direction of the body's up axis (SMPL-X canonical +y)."""
    return Rotation.from_rotvec(np.asarray(global_orient, float)).apply([0.0, 1.0, 0.0])


def tilt_deg(global_orient) -> float:
    u = body_up(global_orient)
    return float(np.degrees(np.arccos(np.clip(u @ UP, -1.0, 1.0))))


def clamp_tilt(global_orient, max_tilt_deg: float = MAX_TILT_DEG):
    """``(orient, clamped)``: the same yaw, tilt reduced to ``max_tilt_deg``.

    The correction is the smallest rotation that brings the body's up axis
    to the cone, applied on the left (in the world), so the facing direction
    survives."""
    r = Rotation.from_rotvec(np.asarray(global_orient, float))
    u = r.apply([0.0, 1.0, 0.0])
    ang = np.degrees(np.arccos(np.clip(u @ UP, -1.0, 1.0)))
    if ang <= max_tilt_deg:
        return np.asarray(global_orient, float), False
    axis = np.cross(u, UP)
    n = np.linalg.norm(axis)
    if n < 1e-9:                                  # upside down: fall back to yaw only
        return upright_from_yaw(yaw_of(global_orient)), True
    fix = Rotation.from_rotvec(axis / n * np.radians(ang - max_tilt_deg))
    return (fix * r).as_rotvec(), True


def yaw_of(global_orient) -> float:
    """Facing direction on the ground: the body's forward (+z) projected."""
    f = Rotation.from_rotvec(np.asarray(global_orient, float)).apply([0.0, 0.0, 1.0])
    return float(np.arctan2(f[1], f[0]))


def upright_from_yaw(yaw: float) -> np.ndarray:
    """An upright body (canonical y up -> world z up) facing ``yaw``."""
    stand = Rotation.from_euler("x", 90.0, degrees=True)           # y -> z, and forward +z -> -y
    turn = Rotation.from_rotvec([0.0, 0.0, yaw + np.pi / 2.0])      # -y (-90 deg) -> yaw
    return (turn * stand).as_rotvec()


# ---- interpolation ---------------------------------------------------------

def interp_axis_angle(frames_known, values_known, frames_out):
    """Per-joint SLERP of ``[N, J, 3]`` axis-angle rows at ``frames_known``
    onto ``frames_out``, held constant beyond the ends."""
    fk = np.asarray(frames_known, float)
    vk = np.asarray(values_known, float)
    fo = np.asarray(frames_out, float)
    if vk.ndim == 2:
        vk = vk[:, None, :]
    if len(fk) == 1:
        return np.repeat(vk, len(fo), axis=0)
    fo_c = np.clip(fo, fk[0], fk[-1])
    out = np.empty((len(fo), vk.shape[1], 3))
    for j in range(vk.shape[1]):
        out[:, j] = Slerp(fk, Rotation.from_rotvec(vk[:, j]))(fo_c).as_rotvec()
    return out


def smooth_xy(xy, *, window: int = 9):
    """Zero-phase moving average with edge handling; NaN rows stay NaN.

    Smoothed within each CONTIGUOUS run of finite rows, never across a gap. The
    earlier version compacted every finite row into one array before convolving,
    so a segment's last ``window // 2`` frames were averaged with the first
    frames of the next segment -- 77 frames and metres away on play 1's id 21,
    whose stationary body marched 0.88 m/frame for four frames toward where its
    track resumed (2026-09-15). Every sparse track did this at every long gap.
    """
    xy = np.asarray(xy, float)
    out = xy.copy()
    ok = np.isfinite(xy).all(1)
    if ok.sum() < 3:
        return out
    idx = np.flatnonzero(ok)
    # split the finite rows into runs of consecutive indices; gaps already filled by
    # fill_gaps are contiguous here, anything longer than its max_gap is a break
    breaks = np.flatnonzero(np.diff(idx) > 1)
    runs = np.split(idx, breaks + 1)
    for run in runs:
        if len(run) < 3:
            continue
        for d in range(2):
            v = xy[run, d]
            k = min(window, len(v) if len(v) % 2 else len(v) - 1)
            if k < 3:
                continue
            pad = k // 2
            vp = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
            out[run, d] = np.convolve(vp, np.ones(k) / k, mode="valid")
    return out


def smooth_axis_angles(seq, *, window: int = POSE_SMOOTH_FRAMES):
    """Moving MEDIAN along axis 0 of ``seq [T, ...]`` (axis-angle vectors per
    joint), edges held. ``window`` <= 1 returns the input.

    A moving mean (v24-v27) smeared fast limbs: on play 1's runner a 9-frame
    mean took the legs' reprojection from 19 to 25 px at the median (44 px at
    the p90) and swung a leg out sideways where the stride turned, while it
    halved a lineman's jitter (limb speed in the body frame 1.24 -> 0.50 m/s).
    The median keeps the runner where the fit put him (19 px, p90 37) and
    still takes the lineman to 0.66 m/s (window 7); a one-frame flip is
    dropped rather than blended in. Where a component turns more than
    POSE_SMOOTH_RANGE_RAD within the window the raw value stays (the
    runner's arms: 17.1 -> 14.7 px, the lineman unchanged)."""
    from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d

    a = np.asarray(seq, float)
    if window is None or window <= 1 or len(a) < 3:
        return a
    k = min(int(window), len(a) if len(a) % 2 else len(a) - 1)
    if k < 3:
        return a
    shape = a.shape
    flat = a.reshape(len(a), -1)
    med = median_filter(flat, size=(k, 1), mode="nearest")
    # a component that turns more than POSE_SMOOTH_RANGE_RAD within the window is real
    # motion (a runner's arm), kept raw; the median holds only where the window is quiet
    rng = maximum_filter1d(flat, k, axis=0, mode="nearest") - minimum_filter1d(flat, k, axis=0, mode="nearest")
    out = np.where(rng <= POSE_SMOOTH_RANGE_RAD, med, flat)
    return out.reshape(shape)


def unwrap_axis_angles(seq):
    """``seq [T, ...]`` axis-angle vectors with each row re-expressed so that neighbours are close in
    the vector space: a rotation by ``a`` about ``u`` is also a rotation by ``2 pi - a`` about ``-u``,
    and scipy's ``as_rotvec`` (the keyframe SLERP's output) always returns the one under pi. A body
    turning THROUGH a half turn therefore flips sign between two frames, and any component-wise
    smoother -- median or Gaussian -- then mixes antipodal vectors into a garbage rotation for a
    window's worth of frames (play 1, id 0: legs splayed for 12 frames at the snap while the
    cache's own legs sat 4 px on the keypoints). Per row the representation nearer the previous
    unwrapped row is kept; the rotations themselves are unchanged."""
    a = np.array(seq, float, copy=True)
    if a.ndim < 2 or len(a) < 2:
        return a
    flat = a.reshape(len(a), -1, 3)
    for j in range(flat.shape[1]):
        prev = flat[0, j]
        for i in range(1, len(flat)):
            v = flat[i, j]
            n = np.linalg.norm(v)
            if n > 1e-9:
                alt = v * (1.0 - 2.0 * np.pi / n)
                if np.sum((alt - prev) ** 2) < np.sum((v - prev) ** 2):
                    flat[i, j] = alt
            prev = flat[i, j]
    return flat.reshape(a.shape)


def smooth_axis_angles_gaussian(seq, *, sigma: float = POSE_SMOOTH_SIGMA):
    """Gaussian along axis 0 of ``seq [T, ...]`` (axis-angle vectors per joint), edges held;
    ``sigma`` <= 0 returns the input.

    Why not the median: the per-frame fits are noisy on EVERY frame at the extremities (a hand is
    five pixels), and a median only drops isolated spikes -- on play 1 it left the limbs' jitter
    where it found it (see POSE_SMOOTH_SIGMA). A Gaussian averages the noise down while a limb's
    real swing, which is slow beside a 2-frame sigma, passes through: the runner's arm turns
    ~0.1 rad/frame and loses under 3 % of its amplitude at sigma 2. Component-wise on the
    axis-angle vectors, which is exact for small differences between neighbouring frames and
    safe for joints that never approach a half turn; the median took the same view."""
    a = np.asarray(seq, float)
    if sigma is None or sigma <= 0 or len(a) < 3:
        return a
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(a, float(sigma), axis=0, mode="nearest")


def clamp_hinges(seq, *, hinges=None, flex_min_deg: float = HINGE_FLEX_MIN_DEG,
                 flex_max_deg: float = HINGE_FLEX_MAX_DEG, off_max_deg: float = HINGE_OFF_AXIS_MAX_DEG):
    """``seq [T, 21, 3]`` with each hinge's flexion clipped to [flex_min, flex_max] degrees about its
    axis and its two off-axis components scaled down to at most ``off_max`` degrees together. The
    components of an axis-angle vector are not Euler angles, but for a hinge that never nears a half
    turn the split is close enough to name a backwards or sideways knee, which is all this does.
    Every other joint is returned as it came (see HINGES for why the collars are not here)."""
    out = np.array(seq, float)
    if out.ndim != 3 or out.shape[1] < 19:
        return out
    off_max = np.radians(off_max_deg)
    for j, ax, sign in (hinges or HINGES).values():
        v = out[:, j]
        v[:, ax] = sign * np.clip(sign * v[:, ax], np.radians(flex_min_deg), np.radians(flex_max_deg))
        others = [a for a in range(3) if a != ax]
        mag = np.linalg.norm(v[:, others], axis=1)
        scale = np.where(mag > off_max, off_max / np.maximum(mag, 1e-9), 1.0)
        v[:, others] *= scale[:, None]
    return out


DESPIKE_HALF: int = 2            # neighbours either side the despike's median is taken over
DESPIKE_M: float | None = 0.15   # a frame's xy farther than this from the median of its neighbours (+-2) is a
                                 # measurement spike and takes that median (2026-09-17: live hops 23 -> 2, reprojection unchanged)


def despike_xy(xy, *, excess_m: float | None = DESPIKE_M, half: int = 2):
    """Single-frame position spikes removed BEFORE the smoother: a row farther than ``excess_m``
    from the median of its ``half`` neighbours either side takes that median. The Gaussian
    spreads a spike into a hop over its window; a median sees a spike as the odd one out and a
    real cut (every later frame moves the same way) as the trend. NaN rows are left alone."""
    xy = np.asarray(xy, float).copy()
    if excess_m is None or len(xy) < 2 * half + 1:
        return xy
    ok = np.isfinite(xy).all(1)
    out = xy.copy()
    for i in range(len(xy)):
        if not ok[i]:
            continue
        lo, hi = max(0, i - half), min(len(xy), i + half + 1)
        before = [j for j in range(lo, i) if ok[j]]
        after = [j for j in range(i + 1, hi) if ok[j]]
        if len(before) < half or len(after) < half:
            continue                        # an uneven window carries the trend, not a verdict on this frame
        med = np.median(xy[before + after], axis=0)
        if np.linalg.norm(xy[i] - med) > excess_m:
            out[i] = med
    return out


def fill_gaps(frames, xy, *, max_gap: int = FILL_GAP_FRAMES):
    """Linear fill of NaN rows between known rows when the gap is short."""
    xy = np.asarray(xy, float).copy()
    ok = np.isfinite(xy).all(1)
    idx = np.flatnonzero(ok)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and (frames[b] - frames[a]) <= max_gap:
            w = np.linspace(0, 1, b - a + 1)[1:-1, None]
            xy[a + 1:b] = (1 - w) * xy[a] + w * xy[b]
    return xy


YAW_SMOOTH: int = 5              # frames of circular smoothing on a motion-derived heading


def yaw_from_motion(xy, *, window: int = VEL_WINDOW, fallback: float = 0.0, smooth: int = YAW_SMOOTH):
    """Facing from the direction of travel, per row; ``fallback`` when still."""
    xy = np.asarray(xy, float)
    n = len(xy)
    yaw = np.full(n, fallback)
    for i in range(n):
        a, b = max(0, i - window // 2), min(n - 1, i + window // 2)
        d = xy[b] - xy[a]
        if np.isfinite(d).all() and np.linalg.norm(d) > 0.3:
            yaw[i] = float(np.arctan2(d[1], d[0]))
    # hold the last known heading through still stretches ...
    known = [i for i in range(n) if yaw[i] != fallback]
    last = fallback
    for i in range(n):
        if yaw[i] == fallback and i > 0:
            yaw[i] = last
        last = yaw[i]
    # ... and face the first known heading before it: a default-posed body that starts still and
    # then runs turned from the fallback to its heading in one frame (play 1 id 66 at 582, a joint
    # jump of 0.84 m; 2026-09-17)
    if known and known[0] > 0:
        yaw[: known[0]] = yaw[known[0]]
    if smooth > 1 and known:
        # circular moving mean over ``smooth`` frames: the heading of a runner turns, it does not step
        c, s_ = np.cos(yaw), np.sin(yaw)
        k = np.ones(smooth) / smooth
        cs = np.convolve(np.pad(c, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, mode="valid")
        ss = np.convolve(np.pad(s_, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, mode="valid")
        yaw = np.arctan2(ss, cs)
    return yaw


# ---- stitching ------------------------------------------------------------------

def relabel(ground_by_frame, views_by_frame, poses_by_pid, player_of):
    """Merge ids under ``player_of`` (id -> player id, from tracking.stitch).

    Positions: one per player per frame (mean when two members share a
    frame); views: the union; poses: fused wins over single-view, else the
    earlier id. Returns ``(ground, views, poses, members)`` with ``members``
    mapping player id -> sorted member ids, so a texture fitted to any
    member can dress the player."""
    rank = {"fused": 0, "sideline": 1, "default": 2}
    members: dict[int, list] = {}
    ground, views, poses = {}, {}, {}
    for f, d in ground_by_frame.items():
        acc: dict[int, list] = {}
        vs: dict[int, set] = {}
        for pid, xy in d.items():
            new = int(player_of.get(pid, pid))
            members.setdefault(new, set()).add(int(pid))
            acc.setdefault(new, []).append(np.asarray(xy, float))
            if views_by_frame and pid in views_by_frame.get(f, {}):
                vs.setdefault(new, set()).update(views_by_frame[f][pid])
        ground[f] = {new: np.mean(v, axis=0) for new, v in acc.items()}
        if views_by_frame:
            views[f] = {new: tuple(sorted(v)) for new, v in vs.items()}
    for pid, byf in poses_by_pid.items():
        new = int(player_of.get(pid, pid))
        members.setdefault(new, set()).add(int(pid))
        dst = poses.setdefault(new, {})
        for f, rec in byf.items():
            if f not in dst or (rank.get(rec[3], 3), pid) < (rank.get(dst[f][3], 3), dst[f][4]):
                dst[f] = (rec[0], rec[1], rec[2], rec[3], int(pid))
    poses = {pid: {f: rec[:4] for f, rec in byf.items()} for pid, byf in poses.items()}
    return ground, (views if views_by_frame else None), poses, {k: sorted(v) for k, v in members.items()}


def rider_ids(tl: "Timeline", team_of: dict, *, max_frames: int = RIDER_MAX_FRAMES, ride_m: float = RIDER_M,
              share: float = RIDER_SHARE) -> set:
    """Ids drawn on at most ``max_frames`` frames that stand within ``ride_m`` of another body of the
    same team on at least ``share`` of them (see RIDER_MAX_FRAMES)."""
    by_id: dict = {}
    for f, states in tl.states.items():
        for s in states:
            by_id.setdefault(int(s.pid), []).append((int(f), np.asarray(s.xy, float)))
    out = set()
    for pid, rows in by_id.items():
        if len(rows) > max_frames or len(rows) < 2:
            continue
        team = team_of.get(pid)
        if team is None:
            continue
        near = 0
        for f, xy in rows:
            d = [float(np.linalg.norm(np.asarray(o.xy, float) - xy)) for o in tl.states.get(f, ())
                 if int(o.pid) != pid and team_of.get(int(o.pid)) == team]
            near += bool(d) and min(d) < ride_m
        if near / len(rows) >= share:
            out.add(pid)
    return out


def orphan_ids(tl: "Timeline", team_of: dict, *, max_frames: int = RIDER_MAX_FRAMES) -> set:
    """Ids with no team drawn on at most ``max_frames`` frames. A fragment nobody could team is a
    few detections of a man some other id already draws (play 1's 203: eight frames on the Kansas
    City line, rendered in the default kit, one of the seven live-play hops); the rider rule cannot
    reach it because it matches by team. A teamless id drawn for longer is a real unidentified
    player and stays (his kit is the renderer's problem, not the timeline's)."""
    frames: dict = {}
    for f, states in tl.states.items():
        for s in states:
            frames[int(s.pid)] = frames.get(int(s.pid), 0) + 1
    return {pid for pid, n in frames.items() if n <= max_frames and not team_of.get(pid)}


TWIN_M: float = 0.2          # two drawn bodies this close are one man
TWIN_MIN_RUN: int = 8        # ... when it lasts this many consecutive frames


WEAK_KIT_MARGIN: float | None = 0.2    # a short fragment whose kit reads under this (|median margin|), with no jersey
                                       # and no role, wears a guessed team: left out (2026-09-18: BAL 11.21 -> 10.98, exactly-eleven 41 -> 49)


def unreadable_kit_ids(tl: "Timeline", df, named: dict, *, margin: float = 0.2, max_frames: int = RIDER_MAX_FRAMES,
                       cam: str = "sideline") -> set:
    """Ids drawn on at most ``max_frames`` frames whose ``cam`` kit margin reads under ``margin`` in
    median (the torso saturation could not say which kit) and which carry neither a jersey number
    nor a role (``named``: pid -> True when identity knows the man): their team label is a guess,
    and play 1's 201 (26 frames, margin -0.05) stood as a white body on Kansas City linemen's legs."""
    sub = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    km = sub.groupby("global_player_id")["kit_margin"].median().to_dict()
    frames: dict = {}
    for f, states in tl.states.items():
        for st in states:
            frames[int(st.pid)] = frames.get(int(st.pid), 0) + 1
    out = set()
    for pid, n in frames.items():
        if n > max_frames or named.get(pid):
            continue
        m = km.get(pid)
        if m is not None and np.isfinite(m) and abs(float(m)) < margin:
            out.add(pid)
    return out


def twin_frames(tl: "Timeline", team_of: dict, *, twin_m: float = TWIN_M, min_run: int = TWIN_MIN_RUN) -> set:
    """``{(frame, pid)}`` to drop: for every same-team pair of drawn ids within ``twin_m`` of each other
    on at least ``min_run`` CONSECUTIVE frames, the id drawn on fewer frames overall loses those frames.
    Two sideline boxes 0.13-0.15 m apart on one man for twenty frames, each drawn as a body (play 1,
    2026-09-16: 166-204 and 1-40), are the twins 08o's ankle-ray test (0.03-0.11 m) left; a pile
    stands 0.27 m and up. The rider rule handles short fragments; this one handles long ids."""
    frames_of: dict = {}
    for f, states in tl.states.items():
        for s in states:
            frames_of[int(s.pid)] = frames_of.get(int(s.pid), 0) + 1
    close: dict = {}
    for f, states in tl.states.items():
        for i, a in enumerate(states):
            for b in states[i + 1:]:
                pa, pb = int(a.pid), int(b.pid)
                ta, tb = team_of.get(pa), team_of.get(pb)
                if ta is None or ta != tb:
                    continue
                if float(np.hypot(*(np.asarray(a.xy, float) - np.asarray(b.xy, float)))) <= twin_m:
                    close.setdefault((min(pa, pb), max(pa, pb)), []).append(int(f))
    drop: set = set()
    for (pa, pb), fs in close.items():
        fs = sorted(set(fs))
        loser = pa if frames_of.get(pa, 0) < frames_of.get(pb, 0) else pb
        run = [fs[0]]
        for f in fs[1:] + [None]:
            if f is not None and f == run[-1] + 1:
                run.append(f)
                continue
            if len(run) >= min_run:
                drop.update((g, loser) for g in run)
            run = [f] if f is not None else []
    return drop


BOX_TWIN_IOU: float | None = 0.6    # sideline boxes of two same-team ids overlapping by this much are one man (2026-09-17: with
                                    # the despike in, BAL 11.31 -> 11.21 on the live window, pile pairs 18 -> 16, hops 2 -> 2)
BOX_TWIN_MIN_RUN: int = 8


def box_twin_frames(tl: "Timeline", df, team_of: dict, *, iou_min: float = 0.6, min_run: int = BOX_TWIN_MIN_RUN,
                    cam: str = "sideline") -> set:
    """``{(frame, pid)}`` to drop: two same-team drawn ids whose ``cam`` boxes overlap by IoU >= ``iou_min``
    on at least ``min_run`` consecutive frames are one man under two tracker ids; the id with fewer
    ``cam`` boxes overall loses those frames. Engaged linemen overlap at ~0.4 in the side-on view;
    a tracker's second id on the same man sits at 0.85-0.97 (play 1, 2026-09-17)."""
    sub = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    box = {(int(r.frame), int(r.global_player_id)): (float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2))
           for r in sub.itertuples()}
    n_boxes = sub.groupby("global_player_id").size().to_dict()

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    hits: dict = {}
    for f, states in tl.states.items():
        pids = [int(s.pid) for s in states if (int(f), int(s.pid)) in box]
        for i, pa in enumerate(pids):
            for pb in pids[i + 1:]:
                ta, tb = team_of.get(pa), team_of.get(pb)
                if not ta or ta != tb:
                    continue
                if iou(box[(int(f), pa)], box[(int(f), pb)]) >= iou_min:
                    hits.setdefault((min(pa, pb), max(pa, pb)), []).append(int(f))
    drop: set = set()
    for (pa, pb), fs in hits.items():
        fs = sorted(set(fs))
        loser = pa if n_boxes.get(pa, 0) < n_boxes.get(pb, 0) else pb
        run = [fs[0]]
        for f in fs[1:] + [None]:
            if f is not None and f == run[-1] + 1:
                run.append(f)
                continue
            if len(run) >= min_run:
                drop.update((g, loser) for g in run)
            run = [f] if f is not None else []
    return drop


IMPOSSIBLE_M: float | None = None   # a drawn step past this per frame (0.2 = 12 m/s, past any player) held for
IMPOSSIBLE_RUN: int = 4             # IMPOSSIBLE_RUN frames is a tracker switch smoothed into a glide (off until measured)


def impossible_runs(tl: "Timeline", *, max_m: float = 0.2, min_run: int = IMPOSSIBLE_RUN) -> set:
    """``{(frame, pid)}``: frames inside a run of at least ``min_run`` consecutive drawn steps longer
    than ``max_m`` (metres per frame). No player covers 12 m/s for four frames; a body that does is
    two men under one id joined by the smoother (play 1's 185 at 650-657: 0.33-0.40 m/frame for eight
    frames, three metres across the tackle). The frames of the run go, including its far end."""
    by: dict = {}
    for f, states in tl.states.items():
        for st in states:
            by.setdefault(int(st.pid), {})[int(f)] = np.asarray(st.xy[:2], float)
    drop: set = set()
    for pid, byf in by.items():
        fs = sorted(byf)
        run: list = []
        for a, b in zip(fs, fs[1:] + [None]):
            fast = b is not None and b - a == 1 and float(np.linalg.norm(byf[b] - byf[a])) > max_m
            if fast:
                run.append(a)
                continue
            if len(run) >= min_run:
                drop.update((g, pid) for g in run + [run[-1] + 1])
            run = []
    return drop


def drop_frames(tl: "Timeline", pairs: set) -> int:
    """Remove the states named by ``pairs`` ``{(frame, pid)}``; returns the number removed."""
    n = 0
    for f, pid in pairs:
        before = len(tl.states.get(f, ()))
        tl.states[f] = [s for s in tl.states.get(f, ()) if int(s.pid) != int(pid)]
        n += before - len(tl.states[f])
    return n


def drop_ids(tl: "Timeline", ids: set) -> int:
    """Remove every state of ``ids``; returns the number of body-frames removed."""
    n = 0
    for f in list(tl.states):
        keep = [s for s in tl.states[f] if int(s.pid) not in ids]
        n += len(tl.states[f]) - len(keep)
        tl.states[f] = keep
    return n


# ---- the timeline -------------------------------------------------------------

_SOURCE_RANK = {"fused": 0, "sideline": 1, "default": 2}


# A body the anchor camera detected within this many frames on BOTH sides of a frame it missed is
# that man blinking, not a fragment riding him: it keeps its filled frame whatever stands near it.
# Play 1 (2026-09-16): 18 of the 32 live-play pops (a body gone for 1-5 frames and back 0.17 m away)
# were linemen whose detection dropped for a frame while a neighbour stood inside INTERP_DUP_M.
# Measured on the live play, reach 0 -> 6: pops 32 -> 10, drawn body-frames +40, census 1.32 -> 1.32,
# live hops 4 -> 4, steps 5 -> 5, root jitter p90 0.0116 -> 0.0129 (the readmitted frames are filled
# positions); reach 12: pops 4 but census 1.45. Then, with the twin rule in: reach 6 -> 8 pops 11 -> 7,
# census 1.20 -> 1.19, jitter p90 0.0122 -> 0.0124; reach 10 pops 5 but census 1.21. Eight is the reach
# (id 11's eight-frame hole at 421-428 was one past six).
HOLE_REACH: int = 8


def _holes_by_frame(frames, views_by_frame, anchor_cam: str, reach: int) -> dict:
    """frame -> {pid}: ids the anchor camera detected within ``reach`` frames BEFORE and AFTER that
    frame (a short hole in a detected run)."""
    seen: dict = {}
    for f, d in (views_by_frame or {}).items():
        for pid, v in d.items():
            if anchor_cam in v:
                seen.setdefault(int(pid), []).append(int(f))
    out: dict = {int(f): set() for f in frames}
    fr = np.asarray(sorted(int(f) for f in frames))
    for pid, fs in seen.items():
        fs = np.asarray(sorted(set(fs)))
        for a, b in zip(fs[:-1], fs[1:]):
            if 1 < b - a <= reach + 1:
                for f in fr[(fr > a) & (fr < b)]:
                    out[int(f)].add(pid)
    return out


def _anchored_by_frame(frames, views_by_frame, anchor_cam: str, gap: int) -> dict:
    """frame -> {pid}: ids the anchor camera detected within ``gap`` frames of that
    frame. A body interpolated through a short detection gap is that player,
    not a duplicate of the neighbour it stands 0.8 m from: play 1 (2026-09-08)
    had 223 disappear/reappear events over 54 drawn ids, gaps of 4-16 frames,
    most of them linemen deduped while their track blinked."""
    seen: dict = {}
    for f, d in (views_by_frame or {}).items():
        for pid, v in d.items():
            if anchor_cam in v:
                seen.setdefault(int(pid), []).append(int(f))
    out: dict = {int(f): set() for f in frames}
    fr = np.asarray(sorted(int(f) for f in frames))
    for pid, fs in seen.items():
        fs = np.asarray(sorted(fs))
        lo = np.searchsorted(fr, fs - gap, side="left")
        hi = np.searchsorted(fr, fs + gap, side="right")
        for a, b in zip(lo, hi):
            for f in fr[a:b]:
                out[int(f)].add(pid)
    return out


def dedupe_frames(tl: "Timeline", radius_m: float = DUPLICATE_M, *, views_by_frame=None,
                  anchor_cam: str = "sideline", anchored=None, holes=None) -> int:
    """Drop, per frame, states that are another state's duplicate.

    A state whose id the anchor camera DETECTED in this frame is never a
    duplicate: the sideline sees the whole field and two of its boxes in
    one frame are two people. Measured on play 1 (2026-09-08): the older
    rule -- any one-view state within its view's depth/across radii of a
    kept state is the other camera's copy -- dropped 11.6 states a frame,
    mostly linemen a metre apart along the sideline's depth axis (4 m);
    the sideline had 20 unexcluded ids a frame and 16 bodies were drawn.
    The rest (an id the endzone alone sees this frame, or nobody: an
    interpolated frame) dedupe against the kept states: endzone-only
    within the endzone's depth/across radii (its copy of a sideline
    player the pairing missed), interpolated within ``radius_m``. Without
    ``views_by_frame`` nothing is anchored and every state dedupes at
    ``radius_m``, two-view first, best pose first. ``holes`` (frame -> {pid},
    _holes_by_frame): an interpolated state inside a short hole of its own
    detected run is kept outright. Returns the number dropped."""
    dropped = 0
    for f, states in tl.states.items():
        seen = views_by_frame.get(f, {}) if views_by_frame else {}
        recent = anchored.get(f, set()) if anchored else set()
        hole = holes.get(f, set()) if holes else set()
        detected = [s for s in states if anchor_cam in seen.get(s.pid, ())]
        interp = [s for s in states if anchor_cam not in seen.get(s.pid, ()) and s.pid in recent]
        rest = [s for s in states if anchor_cam not in seen.get(s.pid, ()) and s.pid not in recent]
        order = sorted(rest, key=lambda s: (-min(len(s.views), 2), _SOURCE_RANK.get(s.source, 3), s.pid))
        kept: list = list(detected)
        for s in sorted(interp, key=lambda s: s.pid):
            if s.pid not in hole and any(float(np.hypot(*(s.xy - k.xy))) < INTERP_DUP_M for k in detected):
                dropped += 1                                   # a second fragment id on a detected body
                continue
            kept.append(s)
        for s in order:
            if s.pid in hole:
                kept.append(s)                                 # vouched for (build_timeline's ``keep``): never a duplicate
                continue
            this = seen.get(s.pid, ())
            if this and anchor_cam not in this:
                d = [np.abs(s.xy - k.xy) for k in kept]
                dup = any(dd[0] < ONE_VIEW_DEPTH_M and dd[1] < ONE_VIEW_ACROSS_M for dd in d)
            else:
                dup = any(float(np.hypot(*(s.xy - k.xy))) < radius_m for k in kept)
            if dup:
                dropped += 1
                continue
            kept.append(s)
        tl.states[f] = sorted(kept, key=lambda s: s.pid)
    return dropped


def median_pose(records):
    """Element-wise median body pose over ``records`` (each ``[21, 3]``), a
    data-driven stance to give players who were never posed."""
    if not records:
        return np.zeros((21, 3))
    return np.median(np.stack([np.asarray(r, float).reshape(21, 3) for r in records]), axis=0)


def _nearest_views(views_by_frame, pid, f, frames_with_record):
    """The id's recorded views at ``f``, else at its nearest recorded frame,
    else both cameras (an id nobody recorded is not gated as one-view)."""
    rec = views_by_frame.get(f, {}).get(pid)
    if rec:
        return rec
    best, best_d = None, None
    for g in frames_with_record:
        v = views_by_frame.get(g, {}).get(pid)
        if v and (best_d is None or abs(g - f) < best_d):
            best, best_d = v, abs(g - f)
    return best or ("sideline", "endzone")


def lying_frames(df, *, cam: str = "sideline", aspect: float = LYING_ASPECT) -> set:
    """``{(frame, pid)}`` whose ``cam`` box is wider than ``aspect`` times its height: on the ground."""
    sub = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    h = (sub["bbox_y2"] - sub["bbox_y1"]).to_numpy(float)
    w = np.maximum(1.0, (sub["bbox_x2"] - sub["bbox_x1"]).to_numpy(float))
    on = h / w < aspect
    return {(int(f), int(p)) for f, p in zip(sub["frame"].to_numpy()[on], sub["global_player_id"].to_numpy()[on])}


def build_timeline(frames, ground_by_frame, poses_by_pid, *, default_pose=None,
                   default_betas=None, max_tilt_deg: float = MAX_TILT_DEG,
                   min_frames: int = MIN_FRAMES, views_by_frame=None, exclude=None,
                   pose_smooth: int = POSE_SMOOTH_FRAMES, pose_sigma: float = POSE_SMOOTH_SIGMA,
                   clamp_joints: bool = True, orient_sigma: float = ORIENT_SMOOTH_SIGMA,
                   unwrap: bool = True, hole_reach: int = HOLE_REACH, lying=None,
                   despike_m: float | None = DESPIKE_M, lying_sigma_mult: float | None = None, keep=None) -> Timeline:
    """``frames``: every frame to render. ``ground_by_frame``: frame ->
    {pid: xy}. ``poses_by_pid``: pid -> {frame: (body_pose[21,3],
    global_orient_world[3], betas[10], source)} at posed frames (any
    subset). ``lying``: ``{(frame, pid)}`` on the ground (lying_frames), where
    the tilt is not clamped. Returns a Timeline with a state per player per frame."""
    lying = lying or set()
    if lying_sigma_mult is None:
        lying_sigma_mult = LYING_SIGMA_MULT
    frames = [int(f) for f in frames]
    f_index = {f: i for i, f in enumerate(frames)}
    exclude = set(int(p) for p in (exclude or ()))
    pids = sorted({pid for g in ground_by_frame.values() for pid in g} - exclude)
    default_pose = np.zeros((21, 3)) if default_pose is None else np.asarray(default_pose, float)
    default_betas = np.zeros(10) if default_betas is None else np.asarray(default_betas, float)
    tl = Timeline(frames=frames)
    for pid in pids:
        xy = np.full((len(frames), 2), np.nan)
        for f, g in ground_by_frame.items():
            if pid in g and f in f_index:
                xy[f_index[f]] = g[pid]
        seen = np.flatnonzero(np.isfinite(xy).all(1))
        if len(seen) < min_frames:
            continue
        xy = smooth_xy(despike_xy(fill_gaps(frames, xy), excess_m=despike_m, half=DESPIKE_HALF))
        posed = poses_by_pid.get(pid, {})
        pf = sorted(f for f in posed if f in f_index)
        if pf:
            bp = interp_axis_angle(pf, [posed[f][0] for f in pf], frames)
            go = interp_axis_angle(pf, [np.asarray(posed[f][1]).reshape(1, 3) for f in pf], frames)[:, 0]
            # the limbs: hinge limits first (a backwards knee is the fit, not the man), then a
            # Gaussian, because their noise is on every frame (POSE_SMOOTH_SIGMA) and it rounds the
            # clamp's kinks; the orientation keeps the median, which was never measured against this
            if clamp_joints:
                bp = clamp_hinges(bp)
            if unwrap:
                # the SLERP returns canonical vectors (|v| <= pi): a body turning through a half
                # turn flips sign between frames and the smoothers below would mix antipodes
                bp, go = unwrap_axis_angles(bp), unwrap_axis_angles(go)
            bp = smooth_axis_angles_gaussian(bp, sigma=pose_sigma)
            # a body on the ground (the box says so) is fitted from sparse, poor records and its
            # interpolated pose flails between them (play 1's diving tackler 184 at 655-658, joint
            # acceleration p90 0.43 m/frame^2): those frames take a wider Gaussian
            ly = [i for i, f in enumerate(frames) if (f, pid) in lying] if lying else []
            if ly and lying_sigma_mult and pose_sigma:
                wide = smooth_axis_angles_gaussian(bp, sigma=pose_sigma * lying_sigma_mult)
                # blended in over LYING_BLEND frames either side of a lying run: swapping values on
                # the lying frames alone made a seam (play 1 id 1, isolated lying frames: p90 0.08 -> 0.55)
                idx = np.arange(len(frames))
                dist = np.min(np.abs(idx[:, None] - np.asarray(ly)[None, :]), axis=1)
                w = np.clip(1.0 - dist / float(LYING_BLEND + 1), 0.0, 1.0)[:, None, None]
                bp = (1.0 - w) * bp + w * wide
            go = (smooth_axis_angles_gaussian(go, sigma=orient_sigma) if orient_sigma and orient_sigma > 0
                  else smooth_axis_angles(go, window=pose_smooth))
            betas = np.mean([np.asarray(posed[f][2], float) for f in pf], axis=0)
            source = posed[pf[0]][3]
        else:
            yaw = yaw_from_motion(xy)
            bp = np.repeat(default_pose[None], len(frames), axis=0)
            go = np.stack([upright_from_yaw(y) for y in yaw])
            betas = default_betas
            source = "default"
            tl.n_default += 1
        for i, f in enumerate(frames):
            if not np.isfinite(xy[i]).all():
                continue
            limit = max(max_tilt_deg, MAX_TILT_TWO_VIEW_DEG) if source == "fused" else max_tilt_deg
            if (f, pid) in lying:
                orient, clamped = np.asarray(go[i], float), False        # on the ground: the lean is the pose
            else:
                orient, clamped = clamp_tilt(go[i], limit)
            tl.n_clamped += int(clamped)
            # A frame without a views record for this id (interpolated, filled)
            # inherits the nearest recorded one: defaulting it to two views let
            # a one-view id pass as two-view at every interpolated frame --
            # past the one-view dedupe and under the two-view tilt limit
            # (measured 2026-09-05: "s/e" printed for one-view ids).
            views = (tuple(_nearest_views(views_by_frame, pid, f, seen))
                     if views_by_frame else ("sideline", "endzone"))
            tl.states.setdefault(f, []).append(PlayerState(
                pid=pid, xy=xy[i], body_pose=bp[i], global_orient=orient, betas=betas,
                source=source, clamped=clamped, views=views))
    anchored = _anchored_by_frame(frames, views_by_frame, "sideline", MAX_GAP_FRAMES) if views_by_frame else None
    holes = _holes_by_frame(frames, views_by_frame, "sideline", hole_reach) if (views_by_frame and hole_reach > 0) else None
    if keep:
        # frames a rule vouches for (the quarterback held under centre): never deduped
        holes = dict(holes or {})
        for f_, pids_ in keep.items():
            holes[int(f_)] = set(holes.get(int(f_), set())) | {int(p) for p in pids_}
    tl.n_duplicates = dedupe_frames(tl, DUPLICATE_M, views_by_frame=views_by_frame, anchored=anchored, holes=holes)
    _LOG.info("timeline: %d players, %d frames, median %.0f bodies/frame, %d default-posed, "
              "%d frames tilt-clamped", len(pids), len(frames),
              float(np.median([len(v) for v in tl.states.values()])) if tl.states else 0,
              tl.n_default, tl.n_clamped)
    return tl


# ---- the aftermath: bodies stay where the play left them --------------------------------------------
# The tracker stops on a man the moment the play ends around him -- play 1's receiver steps out of bounds
# at 639 and his track ends at 638 -- and the clip's tail (08x: DEAD_AFTER_BALL frames past the dead ball)
# then drew empty turf where he stood. A body drawn within HOLD_END_REACH frames of the dead-ball frame
# keeps its last state to the end of the clip: the play is dead, nobody moves much, and a man who
# vanishes is worse than a man who stands still.
HOLD_END_REACH: int = 3


def hold_to_end(tl: "Timeline", end: int, last_frame: int, *, reach: int = HOLD_END_REACH) -> int:
    """Copy each id's last state to every frame up to ``last_frame`` when that last state lies within
    ``reach`` frames of ``end`` (the dead ball) or after it. Returns the states added."""
    import dataclasses

    last: dict = {}
    for f, sts in tl.states.items():
        for s in sts:
            if f >= last.get(int(s.pid), (-1, None))[0]:
                last[int(s.pid)] = (int(f), s)
    n = 0
    for pid, (f_last, s) in last.items():
        if f_last < int(end) - int(reach) or f_last >= int(last_frame):
            continue
        for f in range(f_last + 1, int(last_frame) + 1):
            if f not in tl.states:
                continue
            if any(int(t.pid) == pid for t in tl.states[f]):
                continue
            tl.states[f].append(dataclasses.replace(s))
            n += 1
    return n
