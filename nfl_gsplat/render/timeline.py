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

MAX_TILT_DEG: float = 35.0       # a lineman's stance; nobody stands past this
DUPLICATE_M: float = 0.9         # two ids closer than this on one frame are one player
# An id seen by ONE view only is that view's unreconciled detection; its
# position is poor along that view's depth axis (the endzone's is x, the
# sideline's is y). Within these distances of a two-view id it is the same
# man: play 2 drew six red bodies strung along x from one group of players.
ONE_VIEW_DEPTH_M: float = 4.0
ONE_VIEW_ACROSS_M: float = 1.5
MAX_GAP_FRAMES: int = 30         # half a second of missing detections is bridged
MIN_FRAMES: int = 6              # shorter fragments are noise
VEL_WINDOW: int = 12             # frames over which yaw follows the travel direction
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
    """Zero-phase moving average with edge handling; NaN rows stay NaN."""
    xy = np.asarray(xy, float)
    out = xy.copy()
    ok = np.isfinite(xy).all(1)
    if ok.sum() < 3:
        return out
    idx = np.flatnonzero(ok)
    for d in range(2):
        v = xy[idx, d]
        k = min(window, len(v) if len(v) % 2 else len(v) - 1)
        if k < 3:
            continue
        pad = k // 2
        vp = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
        out[idx, d] = np.convolve(vp, np.ones(k) / k, mode="valid")
    return out


def fill_gaps(frames, xy, *, max_gap: int = MAX_GAP_FRAMES):
    """Linear fill of NaN rows between known rows when the gap is short."""
    xy = np.asarray(xy, float).copy()
    ok = np.isfinite(xy).all(1)
    idx = np.flatnonzero(ok)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and (frames[b] - frames[a]) <= max_gap:
            w = np.linspace(0, 1, b - a + 1)[1:-1, None]
            xy[a + 1:b] = (1 - w) * xy[a] + w * xy[b]
    return xy


def yaw_from_motion(xy, *, window: int = VEL_WINDOW, fallback: float = 0.0):
    """Facing from the direction of travel, per row; ``fallback`` when still."""
    xy = np.asarray(xy, float)
    n = len(xy)
    yaw = np.full(n, fallback)
    for i in range(n):
        a, b = max(0, i - window // 2), min(n - 1, i + window // 2)
        d = xy[b] - xy[a]
        if np.isfinite(d).all() and np.linalg.norm(d) > 0.3:
            yaw[i] = float(np.arctan2(d[1], d[0]))
    # hold the last known heading through still stretches
    last = fallback
    for i in range(n):
        if yaw[i] == fallback and i > 0:
            yaw[i] = last
        last = yaw[i]
    return yaw


# ---- the timeline -------------------------------------------------------------

_SOURCE_RANK = {"fused": 0, "sideline": 1, "default": 2}


def _within(s, k) -> bool:
    """Is one-view state ``s`` a duplicate of kept state ``k``?"""
    d = np.abs(s.xy - k.xy)
    if len(s.views) >= 2:
        return float(np.hypot(*d)) < DUPLICATE_M
    depth_axis = 0 if "endzone" in s.views else 1          # endzone depth is x
    across = 1 - depth_axis
    return d[depth_axis] < ONE_VIEW_DEPTH_M and d[across] < ONE_VIEW_ACROSS_M


def dedupe_frames(tl: "Timeline", radius_m: float = DUPLICATE_M) -> int:
    """Drop, per frame, states that are another state's duplicate.

    Two-view (reconciled) states are kept first, best pose first; a one-view
    state within its view's depth/across radii of a kept state is the same
    player seen by the other camera and dropped; one-view states among
    themselves dedupe at ``radius_m``. Returns the number dropped."""
    dropped = 0
    for f, states in tl.states.items():
        order = sorted(states, key=lambda s: (-min(len(s.views), 2), _SOURCE_RANK.get(s.source, 3), s.pid))
        kept: list = []
        for s in order:
            dup = any(_within(s, k) for k in kept) if len(s.views) < 2 else \
                any(float(np.hypot(*(s.xy - k.xy))) < radius_m for k in kept)
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


def build_timeline(frames, ground_by_frame, poses_by_pid, *, default_pose=None,
                   default_betas=None, max_tilt_deg: float = MAX_TILT_DEG,
                   min_frames: int = MIN_FRAMES, views_by_frame=None) -> Timeline:
    """``frames``: every frame to render. ``ground_by_frame``: frame ->
    {pid: xy}. ``poses_by_pid``: pid -> {frame: (body_pose[21,3],
    global_orient_world[3], betas[10], source)} at posed frames (any
    subset). Returns a Timeline with a state per player per frame."""
    frames = [int(f) for f in frames]
    f_index = {f: i for i, f in enumerate(frames)}
    pids = sorted({pid for g in ground_by_frame.values() for pid in g})
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
        xy = smooth_xy(fill_gaps(frames, xy))
        posed = poses_by_pid.get(pid, {})
        pf = sorted(f for f in posed if f in f_index)
        if pf:
            bp = interp_axis_angle(pf, [posed[f][0] for f in pf], frames)
            go = interp_axis_angle(pf, [np.asarray(posed[f][1]).reshape(1, 3) for f in pf], frames)[:, 0]
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
            orient, clamped = clamp_tilt(go[i], max_tilt_deg)
            tl.n_clamped += int(clamped)
            views = (tuple(views_by_frame.get(f, {}).get(pid, ("sideline", "endzone")))
                     if views_by_frame else ("sideline", "endzone"))
            tl.states.setdefault(f, []).append(PlayerState(
                pid=pid, xy=xy[i], body_pose=bp[i], global_orient=orient, betas=betas,
                source=source, clamped=clamped, views=views))
    tl.n_duplicates = dedupe_frames(tl, DUPLICATE_M)
    _LOG.info("timeline: %d players, %d frames, median %.0f bodies/frame, %d default-posed, "
              "%d frames tilt-clamped", len(pids), len(frames),
              float(np.median([len(v) for v in tl.states.values()])) if tl.states else 0,
              tl.n_default, tl.n_clamped)
    return tl
