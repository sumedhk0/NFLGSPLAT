"""Detect the frames where a fitted pose is not a body, and repair them.

The rendered players "flail": a handful of frames per play show limbs in
positions no person adopts, and they are the most visible defect in the output.
They come from per-frame monocular fitting, which has no memory -- every frame
is solved alone, so nothing stops one frame's skeleton from disagreeing with
the last, and a bad frame is a bad body rather than merely a jittery one.

THE STRONGEST SIGNAL COSTS NOTHING TO CHECK. Bone lengths are FIXED. A femur
does not change length between frames, so any variation in a fitted sequence is
fit error by definition, with no biomechanics to argue about and no threshold
to tune from anatomy. Taking each bone's median over the sequence as that
player's true length, a frame whose bones disagree with it has been fitted
wrong -- and it stays valid whatever the player's size, because the reference
comes from that player's own sequence.

The test is on the WORST bone. An earlier version required a FRACTION of the
skeleton to be wrong and was blind to exactly the case it was built for: a
dislocated joint touches only the bones meeting at it, as few as one of 21, so
a 20% fraction rule missed a limb thrown 90 cm out of place for thirty straight
frames while the worst bone read 737% too long.

SPEED AND ACCELERATION are the weaker second check, and deliberately generous.
An NFL player's hand or foot genuinely moves fast, so limits tight enough to
catch every bad frame would reject real athletic motion; these are set to catch
the teleports, not to police plausible sport.

WHAT THIS IS NOT. It says a pose is inconsistent, not that it is wrong in any
particular way, and it cannot see a smoothly wrong pose -- a whole sequence
fitted with the arms in the wrong place has perfectly constant bone lengths.
It catches the visible failures, which is what the flailing is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.pose.forward_kinematics import SMPLX_BODY_PARENTS
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# A frame is rejected when ANY bone disagrees with that player's own median
# bone length by more than this.
#
# Judged on the WORST bone, not on a fraction of them, and that is the whole
# design. A dislocated joint touches only the bones incident to it -- as few as
# one of the 21 in the body tree -- so a rule needing some fraction of the
# skeleton to be wrong cannot see the failure this module exists to catch.
# Measured on an injected dislocation: the worst bone was off by 737% while
# just 2 of 21 bones (9.5%) were affected, and a 20% fraction rule missed all
# thirty frames of it. One impossible bone is already proof the frame is wrong,
# because bones do not change length.
BONE_TOLERANCE: float = 0.15          # 15% of the bone's length

# Generous enough to allow real sprinting limbs. A hand in a throwing motion
# passes 10 m/s, so these catch teleports rather than fast play.
MAX_JOINT_SPEED_MS: float = 25.0
MAX_JOINT_ACCEL_MS2: float = 600.0

# Below this the body has sunk through the turf. Some penetration is normal
# from fitting error and from feet being hard to see, so it is not tight.
MAX_GROUND_PENETRATION_M: float = 0.35


@dataclass(frozen=True)
class PoseAudit:
    """Per-frame verdict plus the evidence behind it."""

    ok: np.ndarray                 # [T] bool -- frame is usable
    bone_error: np.ndarray         # [T] worst relative bone deviation
    max_speed: np.ndarray          # [T] m/s
    max_accel: np.ndarray          # [T] m/s^2
    reasons: list[str]             # [T] "" when ok

    @property
    def n_bad(self) -> int:
        return int((~self.ok).sum())

    def summary(self) -> str:
        return (f"{self.n_bad}/{len(self.ok)} frames implausible "
                f"(median bone error {float(np.nanmedian(self.bone_error)):.1%})")


def bone_lengths(joints, parents=SMPLX_BODY_PARENTS) -> np.ndarray:
    """``[T, B]`` length of every bone in the kinematic tree, per frame."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 2:
        joints = joints[None]
    parents = np.asarray(parents)
    child = np.flatnonzero(parents >= 0)
    par = parents[child]
    return np.linalg.norm(joints[:, child] - joints[:, par], axis=-1)


def reference_bone_lengths(joints, parents=SMPLX_BODY_PARENTS) -> np.ndarray:
    """The player's own bone lengths, as the per-bone MEDIAN over the sequence.

    The median rather than the mean because the bad frames are exactly the
    outliers being looked for, and a mean would let them move the reference
    they are supposed to be measured against.
    """
    return np.nanmedian(bone_lengths(joints, parents), axis=0)


def joint_kinematics(joints, fps: float):
    """``(max_speed[T], max_accel[T])`` over joints, in m/s and m/s^2."""
    joints = np.asarray(joints, dtype=np.float64)
    T = len(joints)
    speed = np.zeros(T)
    accel = np.zeros(T)
    if T < 2:
        return speed, accel
    v = np.linalg.norm(np.diff(joints, axis=0), axis=-1) * fps      # [T-1, J]
    speed[1:] = v.max(axis=1)
    speed[0] = speed[1]
    if T >= 3:
        a = np.linalg.norm(np.diff(np.diff(joints, axis=0), axis=0),
                           axis=-1) * fps * fps                     # [T-2, J]
        accel[2:] = a.max(axis=1)
        accel[:2] = accel[2]
    return speed, accel


def audit(joints, *, fps: float = 59.94, parents=SMPLX_BODY_PARENTS,
          bone_tolerance: float = BONE_TOLERANCE,
          max_speed: float = MAX_JOINT_SPEED_MS,
          max_accel: float = MAX_JOINT_ACCEL_MS2,
          ground_z: float | None = None,
          max_penetration: float = MAX_GROUND_PENETRATION_M) -> PoseAudit:
    """Judge every frame of a ``[T, J, 3]`` joint sequence.

    ``ground_z`` enables the turf check; leave it None when the sequence is not
    in a frame where the ground is known.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3:
        raise ValueError(f"joints must be [T, J, 3], got {joints.shape}")
    T = len(joints)

    lengths = bone_lengths(joints, parents)
    ref = reference_bone_lengths(joints, parents)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(lengths - ref[None, :]) / np.where(ref > 1e-6, ref, np.nan)
    worst = np.nanmax(rel, axis=1)
    n_off = np.nansum(rel > bone_tolerance, axis=1)

    speed, accel = joint_kinematics(joints, fps)

    ok = np.ones(T, bool)
    reasons = [""] * T
    for t in range(T):
        why = []
        if not np.isfinite(joints[t]).all():
            why.append("non-finite joints")
        if worst[t] > bone_tolerance:
            why.append(f"{int(n_off[t])} bone(s) off, worst {worst[t]:.0%}")
        if speed[t] > max_speed:
            why.append(f"joint speed {speed[t]:.0f} m/s")
        if accel[t] > max_accel:
            why.append(f"joint accel {accel[t]:.0f} m/s2")
        if ground_z is not None:
            depth = ground_z - float(np.nanmin(joints[t, :, 2]))
            if depth > max_penetration:
                why.append(f"{depth:.2f} m below the turf")
        if why:
            ok[t] = False
            reasons[t] = "; ".join(why)
    return PoseAudit(ok, worst, speed, accel, reasons)


def repair(joints, audit_result: PoseAudit, *, max_gap: int = 8):
    """Replace implausible frames by interpolating the good ones around them.

    A short run of bad frames sits between good ones and is far better filled
    than shown -- the flailing is brief by nature, since the fit recovers as
    soon as the view does. Runs longer than ``max_gap`` are left as NaN rather
    than invented, and reported, because a long outage means something else is
    wrong and smoothing over it would hide that.
    """
    from nfl_gsplat.pose.temporal_smooth import interpolate_short_gaps

    joints = np.asarray(joints, dtype=np.float64)
    T, J, _ = joints.shape
    flat = joints.reshape(T, J * 3)
    filled, still_valid = interpolate_short_gaps(flat, audit_result.ok,
                                                 max_gap=max_gap)
    out = filled.reshape(T, J, 3)
    out[~still_valid] = np.nan
    n_filled = int(still_valid.sum() - audit_result.ok.sum())
    _LOG.info("pose repair: %d frames filled, %d left as gaps",
              n_filled, int((~still_valid).sum()))
    return out, still_valid


def mark_implausible(joints3d, valid, *, fps: float = 59.94, **audit_kwargs):
    """``(valid, PoseAudit)`` with implausible frames struck out entirely.

    Placed BEFORE the body fit rather than after it. A frame whose triangulated
    joints are not a body will otherwise drag the fit toward itself, and a
    smoother downstream can only average the damage across its neighbours --
    turning one bad frame into several mediocre ones. Marked invalid, the frame
    is simply absent, and the existing short-gap interpolation covers it with
    the good frames on either side.
    """
    joints3d = np.asarray(joints3d, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool).copy()
    got = audit(joints3d, fps=fps, **audit_kwargs)
    valid[~got.ok] = False
    if got.n_bad:
        _LOG.info("pose audit: %s", got.summary())
    return valid, got
