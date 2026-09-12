"""A limb no camera saw is interpolated between the poses either side of the blackout.

WHY. The fit walks forward in time: each frame starts from the one before and is pulled toward
the keypoints. When a limb disappears -- occluded, or its keypoints thrown out by the temporal
filter -- nothing constrains it, and the fit leaves it wherever the generic pose prior and the
previous frame put it. On play 1's motion man, frames 262-272, the detector loses his left elbow
and wrist behind a lineman and his arms swing at 8.9 m/s in his own frame while his shoulders sit
at 3 px. Holding the limb at its last pose was measured NO BETTER (2026-09-10, arm speed p90
8.9 -> 11.7): the pose it is held at came from the frame the detector had already lost.

The information the sequential fit cannot use is the pose AFTER the blackout. A limb that is seen
at frame 260, hidden through 272, and seen again at 273 has two good endpoints; the truth between
them is far closer to their interpolation than to either one held. This runs as a pass over the
finished cache, so it costs no refitting:

  for each player, for each body joint, find the runs of frames where NO camera saw the joint or
  anything below it, bounded on both sides by a frame that did; replace the joint's rotation
  across the run by a geodesic interpolation of the two ends.

Frames where the joint WAS seen are untouched, and a run that reaches the start or the end of the
track keeps what the fit gave it (there is nothing to interpolate toward).
"""
from __future__ import annotations

import numpy as np

JOINT_SLICE = slice(0, 63)          # body_pose is 21 joints of axis-angle


def unseen_runs(frames, seen) -> list:
    """``[(i_before, i_after)]`` index pairs bounding each blackout: a maximal run of False in
    ``seen`` with a True on both sides. ``frames`` is only used for its length."""
    seen = np.asarray(seen, bool)
    out = []
    i = 0
    n = len(seen)
    while i < n:
        if seen[i]:
            i += 1
            continue
        j = i
        while j < n and not seen[j]:
            j += 1
        if i > 0 and j < n:                       # bounded on both sides
            out.append((i - 1, j))
        i = j
    return out


def slerp_rotvec(a, b, t):
    """Rotation vector a fraction ``t`` of the way from ``a`` to ``b``, along the shortest arc."""
    from scipy.spatial.transform import Rotation, Slerp

    key = Rotation.from_rotvec(np.stack([np.asarray(a, float), np.asarray(b, float)]))
    return Slerp([0.0, 1.0], key)(np.clip(float(t), 0.0, 1.0)).as_rotvec()


def fill_track(body_poses, seen_by_joint, *, max_run: int = 40):
    """``(body_poses, n_filled)``: ``body_poses`` ``[T, 63]`` with every blacked-out run of every
    joint replaced by the interpolation of its ends. ``seen_by_joint`` ``{joint: [T] bool}`` for the
    body_pose joints (1..21). A run longer than ``max_run`` frames is left alone -- that is not an
    occlusion, it is a player who left the picture."""
    out = np.array(body_poses, float, copy=True)
    n_filled = 0
    for j, seen in seen_by_joint.items():
        k = (int(j) - 1) * 3
        if k < 0 or k + 3 > out.shape[1]:
            continue
        for lo, hi in unseen_runs(np.arange(len(out)), seen):
            if hi - lo - 1 > max_run:
                continue
            a, b = out[lo, k:k + 3], out[hi, k:k + 3]
            for i in range(lo + 1, hi):
                out[i, k:k + 3] = slerp_rotvec(a, b, (i - lo) / (hi - lo))
                n_filled += 1
    return out, n_filled
