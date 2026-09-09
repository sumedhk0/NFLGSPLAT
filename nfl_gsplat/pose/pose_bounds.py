"""Joint-angle range the one-view fit may not leave for free.

WHY. One camera cannot tell short, foreshortened legs from legs pointed at
the lens, and nothing in the one-view fit forbade the impossible: on play
1 a receiver running toward the sideline camera (id 9, frames 266-281)
got a collapsed skeleton with the legs splayed at the camera while the
keypoints on him looked like a runner. The L2 prior on body_pose (0.02)
is far too soft to stop it.

WHAT. Per axis-angle component of the 21 body joints, the 2nd and 98th
percentile over the two-camera refit records of play 1 (2963 records, 24
players: stances, blocks, runs, a catch) -- the range a body took when
two cameras constrained it. A component outside it costs
``sqrt(weight) * excess`` in the fit; inside it costs nothing, so the
keypoints keep the say within the range. Components the two-camera fit
never moved (spine 2-3, feet, collars: bounds 0) stay put. Measured on
the one-view records before the prior: 54 % of components outside, 84 %
of records with an excess over 0.3 rad somewhere.
"""
from __future__ import annotations

import numpy as np

LO = np.array([-1.32, -0.30, -0.28, -0.88, -0.01, -1.02, 0.03, -0.16, -0.55, -0.16, -0.11, -0.66, 0.00, -0.04, -0.55, -0.15, -0.13, -0.27, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.05, -0.15, -0.26, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00, -0.11, -0.05, -0.11, -0.27, -0.59, -0.55, -0.24, 0.02, -0.20, -0.00, -0.00, -0.00, -0.20, -0.80, -0.72, -0.19, -0.00, -0.17, -0.02, -0.80, -0.54, -0.08, -0.00, -0.34, -0.00, -0.00, -0.00, -0.00, -0.00, -0.00])
HI = np.array([0.28, 0.01, 1.00, 0.05, 0.30, 0.35, 1.05, 0.34, 0.25, 1.10, 0.09, 0.20, 0.74, 0.11, 0.62, 0.29, 0.33, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.42, 0.16, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.14, 0.01, 0.29, 0.06, -0.00, 0.23, 0.08, 0.64, 0.41, 0.00, 0.00, 0.00, 0.07, 0.00, 0.25, 0.08, 0.89, 0.94, 0.06, 0.08, 0.24, 0.06, 1.08, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
MARGIN_RAD: float = 0.1

# Anatomical limits, radians, per SMPL-X body joint (index into the 21 posed
# joints, i.e. joint 1 = left hip) and axis-angle component (x, y, z) in the
# parent frame of the y-up rest pose. The data-driven table above forbids
# nothing that matters (the two-camera fits it comes from allow a hip
# abduction of 1.0 rad -- their own noise); these are the limits a body
# has. Knees and elbows bend one way about one axis; hips abduct little
# and twist little; ankles flex. Signs checked against the data-driven
# table's one-signed components (knee x flexion positive, elbow y bend
# one-signed per side). Joints not listed keep the L2 prior only.
ANATOMICAL = {
    1: ((-1.6, 0.6), (-0.6, 0.6), (-0.3, 0.7)),     # left hip: flexion, twist, abduction
    2: ((-1.6, 0.6), (-0.6, 0.6), (-0.7, 0.3)),     # right hip
    4: ((-0.05, 2.6), (-0.15, 0.15), (-0.15, 0.15)),  # left knee: flexion only
    5: ((-0.05, 2.6), (-0.15, 0.15), (-0.15, 0.15)),  # right knee
    7: ((-0.8, 0.6), (-0.3, 0.3), (-0.3, 0.3)),     # left ankle
    8: ((-0.8, 0.6), (-0.3, 0.3), (-0.3, 0.3)),     # right ankle
    16: ((-1.6, 1.6), (-1.6, 1.6), (-1.6, 1.6)),    # left shoulder (free)
    17: ((-1.6, 1.6), (-1.6, 1.6), (-1.6, 1.6)),    # right shoulder
    18: ((-0.5, 0.5), (-2.6, 0.05), (-0.5, 0.5)),   # left elbow: bends one way about y
    19: ((-0.5, 0.5), (-0.05, 2.6), (-0.5, 0.5)),   # right elbow
}


def anatomical_bounds():
    """``(lo, hi)`` over the 63 components: the table's limits where listed, wide open elsewhere."""
    lo = np.full(63, -np.inf)
    hi = np.full(63, np.inf)
    for j, comps in ANATOMICAL.items():
        for k, (a, b) in enumerate(comps):
            lo[3 * (j - 1) + k] = a
            hi[3 * (j - 1) + k] = b
    return lo, hi


ANAT_LO, ANAT_HI = anatomical_bounds()


def excess(body_pose, *, lo=LO, hi=HI, margin: float = MARGIN_RAD) -> np.ndarray:
    """Per-component distance outside [lo - margin, hi + margin], zero inside."""
    bp = np.asarray(body_pose, float).reshape(-1)
    return np.maximum(0.0, (lo - margin) - bp) + np.maximum(0.0, bp - (hi + margin))
