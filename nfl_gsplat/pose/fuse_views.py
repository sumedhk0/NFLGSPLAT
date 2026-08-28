"""Combine two cameras' independent estimates of the same player.

Each feed produces a whole skeleton on its own: a crop goes into the pose
network, the articulation comes back, and the calibration puts it on the field.
When identity says two tracks are the same person, that yields TWO placements of
one body from cameras 131 m apart, derived from different pixels through
different calibrations.

Two things follow, and the first matters more than the second:

* A MEASUREMENT. Nothing else in this pipeline checks pose against anything
  external -- a plausible-looking skeleton is what a wrong one looks like too.
  The disagreement between two independent estimates is a real error bar.
* A FUSION, and not a naive average. A monocular estimate is badly conditioned
  ALONG the camera's viewing ray -- that is the depth the network had to guess --
  and well conditioned across it. The two cameras here are nearly orthogonal, so
  each one's weak axis is the other's strong one: the sideline's rays graze the
  field at 80-100 m and it measures across-field position poorly, which is
  exactly what the endzone sees best. Weighting per-axis by that geometry uses
  the arrangement instead of averaging it away.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Standard deviation, in metres, of a monocular placement ACROSS the viewing
# ray. This is the well-conditioned direction: the foot point pins it through
# the calibration, and cross-camera nearest-player agreement measured 1.4-1.8 m
# overall, most of which is the along-ray component below.
SIGMA_ACROSS_M: float = 0.35

# ... and ALONG it, where the network is guessing depth from a crop a hundred
# pixels tall. Deliberately several times SIGMA_ACROSS_M rather than fitted:
# what matters is the RATIO, which says "believe each camera about the
# directions it can actually see", and the fusion is insensitive to the exact
# value once the ratio is large.
SIGMA_ALONG_M: float = 2.5

# Measured per-camera placement precision on play_001, as the median departure
# of a track from its own locally smooth path -- which needs only one camera and
# no correspondence. These are NOT interchangeable instruments:
SIDELINE_JITTER_M: float = 0.01
ENDZONE_JITTER_M: float = 1.08
# The box edges are equally steady in both (0.3 px and 0.6 px), so the
# difference is the CAMERA, not the detector. Anything fusing the two should
# scale their sigmas by roughly this ratio rather than averaging them as peers.


def ray_directions(points, camera_centre):
    """Unit vectors from the camera centre to each point. ``[N, 3]``."""
    points = np.asarray(points, float)
    centre = np.asarray(camera_centre, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N, 3], got {points.shape}")
    rays = points - centre
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    if not np.all(norms > 1e-9):
        raise ValueError("a point coincides with the camera centre")
    return rays / norms


def _information(rays, sigma_along: float, sigma_across: float):
    """Per-point inverse covariance, elongated along each viewing ray.

    Built as ``(1/s_across^2) I + (1/s_along^2 - 1/s_across^2) r r^T``: isotropic
    confidence across the ray, reduced along it.
    """
    n = len(rays)
    iso = 1.0 / sigma_across ** 2
    along = 1.0 / sigma_along ** 2
    eye = np.broadcast_to(np.eye(3), (n, 3, 3))
    outer = rays[:, :, None] * rays[:, None, :]
    return iso * eye + (along - iso) * outer


def fuse_skeletons(joints_a, joints_b, centre_a, centre_b, *,
                   sigma_along_m: float = SIGMA_ALONG_M,
                   sigma_across_m: float = SIGMA_ACROSS_M,
                   scale_a: float = 1.0, scale_b: float = 1.0):
    """Inverse-variance fusion of two placed skeletons. Returns ``[J, 3]``.

    ``joints_a``/``joints_b`` are the SAME joints of the SAME player, already in
    field coordinates, one from each camera. ``centre_a``/``centre_b`` are the
    camera centres, which is all the geometry the weighting needs.

    ``scale_a``/``scale_b`` multiply each camera's sigmas, for when the two are
    not equally good. On play_001 they are not, by two orders of magnitude:
    placement jitter measured 0.01 m on the sideline against 1.08 m on the
    endzone, from box edges that move by 0.3 and 0.6 px respectively. The
    endzone zooms across f = 1,532 to 23,200 and re-solves its focal and
    rotation every frame, and its verification checks yard lines across the
    view, which at an 11 degree grazing angle barely constrains depth. Passing
    ``scale_b`` around 10 tells the fusion what the measurement already says:
    believe the sideline about position, and the endzone about very little of
    it. See :data:`SIDELINE_JITTER_M` / :data:`ENDZONE_JITTER_M`.
    """
    a = np.asarray(joints_a, float)
    b = np.asarray(joints_b, float)
    if a.shape != b.shape:
        raise ValueError(f"skeletons differ in shape: {a.shape} vs {b.shape}")
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"joints must be [J, 3], got {a.shape}")

    # Both rays are taken to the SAME reference point, the midpoint of the two
    # estimates, rather than each to its own. The covariance model describes
    # where the player IS, and evaluating it at two different guesses makes the
    # fusion asymmetric -- swapping the arguments would change the answer, and
    # two identical estimates from one camera centre would not average. The
    # direction barely moves over the metre or two the estimates disagree by.
    reference = 0.5 * (a + b)
    info_a = _information(ray_directions(reference, centre_a),
                          sigma_along_m * scale_a, sigma_across_m * scale_a)
    info_b = _information(ray_directions(reference, centre_b),
                          sigma_along_m * scale_b, sigma_across_m * scale_b)
    total = info_a + info_b
    # Keep the right-hand side as a STACK OF COLUMN VECTORS, [J, 3, 1]. numpy 2
    # reads a [J, 3] operand as one matrix rather than J vectors and fails with
    # a core-dimension mismatch.
    rhs = info_a @ a[:, :, None] + info_b @ b[:, :, None]
    return np.linalg.solve(total, rhs)[:, :, 0]


def disagreement(joints_a, joints_b):
    """Per-joint distance between two independent placements, in metres.

    The honest error bar on a monocular pose: two cameras that share no pixels,
    no tracker and no calibration should not disagree by more than the accuracy
    either one can claim.
    """
    a = np.asarray(joints_a, float)
    b = np.asarray(joints_b, float)
    if a.shape != b.shape:
        raise ValueError(f"skeletons differ in shape: {a.shape} vs {b.shape}")
    return np.linalg.norm(a - b, axis=1)


def summarise(joints_a, joints_b):
    """``{median, p90, max, root}`` disagreement in metres.

    ``root`` is the whole-body offset -- how far apart the two cameras put the
    player -- separated from the rest, because a large root with small residual
    spread means the placements disagree while the POSES agree, which is a
    calibration or foot-point problem rather than a pose one.
    """
    d = disagreement(joints_a, joints_b)
    a = np.asarray(joints_a, float)
    b = np.asarray(joints_b, float)
    root = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    centred = disagreement(a - a.mean(axis=0), b - b.mean(axis=0))
    return {"median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)),
            "max_m": float(d.max()), "root_m": root,
            "shape_median_m": float(np.median(centred))}
