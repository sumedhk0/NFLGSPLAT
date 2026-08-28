"""Put a SMPL-X skeleton into FIELD coordinates using the calibrated camera.

SMPLest-X returns two things of very different reliability:

* ``joints3d_cam`` -- the articulated skeleton, root-relative, in metres. This is
  what the network is actually good at, and it is metric because the body model
  is metric.
* ``transl`` -- where it thinks the body sits in the camera frame. This is a
  monocular depth estimate made against an ASSUMED crop focal length, not our
  calibrated one, so its scale is not tied to the field at all.

Using ``transl`` would throw away the calibration and inherit a depth guess.
Instead: take articulation from the network and GLOBAL PLACEMENT from geometry.
A player standing on the field touches z = 0, so the ray through the foot pixel
meets the field plane at exactly one point, and the calibration is what is
trusted for position.

Measured on play_001 before building this: grounded foot points from both
cameras land 100% on the field, and recovered stature comes out right once the
ankle's own height above the turf is accounted for.

Frame matches the rest of the project: X along the field's length, Y across it,
Z up, origin at the centre, playing surface at Z = 0.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.errors import CalibrationError

# Height above the turf at which the foot REFERENCE joint is pinned. The ankle
# joint is not the sole of the foot; ignoring the difference sinks every
# skeleton, which reads as a systematic stature error -- measured ~5% before it
# was accounted for.
#
# The reference is the _FOOT_QUANTILE-th percentile of joint height, NOT the
# minimum, so one badly estimated toe cannot drag a whole body into the ground.
# A consequence worth knowing: the very lowest joint may sit slightly below zero,
# which is physically unremarkable (a toe presses into turf) and visually
# negligible.
ANKLE_HEIGHT_M: float = 0.08

# Fraction of joints treated as "the feet" when locating the bottom of a body.
# Using the lowest few rather than named indices keeps this independent of
# whichever joint convention the network returns (SMPL-X body is 22, but the
# full return here is 137 including hands and face).
_FOOT_QUANTILE: float = 0.05


def ground_point(foot_uv, k_mat, rot, tvec):
    """Where the ray through ``foot_uv`` meets the field plane, in world metres.

    Raises rather than returning a bogus point when the ray runs parallel to the
    field or points away from it -- a silent wrong answer here would place a
    player in the stands with no other symptom.
    """
    k_mat = np.asarray(k_mat, float)
    rot = np.asarray(rot, float)
    centre = -rot.T @ np.asarray(tvec, float)

    direction = rot.T @ (np.linalg.inv(k_mat) @ np.array([foot_uv[0], foot_uv[1], 1.0]))
    if abs(direction[2]) < 1e-9:
        raise CalibrationError(
            "foot ray is parallel to the field plane; it never meets z=0, so "
            "this detection cannot be placed.")
    scale = -centre[2] / direction[2]
    if scale <= 0:
        raise CalibrationError(
            "field plane lies BEHIND the camera along this ray -- the pose or "
            "the detection is wrong, not merely imprecise.")
    return centre + scale * direction


def place_skeleton(joints_cam, foot_uv, k_mat, rot, tvec, *,
                   ankle_height_m: float = ANKLE_HEIGHT_M):
    """Rotate a root-relative skeleton into world axes and stand it on the field.

    ``joints_cam`` is ``[J, 3]`` root-relative, in the camera's axes.
    Returns ``[J, 3]`` in field coordinates.

    The camera's rotation is reused directly for the crop. A crop is a
    sub-window of the same image plane -- no rotation of its own -- so the axes
    agree; the residual is the perspective difference between the crop centre
    and the principal point, which is small for a player-sized box.
    """
    rot_world, offset = placement_transform(joints_cam, foot_uv, k_mat, rot,
                                            tvec, ankle_height_m=ankle_height_m)
    return (rot_world @ np.asarray(joints_cam, float).T).T + offset


def placement_transform(joints_cam, foot_uv, k_mat, rot, tvec, *,
                        ankle_height_m: float = ANKLE_HEIGHT_M):
    """``(rot_world, offset)`` taking camera-axis points to field coordinates.

    Exposed because a MESH has to move with its skeleton. The transform is
    derived from the JOINTS in both cases: the lowest joints are the ankles, and
    :data:`ANKLE_HEIGHT_M` is defined against the ankle. Deriving it from
    vertices instead would pin the sole of the shoe at ankle height and float
    every player 8 cm above the turf.
    """
    joints_cam = np.asarray(joints_cam, float)
    if joints_cam.ndim != 2 or joints_cam.shape[1] != 3:
        raise ValueError(f"joints_cam must be [J, 3], got {joints_cam.shape}")

    ground = ground_point(foot_uv, k_mat, rot, tvec)

    # camera axes -> world axes (rotation only; position comes from the ground)
    rot_world = np.asarray(rot, float).T
    world_rel = (rot_world @ joints_cam.T).T

    # Stand it up: the lowest joints are the feet, and the ankle sits
    # ankle_height_m above the turf rather than on it.
    foot_z = np.quantile(world_rel[:, 2], _FOOT_QUANTILE)
    offset = np.array([ground[0], ground[1], ankle_height_m - foot_z])
    # the horizontal offset is applied about the feet, not the root, so the
    # contact point is what the calibration pins
    foot_xy = np.array([
        np.mean(world_rel[world_rel[:, 2] <= foot_z + 1e-6, 0]),
        np.mean(world_rel[world_rel[:, 2] <= foot_z + 1e-6, 1]),
    ])
    offset[0] -= foot_xy[0]
    offset[1] -= foot_xy[1]
    return rot_world, offset


def place_mesh(vertices_cam, joints_cam, foot_uv, k_mat, rot, tvec, *,
               ankle_height_m: float = ANKLE_HEIGHT_M):
    """Same placement as :func:`place_skeleton`, applied to mesh vertices.

    ``joints_cam`` still drives the transform, so the mesh stays rigidly
    attached to the skeleton it came from and the soles reach the turf.
    """
    vertices_cam = np.asarray(vertices_cam, float)
    if vertices_cam.ndim != 2 or vertices_cam.shape[1] != 3:
        raise ValueError(f"vertices_cam must be [V, 3], got {vertices_cam.shape}")
    rot_world, offset = placement_transform(joints_cam, foot_uv, k_mat, rot,
                                            tvec, ankle_height_m=ankle_height_m)
    return (rot_world @ vertices_cam.T).T + offset


def stature(joints_world) -> float:
    """Vertical extent of a placed skeleton, in metres.

    Worth checking on real output: nothing in the placement supplies a human
    height, so a plausible value is independent evidence that the calibration,
    the ground assumption and the network all agree.
    """
    joints_world = np.asarray(joints_world, float)
    return float(joints_world[:, 2].max() - joints_world[:, 2].min())
