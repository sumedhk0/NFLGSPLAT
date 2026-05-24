"""SMPL-X forward kinematics → LBS joint transforms (T1.3)."""
from __future__ import annotations

import numpy as np

from nfl_gsplat.avatars.lbs_animate import animate_gaussians
from nfl_gsplat.pose.forward_kinematics import (
    SMPLX_BODY_PARENTS,
    axis_angle_to_matrix,
    global_joint_transforms,
    joint_tfms_sequence,
    pose_params_to_rotmats,
)

# Small synthetic chain: 0←1←2 along +X.
_CHAIN_PARENTS = (-1, 0, 1)
_CHAIN_REST = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_RZ90 = np.array([0.0, 0.0, np.pi / 2])     # axis-angle, 90° about +Z


def test_axis_angle_identity_and_known_rotation():
    assert np.allclose(axis_angle_to_matrix(np.zeros(3)), np.eye(3))
    Rz = axis_angle_to_matrix(_RZ90)
    assert np.allclose(Rz @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-9)


def test_identity_pose_gives_identity_transforms():
    rot = np.tile(np.eye(3), (3, 1, 1))
    A = global_joint_transforms(_CHAIN_REST, _CHAIN_PARENTS, rot)
    assert np.allclose(A, np.tile(np.eye(4), (3, 1, 1)))


def test_subtree_only_moves_below_rotated_joint():
    rot = np.tile(np.eye(3), (3, 1, 1))
    rot[1] = axis_angle_to_matrix(_RZ90)     # rotate joint 1
    A = global_joint_transforms(_CHAIN_REST, _CHAIN_PARENTS, rot)
    # Joint 0 (above the rotation) is unaffected.
    assert np.allclose(A[0], np.eye(4))
    # Joints 1 and 2 (the subtree) are rotated.
    assert not np.allclose(A[1], np.eye(4))
    assert not np.allclose(A[2], np.eye(4))


def test_rest_point_maps_to_posed_joint_location():
    rot = np.tile(np.eye(3), (3, 1, 1))
    rot[1] = axis_angle_to_matrix(_RZ90)
    A = global_joint_transforms(_CHAIN_REST, _CHAIN_PARENTS, rot)
    # A point at rest joint 2, rigidly attached to joint 2, lands at the posed
    # joint-2 location, computed independently as (1, 1, 0).
    p_rest = np.array([2.0, 0.0, 0.0, 1.0])
    p_world = A[2] @ p_rest
    assert np.allclose(p_world[:3], [1.0, 1.0, 0.0], atol=1e-9)


def test_roundtrip_through_animate_gaussians():
    rot = np.tile(np.eye(3), (3, 1, 1))
    rot[1] = axis_angle_to_matrix(_RZ90)
    A = global_joint_transforms(_CHAIN_REST, _CHAIN_PARENTS, rot)
    # One Gaussian at rest joint 2, one-hot LBS weight to joint 2.
    xyz = _CHAIN_REST[2:3].copy()
    quat = np.array([[1.0, 0, 0, 0]])
    w = np.array([[0.0, 0.0, 1.0]])
    xyz_w, _ = animate_gaussians(xyz, quat, w, A)
    assert np.allclose(xyz_w[0], [1.0, 1.0, 0.0], atol=1e-9)


def test_translation_shifts_all_joints():
    rot = np.tile(np.eye(3), (3, 1, 1))
    transl = np.array([5.0, -2.0, 1.0])
    A0 = global_joint_transforms(_CHAIN_REST, _CHAIN_PARENTS, rot)
    A1 = global_joint_transforms(_CHAIN_REST, _CHAIN_PARENTS, rot, transl=transl)
    assert np.allclose(A1[:, :3, 3] - A0[:, :3, 3], transl[None, :])


def test_sequence_shape_and_zero_pose_identity():
    T = 4
    rest = np.random.default_rng(0).normal(size=(22, 3))
    go = np.zeros((T, 3))
    bp = np.zeros((T, 63))
    tr = np.zeros((T, 3))
    A = joint_tfms_sequence(go, bp, tr, rest, SMPLX_BODY_PARENTS)
    assert A.shape == (T, 22, 4, 4)
    assert np.allclose(A[0], np.tile(np.eye(4), (22, 1, 1)))


def test_pose_params_stack_is_22_joints():
    R = pose_params_to_rotmats(np.zeros(3), np.zeros((21, 3)))
    assert R.shape == (22, 3, 3)
    assert np.allclose(R, np.tile(np.eye(3), (22, 1, 1)))
