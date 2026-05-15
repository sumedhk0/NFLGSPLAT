"""Apply Linear Blend Skinning to a canonical Gaussian cloud per frame.

Given:

- ``canonical_xyz      [N, 3]`` Gaussian centers in the T-pose canonical frame
- ``canonical_rot_quat [N, 4]`` per-Gaussian orientation (wxyz) in canonical frame
- ``lbs_weights        [N, J]`` convex weights summing to 1 over joints
- ``joint_world_tfms   [J, 4, 4]`` per-joint rigid transform (canonical → world)

Returns the animated ``(xyz_world, rot_world_quat)``.

Math: LBS blends per-joint rigid transforms::

    T_skin = sum_j  w_{n,j} * T_j

and applies ``T_skin`` to each canonical point. For orientations, we blend
the 3×3 rotation parts (not exact but visually fine for Gaussians with
small scale) and re-orthonormalize via SVD.

This module is numpy-only. A torch-tensor variant used by the GPU renderer
is in ``compositing/render_gsplat.py``.
"""
from __future__ import annotations

import numpy as np


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert (..., 4) wxyz quaternions to (..., 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrix to (..., 4) wxyz quaternion.

    Shepperd's branching method: pick the largest of ``{trace, m00, m11, m22}``
    to avoid divide-by-small-number blow-ups near 180° rotations.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q = np.empty(R.shape[:-2] + (4,), dtype=np.float64)

    case_w = tr > 0
    case_x = ~case_w & (m00 >= m11) & (m00 >= m22)
    case_y = ~case_w & ~case_x & (m11 >= m22)
    case_z = ~case_w & ~case_x & ~case_y

    if np.any(case_w):
        s = np.sqrt(tr[case_w] + 1.0) * 2
        q[case_w, 0] = 0.25 * s
        q[case_w, 1] = (m21[case_w] - m12[case_w]) / s
        q[case_w, 2] = (m02[case_w] - m20[case_w]) / s
        q[case_w, 3] = (m10[case_w] - m01[case_w]) / s
    if np.any(case_x):
        s = np.sqrt(1.0 + m00[case_x] - m11[case_x] - m22[case_x]) * 2
        q[case_x, 0] = (m21[case_x] - m12[case_x]) / s
        q[case_x, 1] = 0.25 * s
        q[case_x, 2] = (m01[case_x] + m10[case_x]) / s
        q[case_x, 3] = (m02[case_x] + m20[case_x]) / s
    if np.any(case_y):
        s = np.sqrt(1.0 + m11[case_y] - m00[case_y] - m22[case_y]) * 2
        q[case_y, 0] = (m02[case_y] - m20[case_y]) / s
        q[case_y, 1] = (m01[case_y] + m10[case_y]) / s
        q[case_y, 2] = 0.25 * s
        q[case_y, 3] = (m12[case_y] + m21[case_y]) / s
    if np.any(case_z):
        s = np.sqrt(1.0 + m22[case_z] - m00[case_z] - m11[case_z]) * 2
        q[case_z, 0] = (m10[case_z] - m01[case_z]) / s
        q[case_z, 1] = (m02[case_z] + m20[case_z]) / s
        q[case_z, 2] = (m12[case_z] + m21[case_z]) / s
        q[case_z, 3] = 0.25 * s

    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def animate_gaussians(
    canonical_xyz: np.ndarray,          # [N, 3]
    canonical_rot_quat: np.ndarray,     # [N, 4]
    lbs_weights: np.ndarray,            # [N, J]
    joint_world_tfms: np.ndarray,       # [J, 4, 4]
) -> tuple[np.ndarray, np.ndarray]:
    """LBS-blend the per-joint rigid transforms and apply to each Gaussian.

    Returns ``(xyz_world [N, 3], rot_world_quat [N, 4])``.
    """
    canonical_xyz = np.asarray(canonical_xyz, dtype=np.float64)
    canonical_rot_quat = np.asarray(canonical_rot_quat, dtype=np.float64)
    lbs_weights = np.asarray(lbs_weights, dtype=np.float64)
    joint_world_tfms = np.asarray(joint_world_tfms, dtype=np.float64)
    N, _ = canonical_xyz.shape
    J = joint_world_tfms.shape[0]
    assert lbs_weights.shape == (N, J), f"lbs_weights shape {lbs_weights.shape} != ({N}, {J})"

    # Blend the 3×4 portions: (N, 3, 4) = sum_j w_{n,j} * T_j[:3, :4]
    T34 = joint_world_tfms[:, :3, :4]                        # [J, 3, 4]
    blend = np.einsum("nj,jik->nik", lbs_weights, T34)       # [N, 3, 4]
    xyz_hom = np.concatenate(
        [canonical_xyz, np.ones((N, 1), dtype=np.float64)], axis=-1
    )                                                        # [N, 4]
    xyz_world = np.einsum("nik,nk->ni", blend, xyz_hom)      # [N, 3]

    R_blend = blend[:, :, :3]                                # [N, 3, 3]
    # Orthonormalize via SVD per Gaussian. SVD is O(N · 27 flops) — cheap.
    U, _, Vt = np.linalg.svd(R_blend)
    R_blend_orth = U @ Vt
    # Guard against reflections (det == -1) from the SVD.
    det = np.linalg.det(R_blend_orth)
    fix = np.where(det < 0, -1.0, 1.0).reshape(N, 1, 1)
    R_blend_orth = R_blend_orth * fix

    R_canonical = _quat_to_rotmat(canonical_rot_quat)
    R_world = np.einsum("nij,njk->nik", R_blend_orth, R_canonical)
    rot_world_quat = _rotmat_to_quat(R_world)
    return xyz_world, rot_world_quat
