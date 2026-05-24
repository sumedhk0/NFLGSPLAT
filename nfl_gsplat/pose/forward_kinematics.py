"""SMPL-X forward kinematics → per-joint world transforms for LBS.

This is the bridge from the pose solve to the renderer. ``fuse_smplx`` outputs
SMPL-X pose params (``body_pose``, ``global_orient``, ``transl``); the avatar
animator (``lbs_animate.animate_gaussians``) needs per-joint canonical→world
transforms ``A[J, 4, 4]``. This module computes ``A`` from the kinematic tree.

Math (the standard SMPL/SMPL-X rigid-transform recursion):

- local transform of joint ``i``: ``T_local_i = [R_i | (J_i - J_parent)]``
- global: ``T_global_i = T_global_parent @ T_local_i``
- LBS transform that maps a *rest-pose* point attached to joint ``i`` to world::

      A_i[:3,:3] = T_global_i[:3,:3]
      A_i[:3, 3] = T_global_i[:3, 3] - T_global_i[:3,:3] @ J_i

  (subtracting the rest joint location so a point at ``J_i`` maps to the posed
  joint location). Root translation ``transl`` is then added to every ``A[:, :3, 3]``.

The core is pure numpy (CPU-testable with a synthetic chain). Loading the real
SMPL-X rest joints is env-gated (needs the body models).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# SMPL-X body kinematic tree (first 22 joints; -1 = root has no parent).
SMPLX_BODY_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
)
NUM_BODY_JOINTS = 22


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)`` axis-angle vectors to ``(..., 3, 3)`` rotations."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)            # (..., 1)
    small = theta < 1e-8
    axis = aa / np.where(small, 1.0, theta)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([
        np.stack([zero, -z, y], axis=-1),
        np.stack([z, zero, -x], axis=-1),
        np.stack([-y, x, zero], axis=-1),
    ], axis=-2)                                                    # (..., 3, 3)
    th = theta[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape).copy()
    R = eye + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
    # theta≈0 → identity (sin/cos terms vanish, but guard explicitly).
    R = np.where(small[..., None], eye, R)
    return R


def pose_params_to_rotmats(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """Stack root + body joint rotations → ``[J, 3, 3]``.

    ``global_orient`` is ``[3]`` (joint 0), ``body_pose`` is ``[21, 3]``.
    """
    go = np.asarray(global_orient, dtype=np.float64).reshape(3)
    bp = np.asarray(body_pose, dtype=np.float64).reshape(NUM_BODY_JOINTS - 1, 3)
    aa = np.concatenate([go[None, :], bp], axis=0)                 # [22, 3]
    return axis_angle_to_matrix(aa)


def global_joint_transforms(
    rest_joints: np.ndarray,       # [J, 3]
    parents: tuple[int, ...] | np.ndarray,
    rot_mats: np.ndarray,          # [J, 3, 3]
    transl: np.ndarray | None = None,
) -> np.ndarray:
    """Return LBS transforms ``A[J, 4, 4]`` (canonical→world per joint)."""
    rest = np.asarray(rest_joints, dtype=np.float64)
    J = rest.shape[0]
    parents = list(parents)
    R = np.asarray(rot_mats, dtype=np.float64)
    assert R.shape == (J, 3, 3), f"rot_mats {R.shape} != ({J}, 3, 3)"

    # Local joint offsets relative to parent.
    rel = rest.copy()
    for i in range(1, J):
        rel[i] = rest[i] - rest[parents[i]]

    T_global = np.zeros((J, 4, 4), dtype=np.float64)
    for i in range(J):
        T_local = np.eye(4)
        T_local[:3, :3] = R[i]
        T_local[:3, 3] = rel[i]
        T_global[i] = T_local if parents[i] < 0 else T_global[parents[i]] @ T_local

    A = np.zeros((J, 4, 4), dtype=np.float64)
    for i in range(J):
        A[i, :3, :3] = T_global[i, :3, :3]
        A[i, :3, 3] = T_global[i, :3, 3] - T_global[i, :3, :3] @ rest[i]
        A[i, 3, 3] = 1.0
    if transl is not None:
        A[:, :3, 3] += np.asarray(transl, dtype=np.float64).reshape(3)[None, :]
    return A


def joint_tfms_sequence(
    global_orient: np.ndarray,     # [T, 3]
    body_pose: np.ndarray,         # [T, 63] or [T, 21, 3]
    transl: np.ndarray,            # [T, 3]
    rest_joints: np.ndarray,       # [J, 3]
    parents: tuple[int, ...] | np.ndarray = SMPLX_BODY_PARENTS,
) -> np.ndarray:
    """Per-frame LBS transforms ``[T, J, 4, 4]`` for a fitted pose sequence."""
    go = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    bp = np.asarray(body_pose, dtype=np.float64).reshape(go.shape[0], NUM_BODY_JOINTS - 1, 3)
    tr = np.asarray(transl, dtype=np.float64).reshape(-1, 3)
    T = go.shape[0]
    J = rest_joints.shape[0]
    out = np.zeros((T, J, 4, 4), dtype=np.float64)
    for t in range(T):
        R = pose_params_to_rotmats(go[t], bp[t])
        out[t] = global_joint_transforms(rest_joints, parents, R, transl=tr[t])
    return out


# --- Env-gated SMPL-X skeleton loader --------------------------------------

def load_smplx_skeleton(
    body_models_dir: Path | str,
    gender: str = "neutral",
    betas: np.ndarray | None = None,
):
    """Load ``(rest_joints[22,3], parents)`` from the SMPL-X model.

    Rest joints depend on shape: ``J = J_regressor @ (v_template + shapedirs·betas)``.
    Requires the SMPL-X ``.npz`` under ``body_models_dir`` (license-gated, see
    SETUP.md §2). Returns the first 22 (body) joints to match our rig.
    """
    from nfl_gsplat.errors import SetupError

    path = Path(body_models_dir) / "smplx" / f"SMPLX_{gender.upper()}.npz"
    if not path.exists():
        raise SetupError(
            f"SMPL-X model not found at {path}. See SETUP.md §2 (license-gated download)."
        )
    data = np.load(path, allow_pickle=True)
    v_template = data["v_template"].astype(np.float64)             # [V, 3]
    J_regressor = data["J_regressor"].astype(np.float64)           # [J, V]
    if betas is not None:
        shapedirs = data["shapedirs"].astype(np.float64)           # [V, 3, n_betas]
        b = np.asarray(betas, dtype=np.float64).reshape(-1)
        n = min(b.shape[0], shapedirs.shape[-1])
        v_template = v_template + (shapedirs[..., :n] @ b[:n])
    joints = J_regressor @ v_template                              # [J, 3]
    return joints[:NUM_BODY_JOINTS], SMPLX_BODY_PARENTS
