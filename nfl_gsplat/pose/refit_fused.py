"""SMPL-X pose parameters refit to fused two-view world joints.

WHY. scripts/05e fuses the sideline and endzone skeletons of one player into
world-frame joints (pose shape agrees to 0.19 m between views; placement is
the compromise of both). The renderer (scripts/05d) does not draw joints: it
runs the SMPL-X body from (betas, body_pose, global_orient, transl). This
module closes that gap by fitting those parameters to the fused joints with
the same forward kinematics the animator uses (pose.forward_kinematics), so
the mesh sits on the joints it was fitted to.

HOW. Per player, per frame: :func:`fuse_smplx.fuse_sequence` (least squares
over 69 parameters, warm-started frame to frame). The one thing added here is
the START of the first frame: a rigid Kabsch fit of the rest skeleton onto
the target gives global_orient and transl, because least squares from a zero
orientation stalls in a local minimum when the player faces the other way
(a 180 deg turn is the worst case, and half the players face each way).

Pure numpy; the SMPL-X rest skeleton comes in as an array so tests run on a
synthetic one.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.pose.forward_kinematics import NUM_BODY_JOINTS, fk_forward
from nfl_gsplat.pose.fuse_smplx import (SMPLXFitConfig, _pack_params,
                                        fuse_sequence)
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

PELVIS = 0


def rigid_start(rest_joints, target, valid):
    """``(global_orient[3], transl[3])`` that rigidly lay the rest skeleton
    on the valid target joints (Kabsch)."""
    from scipy.spatial.transform import Rotation

    rest = np.asarray(rest_joints, float)
    tgt = np.asarray(target, float)
    v = np.asarray(valid, bool)
    if v.sum() < 3:
        return np.zeros(3), np.nan_to_num(np.nanmean(tgt[v], axis=0) - rest[PELVIS])
    a = rest[v] - rest[v].mean(0)
    b = tgt[v] - tgt[v].mean(0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    R = (u @ np.diag([1.0, 1.0, d]) @ vt).T
    # SMPL-X rotates about the PELVIS (the root stays put), so the translation
    # is what carries the rotated pelvis onto the target's centroid offset.
    transl = tgt[v].mean(0) - R @ (rest[v].mean(0) - rest[PELVIS]) - rest[PELVIS]
    return Rotation.from_matrix(R).as_rotvec(), transl


def refit_player(joints_world, rest_joints, *, cfg: SMPLXFitConfig | None = None,
                 init_body_pose=None):
    """Fit SMPL-X pose parameters to ``joints_world[T, >=22, 3]`` (NaN where
    a joint is unknown) on the ``rest_joints[22, 3]`` skeleton.

    Returns ``(body_pose[T, 21, 3], global_orient[T, 3], transl[T, 3],
    valid[T], rms_m[T])``; raises :class:`PoseFusionError` when fewer than
    ``cfg.min_frame_validity_frac`` of the frames fit.
    """
    cfg = cfg or SMPLXFitConfig()
    target = np.asarray(joints_world, float)[:, :NUM_BODY_JOINTS]
    valid = np.isfinite(target).all(-1)
    rest = np.asarray(rest_joints, float)[:NUM_BODY_JOINTS]
    forward = fk_forward(rest)

    first = int(np.argmax(valid.sum(1) >= cfg.min_valid_joints))
    go0, tr0 = rigid_start(rest, target[first], valid[first])
    bp0 = (np.zeros(cfg.body_pose_dim) if init_body_pose is None
           else np.asarray(init_body_pose, float).reshape(-1)[:cfg.body_pose_dim])
    init = _pack_params(bp0, go0, tr0)
    # Frames before the first fittable one would drag the warm start through
    # a failed fit; fuse_sequence skips them (too few joints) and keeps `init`.
    res = fuse_sequence(target, valid, init, forward, cfg)
    return (res.body_pose.reshape(-1, 21, 3), res.global_orient, res.transl,
            res.valid_frames, res.residual_rms_m)


def render_blob(fits, betas_of, *, stride: int, appearance_cam: str = "sideline"):
    """The pose cache scripts/05d reads, in WORLD mode: ``fits`` maps player
    id -> ``(frames, body_pose, global_orient, transl, valid)``; recs carry
    ``transl`` in metres on the field instead of a foot pixel."""
    frames: dict[int, dict[int, dict]] = {}
    for pid, (fs, body_pose, orient, transl, valid) in fits.items():
        for i, f in enumerate(fs):
            if not valid[i]:
                continue
            frames.setdefault(int(f), {})[int(pid)] = {
                "betas": np.asarray(betas_of[pid], np.float32),
                "body_pose": np.asarray(body_pose[i], np.float32),
                "global_orient": np.asarray(orient[i], np.float32),
                "transl": np.asarray(transl[i], np.float32),
            }
    return {"cam": "fused", "world": True, "appearance_cam": appearance_cam,
            "stride": int(stride), "frames": frames}
