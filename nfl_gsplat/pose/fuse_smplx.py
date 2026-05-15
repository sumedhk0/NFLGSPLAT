"""Per-frame SMPL-X pose refit to triangulated 3D joints.

Decomposing the pose-fit problem so it is testable without SMPL-X weights:

- ``ForwardFn = (params: np.ndarray) -> (joints3d: [J, 3], jacobian: [J, 3, D] | None)``

The actual SMPL-X forward is wrapped in :class:`SMPLXForward` (lazy torch
import + weights path). Tests can pass a trivial forward (rigid body + root
translation) and still exercise the optimizer.

Optimized variables per frame: ``(body_pose, global_orient, transl)`` — betas
are held at the reference-view estimate to stabilize the scale. We use
``scipy.optimize.least_squares`` with ``soft_l1`` loss. A small L2 regularizer
on ``body_pose`` acts as a pose prior (surrogate for VPoser/GMM when those
weights are unavailable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from nfl_gsplat.errors import PoseFusionError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

ForwardFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SMPLXFitConfig:
    num_body_joints: int = 22
    body_pose_dim: int = 63           # 21 body joints × 3 axis-angle
    global_orient_dim: int = 3
    transl_dim: int = 3
    pose_prior_weight: float = 0.01   # L2 on body_pose (axis-angle magnitudes)
    min_valid_joints: int = 10
    min_frame_validity_frac: float = 0.7
    max_iter: int = 50
    loss: str = "soft_l1"             # scipy least_squares loss


@dataclass
class FitResult:
    body_pose: np.ndarray        # [T, 63]
    global_orient: np.ndarray    # [T, 3]
    transl: np.ndarray           # [T, 3]
    valid_frames: np.ndarray     # [T] bool
    residual_rms_m: np.ndarray   # [T] per-frame RMS residual after fit


def _param_slices(cfg: SMPLXFitConfig) -> tuple[slice, slice, slice]:
    bp = slice(0, cfg.body_pose_dim)
    go = slice(cfg.body_pose_dim, cfg.body_pose_dim + cfg.global_orient_dim)
    tr = slice(cfg.body_pose_dim + cfg.global_orient_dim,
               cfg.body_pose_dim + cfg.global_orient_dim + cfg.transl_dim)
    return bp, go, tr


def _pack_params(body_pose: np.ndarray, global_orient: np.ndarray,
                 transl: np.ndarray) -> np.ndarray:
    return np.concatenate([body_pose.ravel(), global_orient.ravel(), transl.ravel()])


def _unpack_params(p: np.ndarray, cfg: SMPLXFitConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp, go, tr = _param_slices(cfg)
    return p[bp].copy(), p[go].copy(), p[tr].copy()


def fit_single_frame(
    target_joints: np.ndarray,   # [J, 3], NaN where invalid
    valid: np.ndarray,           # [J] bool
    init_params: np.ndarray,     # [D]
    forward: ForwardFn,
    cfg: SMPLXFitConfig,
) -> tuple[np.ndarray, float]:
    """Optimize (body_pose, global_orient, transl) for one frame.

    Returns ``(params, residual_rms_m)``.
    """
    target = np.asarray(target_joints, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if mask.sum() < cfg.min_valid_joints:
        raise ValueError(
            f"only {int(mask.sum())} valid joints (need >= {cfg.min_valid_joints})"
        )

    bp_slice, _, _ = _param_slices(cfg)

    def residuals(p: np.ndarray) -> np.ndarray:
        joints = forward(p)                       # [J, 3]
        diff = (joints - target)[mask]            # [Mv, 3]
        data_res = diff.reshape(-1)
        prior_res = np.sqrt(cfg.pose_prior_weight) * p[bp_slice]
        return np.concatenate([data_res, prior_res])

    sol = least_squares(
        residuals, init_params, method="trf", loss=cfg.loss,
        max_nfev=cfg.max_iter * 10, x_scale="jac",
    )
    joints_final = forward(sol.x)
    err = np.linalg.norm(joints_final[mask] - target[mask], axis=-1)
    return sol.x, float(np.sqrt(np.mean(err * err)))


def fuse_sequence(
    target_joints: np.ndarray,    # [T, J, 3]
    valid: np.ndarray,            # [T, J]
    init_params: np.ndarray,      # [D] — warm-start for frame 0
    forward: ForwardFn,
    cfg: SMPLXFitConfig,
) -> FitResult:
    """Fit SMPL-X per frame, warm-starting each frame from the previous
    frame's solution. Raises :class:`PoseFusionError` if the fraction of
    successfully fit frames is below ``cfg.min_frame_validity_frac``.
    """
    target_joints = np.asarray(target_joints, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    T, J, _ = target_joints.shape
    assert valid.shape == (T, J), "valid must be [T, J]"

    bp_dim = cfg.body_pose_dim
    go_dim = cfg.global_orient_dim
    tr_dim = cfg.transl_dim
    body_pose = np.zeros((T, bp_dim), dtype=np.float64)
    global_orient = np.zeros((T, go_dim), dtype=np.float64)
    transl = np.zeros((T, tr_dim), dtype=np.float64)
    valid_frames = np.zeros(T, dtype=bool)
    rms = np.full(T, np.nan, dtype=np.float64)

    warm = init_params.copy()
    for t in range(T):
        j_valid = valid[t]
        if j_valid.sum() < cfg.min_valid_joints:
            continue
        try:
            params, res_rms = fit_single_frame(
                target_joints[t], j_valid, warm, forward, cfg
            )
        except Exception as exc:
            _LOG.warning(f"fit_single_frame failed at t={t}: {exc}")
            continue
        bp, go, tr = _unpack_params(params, cfg)
        body_pose[t] = bp
        global_orient[t] = go
        transl[t] = tr
        valid_frames[t] = True
        rms[t] = res_rms
        warm = params   # next frame starts from this solution

    frac = float(valid_frames.sum() / T) if T > 0 else 0.0
    if frac < cfg.min_frame_validity_frac:
        raise PoseFusionError(
            f"only {frac*100:.1f}% of frames produced a valid fit "
            f"(threshold {cfg.min_frame_validity_frac*100:.0f}%)"
        )

    return FitResult(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        valid_frames=valid_frames,
        residual_rms_m=rms,
    )


# --- Trivial rigid-body forward for tests ----------------------------------

def rigid_translation_forward(template_joints: np.ndarray, cfg: SMPLXFitConfig) -> ForwardFn:
    """Return a trivial forward used only by tests: joints = template + transl.

    ``body_pose`` and ``global_orient`` are ignored by this forward — the only
    degrees of freedom are the 3-vector ``transl``. Useful for verifying the
    optimizer plumbing without SMPL-X weights.
    """
    template = np.asarray(template_joints, dtype=np.float64)
    if template.shape != (cfg.num_body_joints, 3):
        raise ValueError(
            f"template_joints must be [{cfg.num_body_joints}, 3], got {template.shape}"
        )
    bp_slice, go_slice, tr_slice = _param_slices(cfg)

    def forward(p: np.ndarray) -> np.ndarray:
        transl = p[tr_slice]
        return template + transl[None, :]

    return forward
