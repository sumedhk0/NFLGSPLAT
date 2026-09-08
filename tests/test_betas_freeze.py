"""Betas-source policy for the pose stage (Layer C).

Verifies ``resolve_betas`` picks the library value when freezing is on, falls
back to a per-play estimate otherwise, and that the resolved betas actually
drive the rest skeleton the optimizer fits against.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.pose.fuse_smplx import (
    SMPLXFitConfig,
    betas_scaled_template,
    fuse_sequence,
    resolve_betas,
)
from tests.fixtures.generate import TEMPLATE_JOINTS_22


def _estimate_fn(value: float):
    return lambda: np.full(10, value, dtype=np.float64)


def test_resolve_uses_library_when_freezing():
    cfg = SMPLXFitConfig(use_library_betas=True)
    lib = np.full(10, 0.3)
    betas, source = resolve_betas(lib, _estimate_fn(0.9), cfg)
    assert source == "library"
    assert np.allclose(betas, 0.3)


def test_resolve_falls_back_to_estimate_when_no_library():
    cfg = SMPLXFitConfig(use_library_betas=True)
    betas, source = resolve_betas(None, _estimate_fn(0.9), cfg)
    assert source == "estimated"
    assert np.allclose(betas, 0.9)


def test_resolve_estimates_when_freezing_disabled():
    cfg = SMPLXFitConfig(use_library_betas=False)
    lib = np.full(10, 0.3)
    betas, source = resolve_betas(lib, _estimate_fn(0.9), cfg)
    assert source == "estimated"
    assert np.allclose(betas, 0.9)


def test_frozen_betas_drive_the_fit_skeleton():
    """A sequence generated from a TRUE shape is best recovered when the fit
    uses that same (frozen) shape; a wrong frozen shape leaves residual error."""
    cfg = SMPLXFitConfig(min_valid_joints=10, max_iter=60)
    true_betas = np.full(10, 0.25)
    transl = np.array([1.0, -0.5, 0.3])

    true_template = betas_scaled_template(TEMPLATE_JOINTS_22, true_betas)
    target = (true_template + transl[None, :])[None, :, :]      # [1, 22, 3]
    valid = np.ones((1, 22), dtype=bool)
    init = np.zeros(cfg.body_pose_dim + cfg.global_orient_dim + cfg.transl_dim)

    from nfl_gsplat.pose.fuse_smplx import rigid_translation_forward

    # Correct frozen betas → near-zero residual.
    fwd_correct = rigid_translation_forward(
        betas_scaled_template(TEMPLATE_JOINTS_22, true_betas), cfg)
    res_correct = fuse_sequence(target, valid, init, fwd_correct, cfg)
    assert res_correct.residual_rms_m[0] < 1e-3

    # Wrong frozen betas → the rest skeleton is mis-scaled; residual is larger.
    fwd_wrong = rigid_translation_forward(
        betas_scaled_template(TEMPLATE_JOINTS_22, np.full(10, 0.6)), cfg)
    res_wrong = fuse_sequence(target, valid, init, fwd_wrong, cfg)
    assert res_wrong.residual_rms_m[0] > res_correct.residual_rms_m[0]

def test_temporal_term_and_robust_scale_damp_a_one_frame_spike():
    """A wrist target that jumps 0.4 m for one frame and returns: without the
    temporal term and with the loss at 1 m the fit follows the spike; with
    them it does not, and the other frames are untouched."""
    from nfl_gsplat.pose.forward_kinematics import fk_forward
    from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig, _pack_params, fuse_sequence

    rest = np.zeros((22, 3))
    rest[0] = [0, 0.95, 0]
    rest[1], rest[2] = [0.09, 0.9, 0], [-0.09, 0.9, 0]
    rest[3], rest[6], rest[9] = [0, 1.05, 0], [0, 1.18, 0], [0, 1.32, 0]
    rest[4], rest[5] = [0.1, 0.5, 0], [-0.1, 0.5, 0]
    rest[7], rest[8] = [0.1, 0.08, 0], [-0.1, 0.08, 0]
    rest[10], rest[11] = [0.1, 0.0, 0.12], [-0.1, 0.0, 0.12]
    rest[12], rest[15] = [0, 1.45, 0], [0, 1.65, 0]
    rest[13], rest[14] = [0.08, 1.4, 0], [-0.08, 1.4, 0]
    rest[16], rest[17] = [0.2, 1.4, 0], [-0.2, 1.4, 0]
    rest[18], rest[19] = [0.45, 1.4, 0], [-0.45, 1.4, 0]
    rest[20], rest[21] = [0.7, 1.4, 0], [-0.7, 1.4, 0]
    forward = fk_forward(rest)
    truth = _pack_params(np.zeros(63), np.zeros(3), np.array([1.0, 0.0, 2.0]))
    J = forward(truth)
    T = 7
    target = np.stack([J] * T)
    target[3, 20] += [0.0, 0.4, 0.0]                      # one-frame spike on the left wrist
    valid = np.ones((T, 22), bool)
    init = _pack_params(np.zeros(63), np.zeros(3), np.array([1.0, 0.0, 2.0]))
    plain = fuse_sequence(target, valid, init, forward, SMPLXFitConfig(min_valid_joints=6))
    damped = fuse_sequence(target, valid, init, forward,
                           SMPLXFitConfig(min_valid_joints=6, temporal_weight=0.3, f_scale=0.1))
    wrist_plain = forward(_pack_params(plain.body_pose[3], plain.global_orient[3], plain.transl[3]))[20]
    wrist_damped = forward(_pack_params(damped.body_pose[3], damped.global_orient[3], damped.transl[3]))[20]
    # measured on this figure: plain follows 0.28 of the 0.40 m spike, damped 0.16;
    # a real 0.40 m step is followed to 0.26 within four frames under the same damping
    assert abs(wrist_plain[1] - J[20, 1]) > 0.25, "the plain fit follows the spike"
    assert abs(wrist_damped[1] - J[20, 1]) < 0.2, "the damped fit follows it less"
    w1 = forward(_pack_params(damped.body_pose[1], damped.global_orient[1], damped.transl[1]))[20]
    w6 = forward(_pack_params(damped.body_pose[6], damped.global_orient[6], damped.transl[6]))[20]
    assert np.linalg.norm(w1 - J[20]) < 0.03 and np.linalg.norm(w6 - J[20]) < 0.06
