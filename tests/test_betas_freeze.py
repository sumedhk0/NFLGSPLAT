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
