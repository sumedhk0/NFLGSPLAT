"""Triangulation + pose-fit + temporal-smooth tests.

Everything here is CPU-only and free of SMPL-X weights. The plan calls for
<5 cm reconstruction error on synthetic joints; we demand <1 cm since the
fixture is noiseless.
"""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.errors import PoseFusionError
from nfl_gsplat.pose.fuse_smplx import (
    SMPLXFitConfig,
    fit_single_frame,
    fuse_sequence,
    rigid_translation_forward,
)
from nfl_gsplat.pose.temporal_smooth import (
    OneEuroConfig,
    interpolate_short_gaps,
    one_euro_1d,
    smooth_param_sequence,
)
from nfl_gsplat.pose.triangulate import (
    TriangulationConfig,
    triangulate_joints_two_view,
)
from nfl_gsplat.utils.geometry import project_points
from tests.fixtures.generate import (
    PLAYER_ROOTS,
    TEMPLATE_JOINTS_22,
    _endzone_camera,
    _sideline_camera,
)


# --- Triangulation ----------------------------------------------------------

def _observations_for_player(root_xyz: np.ndarray, num_frames: int = 5):
    """Project a T-pose player through both fixture cameras and pack into the
    triangulate.py observation dict."""
    intr_s, pose_s = _sideline_camera()
    intr_e, pose_e = _endzone_camera()
    cameras = {
        "sideline": (intr_s, pose_s),
        "endzone":  (intr_e, pose_e),
    }

    joints_world = root_xyz[None, None, :] + TEMPLATE_JOINTS_22[None, :, :]
    joints_world = np.broadcast_to(joints_world, (num_frames,) + TEMPLATE_JOINTS_22.shape)

    obs = {}
    for cam, (intr, pose) in cameras.items():
        K, R, t = intr.K(), pose.R, pose.t
        flat = joints_world.reshape(-1, 3)
        uv = project_points(flat, K, R, t).reshape(num_frames, -1, 2)
        conf = np.full(uv.shape[:2], 0.9, dtype=np.float64)
        obs[cam] = {"uv": uv, "conf": conf}
    return obs, cameras, joints_world


def test_triangulate_two_view_recovers_joints_under_one_cm():
    obs, cams, gt = _observations_for_player(PLAYER_ROOTS[0], num_frames=3)
    cfg = TriangulationConfig(reproj_px_max=1.0, conf_min=0.5)
    res = triangulate_joints_two_view(obs, cams, cfg)
    assert res.valid.all(), "every joint should be valid on noiseless input"
    err = np.linalg.norm(res.joints3d - gt, axis=-1)
    assert err.max() < 0.01, f"worst-joint reconstruction error {err.max():.4f} m"


def test_triangulate_rejects_low_confidence():
    obs, cams, _ = _observations_for_player(PLAYER_ROOTS[0], num_frames=2)
    obs["sideline"]["conf"][:, 5] = 0.1       # kill joint 5 on sideline
    cfg = TriangulationConfig(reproj_px_max=20.0, conf_min=0.3)
    res = triangulate_joints_two_view(obs, cams, cfg)
    assert not res.valid[:, 5].any(), "low-conf joint must be invalidated"
    # Other joints are unaffected.
    assert res.valid[:, 0].all() and res.valid[:, 21].all()


def test_triangulate_rejects_high_reprojection():
    obs, cams, _ = _observations_for_player(PLAYER_ROOTS[1], num_frames=1)
    obs["endzone"]["uv"][0, 3] += np.array([50.0, 50.0])   # 50 px shove on joint 3
    cfg = TriangulationConfig(reproj_px_max=5.0, conf_min=0.3)
    res = triangulate_joints_two_view(obs, cams, cfg)
    assert not res.valid[0, 3], "joint with huge reproj error must be rejected"


# --- SMPL-X fit (trivial forward) ------------------------------------------

def test_fit_single_frame_recovers_translation():
    cfg = SMPLXFitConfig(min_valid_joints=10, max_iter=50)
    fwd = rigid_translation_forward(TEMPLATE_JOINTS_22, cfg)
    true_transl = np.array([3.0, -2.0, 0.92])
    target = TEMPLATE_JOINTS_22 + true_transl[None, :]
    valid = np.ones(cfg.num_body_joints, dtype=bool)

    init = np.zeros(cfg.body_pose_dim + cfg.global_orient_dim + cfg.transl_dim)
    params, rms = fit_single_frame(target, valid, init, fwd, cfg)

    tr = params[cfg.body_pose_dim + cfg.global_orient_dim:]
    np.testing.assert_allclose(tr, true_transl, atol=1e-4)
    assert rms < 1e-4


def test_fit_single_frame_handles_missing_joints():
    cfg = SMPLXFitConfig(min_valid_joints=10, max_iter=50)
    fwd = rigid_translation_forward(TEMPLATE_JOINTS_22, cfg)
    true_transl = np.array([1.0, 1.0, 0.0])
    target = TEMPLATE_JOINTS_22 + true_transl[None, :]
    target[5:12] = np.nan       # simulate missing limb data
    valid = np.ones(cfg.num_body_joints, dtype=bool)
    valid[5:12] = False

    init = np.zeros(cfg.body_pose_dim + cfg.global_orient_dim + cfg.transl_dim)
    params, rms = fit_single_frame(target, valid, init, fwd, cfg)

    tr = params[cfg.body_pose_dim + cfg.global_orient_dim:]
    np.testing.assert_allclose(tr, true_transl, atol=1e-4)


def test_fuse_sequence_propagates_warm_start():
    cfg = SMPLXFitConfig(min_valid_joints=10, max_iter=30,
                        min_frame_validity_frac=1.0)
    fwd = rigid_translation_forward(TEMPLATE_JOINTS_22, cfg)
    T = 6
    transls = np.array([[0.1 * t, 0.05 * t, 0.92] for t in range(T)])
    target = TEMPLATE_JOINTS_22[None, :, :] + transls[:, None, :]
    valid = np.ones((T, cfg.num_body_joints), dtype=bool)
    init = np.zeros(cfg.body_pose_dim + cfg.global_orient_dim + cfg.transl_dim)

    res = fuse_sequence(target, valid, init, fwd, cfg)
    np.testing.assert_allclose(res.transl, transls, atol=1e-3)
    assert res.valid_frames.all()
    assert (res.residual_rms_m < 1e-3).all()


def test_fuse_sequence_raises_when_too_few_frames_fit():
    cfg = SMPLXFitConfig(min_valid_joints=10,
                        min_frame_validity_frac=0.9)
    fwd = rigid_translation_forward(TEMPLATE_JOINTS_22, cfg)
    T = 10
    target = np.tile(TEMPLATE_JOINTS_22[None, :, :], (T, 1, 1))
    valid = np.ones((T, cfg.num_body_joints), dtype=bool)
    valid[2:] = False                  # kill 80% of frames
    init = np.zeros(cfg.body_pose_dim + cfg.global_orient_dim + cfg.transl_dim)

    with pytest.raises(PoseFusionError):
        fuse_sequence(target, valid, init, fwd, cfg)


# --- Temporal smoothing ----------------------------------------------------

def test_one_euro_reduces_noise_on_stationary_signal():
    """When the signal is (nearly) stationary, the 1€ filter should strongly
    attenuate high-frequency noise. Tested after a warm-up window to exclude
    the filter's initial transient."""
    rng = np.random.default_rng(0)
    T = 400
    clean = np.full(T, 0.5)
    noisy = clean + rng.normal(0, 0.05, T)
    smoothed = one_euro_1d(noisy, OneEuroConfig(min_cutoff=0.5, beta=0.0))
    warm = T // 4
    noise_before = float(np.std(noisy[warm:] - clean[warm:]))
    noise_after = float(np.std(smoothed[warm:] - clean[warm:]))
    assert noise_after < 0.5 * noise_before, (
        f"1€ should more than halve stationary noise: {noise_before:.3f} → {noise_after:.3f}"
    )


def test_one_euro_tracks_slow_signal():
    """At low signal frequencies the filter should pass the signal through
    with only a modest phase lag — output and clean signal stay close."""
    T = 300
    t = np.arange(T) / 30.0
    clean = np.sin(2 * np.pi * 0.2 * t)          # 0.2 Hz << 1 Hz cutoff
    smoothed = one_euro_1d(clean, OneEuroConfig(min_cutoff=1.0, beta=0.0))
    warm = T // 4
    err = float(np.max(np.abs(smoothed[warm:] - clean[warm:])))
    assert err < 0.2, f"slow signal should pass through with small error, got {err:.3f}"


def test_smooth_param_sequence_preserves_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (50, 63))
    y = smooth_param_sequence(x, OneEuroConfig())
    assert y.shape == x.shape


def test_interpolate_short_gaps_fills_small_runs():
    values = np.array([0.0, 1.0, np.nan, np.nan, 4.0, 5.0])
    valid = np.array([True, True, False, False, True, True])
    filled, new_valid = interpolate_short_gaps(values, valid, max_gap=2)
    assert new_valid.all()
    np.testing.assert_allclose(filled, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def test_interpolate_short_gaps_leaves_long_gaps():
    values = np.array([0.0, np.nan, np.nan, np.nan, np.nan, 5.0])
    valid = np.array([True, False, False, False, False, True])
    filled, new_valid = interpolate_short_gaps(values, valid, max_gap=2)
    assert not new_valid[1:5].any()
    assert np.isnan(filled[1:5]).all()
