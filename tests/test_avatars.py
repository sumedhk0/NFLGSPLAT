"""Avatar tier/LBS tests — CPU only, no LHM or SMPL-X weights required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nfl_gsplat.avatars.lbs_animate import (
    _quat_to_rotmat,
    _rotmat_to_quat,
    animate_gaussians,
)
from nfl_gsplat.avatars.lhm_wrapper import (
    LHMConfig,
    pick_tier,
    write_mock_avatar,
)
from nfl_gsplat.errors import LHMVRAMError
from nfl_gsplat.utils.io import read_npz


# --- VRAM-tier policy ------------------------------------------------------

def test_pick_tier_lhm_1b_at_24gb():
    assert pick_tier(LHMConfig(model_choice="auto"), free_gb=24.0) == "lhm_1b"


def test_pick_tier_lhm_mini_at_12gb():
    assert pick_tier(LHMConfig(model_choice="auto"), free_gb=12.0) == "lhm_mini"


def test_pick_tier_raises_below_floor():
    with pytest.raises(LHMVRAMError):
        pick_tier(LHMConfig(model_choice="auto", vram_floor_gb=8.0), free_gb=4.0)


def test_pick_tier_explicit_lhm_1b_requires_16gb():
    with pytest.raises(LHMVRAMError):
        pick_tier(LHMConfig(model_choice="lhm_1b"), free_gb=10.0)


def test_pick_tier_explicit_mini_allowed_at_floor():
    assert pick_tier(LHMConfig(model_choice="lhm_mini", vram_floor_gb=8.0),
                    free_gb=8.0) == "lhm_mini"


# --- Mock avatar schema ----------------------------------------------------

def test_mock_avatar_schema(tmp_path: Path):
    out = write_mock_avatar(tmp_path / "p0.npz", num_gaussians=1200, num_joints=22, seed=0)
    data = read_npz(out)
    assert data["canonical_xyz"].shape == (1200, 3)
    assert data["canonical_rot"].shape == (1200, 4)
    assert data["canonical_scale"].shape == (1200, 3)
    assert data["canonical_opacity"].shape == (1200,)
    assert data["canonical_sh"].shape[:2] == (1200, 3)
    assert data["lbs_weights"].shape == (1200, 22)
    np.testing.assert_allclose(data["lbs_weights"].sum(axis=1), 1.0)


# --- LBS animation ---------------------------------------------------------

def test_animate_gaussians_identity_is_noop():
    N, J = 50, 5
    rng = np.random.default_rng(0)
    xyz = rng.normal(0, 1, (N, 3))
    q = np.tile(np.array([1, 0, 0, 0], dtype=np.float64), (N, 1))
    lbs = np.zeros((N, J))
    lbs[:, 0] = 1.0
    tfm = np.tile(np.eye(4)[None, :, :], (J, 1, 1))
    xyz_w, q_w = animate_gaussians(xyz, q, lbs, tfm)
    np.testing.assert_allclose(xyz_w, xyz, atol=1e-10)


def test_animate_gaussians_pure_translation():
    N, J = 30, 5
    xyz = np.zeros((N, 3))
    q = np.tile(np.array([1, 0, 0, 0], dtype=np.float64), (N, 1))
    lbs = np.zeros((N, J)); lbs[:, 2] = 1.0
    tfm = np.tile(np.eye(4)[None, :, :], (J, 1, 1))
    tfm[2, :3, 3] = np.array([5.0, -3.0, 2.0])
    xyz_w, _ = animate_gaussians(xyz, q, lbs, tfm)
    np.testing.assert_allclose(xyz_w, np.array([5.0, -3.0, 2.0])[None, :].repeat(N, axis=0))


def test_animate_gaussians_rotation_around_z_90deg():
    N, J = 10, 3
    xyz = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
    q = np.tile(np.array([1, 0, 0, 0], dtype=np.float64), (N, 1))
    lbs = np.zeros((N, J)); lbs[:, 1] = 1.0
    # 90° about +Z world.
    Rz = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    tfm = np.tile(np.eye(4)[None, :, :], (J, 1, 1))
    tfm[1] = Rz
    xyz_w, q_w = animate_gaussians(xyz, q, lbs, tfm)
    np.testing.assert_allclose(xyz_w, np.tile(np.array([0.0, 1.0, 0.0]), (N, 1)), atol=1e-10)
    # Quaternion for 90° rot about Z: wxyz = (cos(45°), 0, 0, sin(45°)).
    expected_q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    for qi in q_w:
        # Quaternion sign ambiguity: ±q are the same rotation.
        if qi[0] < 0:
            qi = -qi
        np.testing.assert_allclose(qi, expected_q, atol=1e-8)


def test_quat_rotmat_roundtrip():
    rng = np.random.default_rng(42)
    N = 20
    axis = rng.normal(0, 1, (N, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = rng.uniform(-np.pi, np.pi, N)
    q = np.concatenate([
        np.cos(angle / 2)[:, None],
        axis * np.sin(angle / 2)[:, None],
    ], axis=1)
    R = _quat_to_rotmat(q)
    q_back = _rotmat_to_quat(R)
    # Normalize sign: align to input hemisphere.
    flip = np.sign((q * q_back).sum(axis=1))
    flip[flip == 0] = 1
    q_back = q_back * flip[:, None]
    np.testing.assert_allclose(q_back, q, atol=1e-8)


def test_animate_blended_weights_interpolates_linearly_for_pure_translation():
    """When two joints have pure translation transforms and a Gaussian has
    50/50 LBS weights, the animated position equals the midpoint. This is a
    fundamental LBS property the compositor depends on."""
    N, J = 1, 2
    xyz = np.zeros((N, 3))
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    lbs = np.array([[0.5, 0.5]])
    tfm = np.tile(np.eye(4)[None, :, :], (J, 1, 1))
    tfm[0, :3, 3] = np.array([1.0, 0.0, 0.0])
    tfm[1, :3, 3] = np.array([0.0, 1.0, 0.0])
    xyz_w, _ = animate_gaussians(xyz, q, lbs, tfm)
    np.testing.assert_allclose(xyz_w[0], np.array([0.5, 0.5, 0.0]))
