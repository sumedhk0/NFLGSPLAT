"""Canonical football asset generation + per-frame orientation (Layer E),
plus the Kalman velocity output that drives it.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.ball.ball_asset import (
    FootballAssetConfig,
    make_football_asset,
    orient_ball,
    principal_axis,
)
from nfl_gsplat.ball.kalman_3d import BallKalmanConfig, run_kalman
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points


def test_asset_has_expected_keys_and_is_elongated():
    asset = make_football_asset()
    assert set(asset.keys()) == {"xyz", "rot", "scale", "opacity", "sh"}
    xyz = asset["xyz"]
    assert xyz.shape == (FootballAssetConfig().num_gaussians, 3)
    # Prolate along +X: spread on X exceeds spread on Y and Z.
    std = xyz.std(axis=0)
    assert std[0] > std[1] and std[0] > std[2]
    # Canonical long axis is +X.
    assert abs(principal_axis(xyz)[0]) > 0.9


def test_orient_translates_to_position():
    asset = make_football_asset()
    pos = np.array([5.0, 2.0, 1.5])
    posed = orient_ball(asset, pos, np.array([0.0, 3.0, 0.0]), t=0.0)
    assert np.allclose(posed["xyz"].mean(axis=0), pos, atol=1e-4)


def test_orient_aligns_long_axis_to_velocity():
    asset = make_football_asset()
    vel = np.array([0.0, 4.0, 0.0])           # flying along +Y
    posed = orient_ball(asset, np.zeros(3), vel, t=0.0)
    axis = principal_axis(posed["xyz"])
    d = vel / np.linalg.norm(vel)
    assert abs(abs(np.dot(axis, d)) - 1.0) < 1e-2


def test_spin_preserves_long_axis_direction():
    asset = make_football_asset()
    vel = np.array([2.0, 0.0, 1.0])
    d = vel / np.linalg.norm(vel)
    for t in (0.0, 0.1, 0.5, 1.0):
        posed = orient_ball(asset, np.zeros(3), vel, t=t, spin_rate=10.0)
        axis = principal_axis(posed["xyz"])
        assert abs(abs(np.dot(axis, d)) - 1.0) < 1e-2, f"long axis drifted at t={t}"


def test_orient_zero_velocity_is_identity_orientation():
    asset = make_football_asset()
    posed = orient_ball(asset, np.array([1.0, 1.0, 1.0]), np.zeros(3))
    # No velocity → no rotation; long axis stays +X.
    assert abs(abs(principal_axis(posed["xyz"])[0]) - 1.0) < 1e-2


# --- Kalman velocity output -------------------------------------------------

def _cam(cx_off: float = 0.0) -> tuple[CameraIntrinsics, CameraPose]:
    intr = CameraIntrinsics(fx=1400, fy=1400, cx=960, cy=540, width=1920, height=1080)
    R = np.eye(3)
    t = np.array([cx_off, 0.0, 60.0])
    return intr, CameraPose(R=R, t=t)


def test_run_kalman_returns_velocity_when_requested():
    # Two cameras looking down -Z-ish; a ball moving along +X at constant speed.
    cam_a = _cam(-5.0)
    cam_b = _cam(5.0)
    cams = {"a": cam_a, "b": cam_b}
    T = 12
    speed = 0.5
    dets = []
    for f in range(T):
        p = np.array([[-2.0 + speed * f, 0.0, 3.0]])
        ua = project_points(p, cam_a[0].K(), cam_a[1].R, cam_a[1].t)[0]
        ub = project_points(p, cam_b[0].K(), cam_b[1].R, cam_b[1].t)[0]
        dets.append({"a": ua, "b": ub})
    cfg = BallKalmanConfig(fps=30.0, triangulated_pos_std=0.02)

    xyz, vel, visible = run_kalman(dets, cams, cfg, return_velocity=True)
    assert xyz.shape == vel.shape == (T, 3)
    # Where visible, velocity points mostly along +X (the motion direction).
    vis = visible & np.isfinite(vel).all(axis=1)
    assert vis.sum() >= 5
    mean_v = vel[vis][-1]
    assert mean_v[0] > abs(mean_v[1]) and mean_v[0] > abs(mean_v[2])
