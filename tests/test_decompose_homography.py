from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.decompose_homography import (
    homography_to_krt, krt_to_homography,
)
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points


def _krt(fx, yaw_deg, pitch_deg, cam_height, W=1920, H=1080):
    intr = CameraIntrinsics(fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H)
    ry, rx = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Rz = np.array([[np.cos(ry), -np.sin(ry), 0], [np.sin(ry), np.cos(ry), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    R = Rx @ Rz @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    cam_center = np.array([0.0, 40.0, cam_height])
    t = -R @ cam_center
    return intr, CameraPose(R=R, t=t)


def test_krt_homography_roundtrip_recovers_params():
    intr, pose = _krt(fx=2600.0, yaw_deg=8.0, pitch_deg=22.0, cam_height=18.0)
    H = krt_to_homography(intr.K(), pose.R, pose.t)
    K2, R2, t2 = homography_to_krt(H, width=intr.width, height=intr.height)
    assert abs(K2[0, 0] - intr.fx) / intr.fx < 0.01
    field_pts = np.array([[0, 0, 0], [20, 10, 0], [-30, -15, 0], [45, 20, 0]], float)
    uv_ref = project_points(field_pts, intr.K(), pose.R, pose.t)
    uv_dec = project_points(field_pts, K2, R2, t2)
    assert np.allclose(uv_ref, uv_dec, atol=1.0)


def test_homography_to_krt_returns_proper_rotation():
    intr, pose = _krt(fx=3000.0, yaw_deg=-5.0, pitch_deg=30.0, cam_height=20.0)
    H = krt_to_homography(intr.K(), pose.R, pose.t)
    _, R2, _ = homography_to_krt(H, width=intr.width, height=intr.height)
    assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R2) - 1.0) < 1e-6
