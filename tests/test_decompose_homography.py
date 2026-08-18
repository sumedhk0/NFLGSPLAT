from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.calibration.decompose_homography import (
    homography_to_krt,
    krt_to_homography,
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


def test_focal_recovers_for_a_centerline_camera_from_a_fitted_homography():
    """fwd_y == 0 (camera on the field centerline -- the nominal endzone rig
    pose) sends denom_o (the r1.r2 orthogonality term) toward 0 exactly. On
    the OLD absolute `abs(denom) > 1e-12` gate, the float residue left by
    fitting H with RANSAC (rather than using the analytic H) still cleared
    that threshold, so a 0/0-shaped blowup (~1.14e6) got averaged into the
    good estimate and corrupted the focal (measured: 1990 instead of 2600).
    The homography here MUST come from a real RANSAC fit to projected
    points -- the analytic H has denom_o == 0.0 exactly and never exercises
    the bug."""
    wh = (1920, 1080)
    C = np.array([-112.0, 0.0, 24.0])
    fwd = np.array([1.0, 0.0, -0.2]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); t = -R @ C
    assert fwd[1] == 0.0        # this is the degenerate pose, by construction
    K = CameraIntrinsics(2600.0, 2600.0, wh[0] / 2, wh[1] / 2, *wh).K()

    world, uv = [], []
    for X in (-18.288, -13.716, -9.144, -4.572, 0.0):
        for Y in (-20.0, 20.0):
            q = project_points(np.array([[X, Y, 0.0]]), K, R, t)[0]
            if np.isfinite(q).all():
                world.append([X, Y, 0.0]); uv.append(q)
    world = np.array(world); uv = np.array(uv)

    H, _mask = cv2.findHomography(world[:, :2], uv, cv2.RANSAC, 3.0)
    K2, _R2, _t2 = homography_to_krt(H, width=wh[0], height=wh[1])
    assert abs(K2[0, 0] - 2600.0) / 2600.0 < 0.02, f"got fx={K2[0, 0]}"
