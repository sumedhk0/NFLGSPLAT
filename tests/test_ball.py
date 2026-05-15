"""3D ball-trajectory Kalman-filter tests.

The plan calls for <0.5 m reconstruction on a synthetic parabolic trajectory.
We also check graceful behavior when one camera loses the ball for several
frames (the filter should coast on dynamics + the other camera's ray).
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.ball.kalman_3d import BallKalmanConfig, run_kalman
from nfl_gsplat.utils.geometry import project_points
from tests.fixtures.generate import _endzone_camera, _sideline_camera, _ball_trajectory


def _project_to_both_cams(xyz: np.ndarray):
    intr_s, pose_s = _sideline_camera()
    intr_e, pose_e = _endzone_camera()
    cams = {"sideline": (intr_s, pose_s), "endzone": (intr_e, pose_e)}
    uv_s = project_points(xyz, intr_s.K(), pose_s.R, pose_s.t)
    uv_e = project_points(xyz, intr_e.K(), pose_e.R, pose_e.t)
    return cams, uv_s, uv_e


def test_ball_kalman_reconstructs_parabola_under_half_meter():
    gt = _ball_trajectory()                         # [T, 3]
    T = gt.shape[0]
    cams, uv_s, uv_e = _project_to_both_cams(gt)
    dets = [{"sideline": uv_s[t], "endzone": uv_e[t]} for t in range(T)]

    cfg = BallKalmanConfig(fps=30.0, triangulated_pos_std=0.02)
    xyz, visible = run_kalman(dets, cams, cfg)

    assert visible.all(), "noiseless: filter must output every frame"
    err = np.linalg.norm(xyz - gt, axis=-1)
    assert err.max() < 0.5, f"ball max error {err.max():.3f} m > 0.5 m"
    assert err.mean() < 0.2, f"ball mean error {err.mean():.3f} m > 0.2 m"


def test_ball_kalman_coasts_through_single_cam_occlusion():
    gt = _ball_trajectory()
    T = gt.shape[0]
    cams, uv_s, uv_e = _project_to_both_cams(gt)
    dets: list[dict] = []
    for t in range(T):
        d = {"sideline": uv_s[t]}
        # endzone missing for 5 consecutive frames mid-trajectory
        if not (20 <= t < 25):
            d["endzone"] = uv_e[t]
        dets.append(d)

    cfg = BallKalmanConfig(fps=30.0, triangulated_pos_std=0.02, ray_residual_std=0.5)
    xyz, visible = run_kalman(dets, cams, cfg)
    assert visible.all()
    err = np.linalg.norm(xyz - gt, axis=-1)
    # Single-cam coast: looser tolerance but still under 1 m.
    assert err.max() < 1.0, f"ball max error during occlusion {err.max():.3f} m"


def test_ball_kalman_delays_output_until_two_cameras_observe():
    gt = _ball_trajectory()
    T = gt.shape[0]
    cams, uv_s, uv_e = _project_to_both_cams(gt)
    dets: list[dict] = []
    for t in range(T):
        d: dict = {}
        if t >= 3:
            d["sideline"] = uv_s[t]
            d["endzone"] = uv_e[t]
        dets.append(d)

    cfg = BallKalmanConfig(fps=30.0)
    xyz, visible = run_kalman(dets, cams, cfg)
    assert not visible[:3].any(), "no output before the filter has initialized"
    assert visible[3:].all(), "should emit every frame once initialized"
