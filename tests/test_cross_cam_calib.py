"""Cross-camera endzone calibration. Synthetic 2-camera scenes, no video/YOLO."""
import numpy as np

from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points


def _look_at(C, target):
    fwd = target - C; fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _cam(C, target, f=2000.0, wh=(1920, 1080)):
    R = _look_at(np.asarray(C, float), np.asarray(target, float))
    t = -R @ np.asarray(C, float)
    K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
    return K, R, t


def test_match_frame_recovers_correct_pairs():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0], [-30.0, -8.0, 0.0], [-20.0, 2.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    uv = project_points(world, K, R, t)                 # exact endzone pixels
    # shuffle the detections; matcher must re-pair them to the right world pts
    perm = [2, 0, 1]
    mw, mu = match_frame(world, uv[perm], K, R, t, max_px=5.0)
    assert mw.shape == (3, 3) and mu.shape == (3, 2)
    # each returned world point's projection equals its matched uv
    proj = project_points(mw, K, R, t)
    assert np.abs(proj - mu).max() < 1e-6


def test_match_frame_distance_gate_drops_outlier():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0], [-30.0, -8.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    uv = project_points(world, K, R, t)
    uv = np.vstack([uv, [10.0, 10.0]])                  # a detection with no world match
    mw, mu = match_frame(world, uv, K, R, t, max_px=5.0)
    assert mw.shape == (2, 3)                            # the 2 real players only


def test_match_frame_far_projection_no_match():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    mw, mu = match_frame(world, np.array([[5.0, 5.0]]), K, R, t, max_px=5.0)
    assert mw.shape == (0, 3) and mu.shape == (0, 2)
