from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_features import DetectedFeatures
from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
from nfl_gsplat.calibration.register_frame import register_frame
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points


def _lookat(eye, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - eye; f = f / np.linalg.norm(f)
    r = np.cross(f, up); r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])
    return CameraPose(R=R, t=-R @ eye)


def _truth():
    intr = CameraIntrinsics(1100.0, 1100.0, 960, 540, 1920, 1080)
    pose = _lookat(np.array([-8.0, -42.0, 13.0]), np.array([-8.0, 0.0, 1.0]))
    return intr, pose


def test_register_frame_recovers_camera_from_projected_features(monkeypatch):
    import nfl_gsplat.calibration.register_frame as rf
    intr, pose = _truth()
    names = ["mid_50_left_sideline", "mid_50_right_sideline", "mid_50_left_hash",
             "mid_50_right_hash", "away_20_left_hash", "away_20_right_sideline",
             "home_20_left_sideline", "home_20_right_hash"]
    pairs = []
    for n in names:
        uv = project_points(NFL_LANDMARKS[n][None], intr.K(), pose.R, pose.t)[0]
        if np.isfinite(uv).all():
            pairs.append((n, (float(uv[0]), float(uv[1]))))
    assert len(pairs) >= 6   # the look-at camera must see at least 6 landmarks
    monkeypatch.setattr(rf, "identify_correspondences", lambda feats, prior: (pairs, object()))
    feats = DetectedFeatures([], [], [], [], (1920, 1080))
    res, _state = register_frame(feats, prior=None, image_size=(1920, 1080))
    assert res is not None
    assert res.rms_px < 2.0


def test_register_frame_returns_none_when_too_few(monkeypatch):
    import nfl_gsplat.calibration.register_frame as rf
    monkeypatch.setattr(rf, "identify_correspondences",
                        lambda feats, prior: ([("mid_50_left_hash", (1.0, 2.0))], object()))
    feats = DetectedFeatures([], [], [], [], (1920, 1080))
    res, _ = register_frame(feats, prior=None, image_size=(1920, 1080))
    assert res is None
