from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_correspondences
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points


def _truth_camera(W=1920, H=1080):
    intr = CameraIntrinsics(980.0, 980.0, W / 2, H / 2, W, H)
    eye = np.array([10.0, -40.0, 12.0])
    target = np.array([10.0, 0.0, 1.0])
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
    down = np.cross(right, fwd)
    R = np.stack([right, -down, fwd])
    t = -R @ eye
    return intr, CameraPose(R=R, t=t)


def test_solve_from_correspondences_recovers_camera():
    intr, pose = _truth_camera()
    names = ["mid_50_left_sideline", "mid_50_right_sideline", "mid_50_left_hash",
             "home_20_left_hash", "home_20_right_sideline", "away_20_left_sideline",
             "away_20_right_hash"]
    pairs = []
    for n in names:
        uv = project_points(NFL_LANDMARKS[n][None], intr.K(), pose.R, pose.t)[0]
        pairs.append((n, (float(uv[0]), float(uv[1]))))
    res = solve_pnp_from_correspondences(pairs, image_size=(intr.width, intr.height))
    assert abs(res.intrinsics.fx - intr.fx) / intr.fx < 0.02
    assert res.rms_px < 2.0


def test_too_few_correspondences_raises():
    import pytest
    from nfl_gsplat.errors import CalibrationError
    with pytest.raises(CalibrationError):
        solve_pnp_from_correspondences([("mid_50_left_hash", (1.0, 2.0))], image_size=(1920, 1080))
