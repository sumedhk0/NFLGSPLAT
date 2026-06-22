from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.calibration.run_autocalib import assemble_track_from_results
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose


def _res(fx, z):
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    return CalibrationResult(
        intrinsics=CameraIntrinsics(fx, fx, 960, 540, 1920, 1080),
        pose=CameraPose(R=np.eye(3), t=np.array([0.0, 0.0, z])),
        rms_px=1.0, num_correspondences=8, refined_with_ba=True,
    )


def test_assemble_fills_short_gap():
    results = [_res(2600, 20.0), None, _res(2604, 22.0)]
    tr = assemble_track_from_results(results, width=1920, height=1080, max_gap=2)
    assert isinstance(tr, CameraTrack)
    assert tr.num_frames == 3
    assert np.isfinite(tr.K).all() and np.isfinite(tr.t).all()


def test_assemble_fails_loud_on_long_gap():
    results = [_res(2600, 20.0), None, None, None, _res(2604, 22.0)]
    with pytest.raises(CalibrationError, match="frames 1-3"):
        assemble_track_from_results(results, width=1920, height=1080, max_gap=2)
