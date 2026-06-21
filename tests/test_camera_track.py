from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration.cameras_io import (
    CameraTrack, load_camera_track, write_camera_track,
)
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose


def _track(T=5, W=1920, H=1080):
    K = np.stack([np.array([[2600.0 + i, 0, W / 2], [0, 2600.0 + i, H / 2], [0, 0, 1]])
                  for i in range(T)])
    R = np.stack([np.eye(3) for _ in range(T)])
    t = np.stack([np.array([0.0, 0.0, 20.0 + i]) for i in range(T)])
    conf = np.ones(T)
    return CameraTrack(K=K, R=R, t=t, conf=conf, width=W, height=H)


def test_camera_track_at_returns_frame_slice():
    tr = _track()
    intr, pose = tr.at(3)
    assert isinstance(intr, CameraIntrinsics) and isinstance(pose, CameraPose)
    assert intr.fx == 2603.0
    assert pose.t[2] == 23.0
    assert (intr.width, intr.height) == (1920, 1080)


def test_camera_track_at_clamps_out_of_range():
    tr = _track(T=5)
    assert tr.at(99)[1].t[2] == 24.0


def test_write_load_roundtrip(tmp_path):
    tracks = {"sideline": _track(), "endzone": _track()}
    p = tmp_path / "cameras.npz"
    write_camera_track(p, tracks, fps=59.94)
    loaded = load_camera_track(p)
    assert set(loaded) == {"sideline", "endzone"}
    np.testing.assert_allclose(loaded["sideline"].K, tracks["sideline"].K)
    assert loaded["sideline"].at(0)[0].fx == 2600.0


def test_load_missing_raises(tmp_path):
    with pytest.raises(SetupError, match="cameras.npz"):
        load_camera_track(tmp_path / "nope.npz")
