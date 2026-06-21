from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration import run_tracking as rt
from nfl_gsplat.calibration.cameras_io import CameraTrack, load_camera_track
from nfl_gsplat.calibration.keyframes import Keyframe, save_keyframes


def test_build_camera_npz_writes_both_cams(tmp_path, monkeypatch):
    for cam in ("sideline", "endzone"):
        save_keyframes(tmp_path / f"{cam}_keyframes.json",
                       [Keyframe(0, {"mid_50_left_hash": (10.0, 20.0)})])
    monkeypatch.setattr(rt, "_anchor_homographies", lambda kfs, wh: {0: np.eye(3)})
    monkeypatch.setattr(rt, "_track", lambda video, anchors, **kw: CameraTrack(
        K=np.repeat(np.eye(3)[None], 4, 0), R=np.repeat(np.eye(3)[None], 4, 0),
        t=np.zeros((4, 3)), conf=np.ones(4), width=1920, height=1080))
    monkeypatch.setattr(rt, "_video_dims_frames", lambda video: (1920, 1080, 4))
    out = rt.build_camera_npz(
        play_dir=tmp_path,
        videos={"sideline": tmp_path / "s.mp4", "endzone": tmp_path / "e.mp4"},
        fps=30.0,
    )
    tracks = load_camera_track(out)
    assert set(tracks) == {"sideline", "endzone"}
    assert tracks["sideline"].num_frames == 4
