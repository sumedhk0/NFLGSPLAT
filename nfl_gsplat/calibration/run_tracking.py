"""Batch per-frame calibration: keyframes -> anchors -> tracked CameraTrack.

Headless (no display). For each camera: solve each keyframe's PnP -> anchor
field->image homography, then track_camera_sequence between anchors, and pack all
cameras into one cameras.npz. The PnP solve + video reads are isolated seams.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
from nfl_gsplat.calibration.decompose_homography import krt_to_homography
from nfl_gsplat.calibration.keyframes import load_keyframes
from nfl_gsplat.calibration.track_homography import TrackConfig, track_camera_sequence


def _video_dims_frames(video: Path) -> tuple[int, int, int]:
    from nfl_gsplat.utils.video import ffprobe_meta
    m = ffprobe_meta(video)
    return m.width, m.height, m.num_frames


def _anchor_homographies(keyframes, wh: tuple[int, int]) -> dict[int, np.ndarray]:
    """Solve each keyframe to a field->image homography via solve_pnp."""
    from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_annotations
    from nfl_gsplat.utils.io import write_json

    width, height = wh
    out: dict[int, np.ndarray] = {}
    for kf in keyframes:
        entries = [{"name": n, "uv": [u, v], "frame": kf.frame}
                   for n, (u, v) in kf.landmarks.items()]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            tmp = Path(fh.name)
        write_json(tmp, entries)
        try:
            res = solve_pnp_from_annotations(tmp, image_size=(width, height))
        finally:
            tmp.unlink(missing_ok=True)
        out[kf.frame] = krt_to_homography(res.intrinsics.K(), res.pose.R, res.pose.t)
    return out


def _track(video, anchors, *, num_frames, width, height, masks_provider, cfg):
    return track_camera_sequence(
        video, anchors, num_frames=num_frames, width=width, height=height,
        masks_provider=masks_provider, cfg=cfg,
    )


def build_camera_npz(
    *, play_dir: Path | str, videos: dict[str, Path], fps: float,
    masks_provider=lambda cam: (lambda frame: []),
    cfg: TrackConfig = TrackConfig(),
) -> Path:
    """Produce cameras.npz for all cameras from their keyframes + clips."""
    play_dir = Path(play_dir)
    tracks: dict[str, CameraTrack] = {}
    for cam, video in videos.items():
        keyframes = load_keyframes(play_dir / f"{cam}_keyframes.json")
        width, height, num_frames = _video_dims_frames(video)
        anchors = _anchor_homographies(keyframes, (width, height))
        tracks[cam] = _track(
            video, anchors, num_frames=num_frames, width=width, height=height,
            masks_provider=masks_provider(cam), cfg=cfg,
        )
    return write_camera_track(play_dir / "cameras.npz", tracks, fps=fps)
