"""Persist the Kalman ball track for the renderer.

The ball stage runs detection + the 3D Kalman filter, then writes ``ball.npz``
with the per-frame world position, velocity, and visibility. ``scripts/05``
loads it and orients the canonical football asset along ``vel`` each frame.

Schema (``ball.npz``)::

    xyz      [T, 3]   world meters (NaN where the filter never emitted)
    vel      [T, 3]   m/s (NaN likewise)
    visible  [T]      bool
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from nfl_gsplat.ball.kalman_3d import BallKalmanConfig, run_kalman
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
from nfl_gsplat.utils.io import read_npz, write_npz


def write_ball_npz(path: Path | str, xyz: np.ndarray, vel: np.ndarray,
                   visible: np.ndarray) -> Path:
    write_npz(
        path,
        xyz=np.asarray(xyz, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        visible=np.asarray(visible, dtype=bool),
    )
    return Path(path)


def read_ball_npz(path: Path | str) -> dict[str, np.ndarray]:
    return read_npz(path)


def build_and_write_ball_track(
    path: Path | str,
    detections_per_frame: Sequence[Mapping[str, np.ndarray]],
    cameras: Mapping[str, tuple[CameraIntrinsics, CameraPose]],
    cfg: BallKalmanConfig,
) -> Path:
    """Run the 3D Kalman filter (with velocity) and persist the track."""
    xyz, vel, visible = run_kalman(detections_per_frame, cameras, cfg, return_velocity=True)
    return write_ball_npz(path, xyz, vel, visible)
