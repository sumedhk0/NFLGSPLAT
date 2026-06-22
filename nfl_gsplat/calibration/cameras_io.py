"""Per-frame camera calibration I/O.

Calibration is per-frame (cameras pan/tilt/zoom during a play). A
:class:`CameraTrack` holds per-frame K/R/t for one camera; ``cameras.npz`` packs
all cameras of a play. ``.at(frame)`` returns the (CameraIntrinsics, CameraPose)
for a given frame (clamped to range, since the two synced clips may differ by a
frame). Produced by scripts/02_autocalibrate.py (or the manual 02b fallback); consumed by every 3D stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose


@dataclass(frozen=True)
class CameraTrack:
    K: np.ndarray        # [T, 3, 3]
    R: np.ndarray        # [T, 3, 3]
    t: np.ndarray        # [T, 3]
    conf: np.ndarray     # [T]
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        return int(self.K.shape[0])

    def at(self, frame: int) -> tuple[CameraIntrinsics, CameraPose]:
        i = max(0, min(int(frame), self.num_frames - 1))
        K = self.K[i]
        intr = CameraIntrinsics(
            fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
            width=int(self.width), height=int(self.height),
        )
        return intr, CameraPose(R=self.R[i].astype(np.float64), t=self.t[i].astype(np.float64))


def write_camera_track(path: Path | str, tracks: dict[str, CameraTrack], *, fps: float) -> Path:
    """Write all cameras to a single cameras.npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cams": np.array(list(tracks.keys())),
        "fps": np.array(float(fps)),
    }
    for cam, tr in tracks.items():
        arrays[f"{cam}_K"] = tr.K.astype(np.float64)
        arrays[f"{cam}_R"] = tr.R.astype(np.float64)
        arrays[f"{cam}_t"] = tr.t.astype(np.float64)
        arrays[f"{cam}_conf"] = tr.conf.astype(np.float64)
        arrays[f"{cam}_wh"] = np.array([tr.width, tr.height], dtype=np.int64)
    np.savez(path, **arrays)
    return path


def load_camera_track(path: Path | str) -> dict[str, CameraTrack]:
    """Load ``{cam: CameraTrack}`` from a cameras.npz. Raises SetupError if missing."""
    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"cameras.npz not found at {path}. Calibrate keyframes "
            "(scripts/02_calibrate_cameras.py --play-dir <dir> --annotate) then run "
            "scripts/02b_track_calibration.py --play-dir <dir>. See SETUP.md §3."
        )
    d = np.load(path, allow_pickle=False)
    cams = [str(c) for c in d["cams"]]
    out: dict[str, CameraTrack] = {}
    for cam in cams:
        wh = d[f"{cam}_wh"]
        out[cam] = CameraTrack(
            K=d[f"{cam}_K"], R=d[f"{cam}_R"], t=d[f"{cam}_t"], conf=d[f"{cam}_conf"],
            width=int(wh[0]), height=int(wh[1]),
        )
    if not out:
        raise SetupError(f"{path}: no cameras found.")
    return out


def constant_track(intr: CameraIntrinsics, pose: CameraPose, num_frames: int) -> CameraTrack:
    """A CameraTrack with one pose repeated — the single-pose shim for testing
    and for static-camera footage."""
    K = np.repeat(intr.K()[None], num_frames, axis=0)
    R = np.repeat(np.asarray(pose.R, float)[None], num_frames, axis=0)
    t = np.repeat(np.asarray(pose.t, float).reshape(1, 3), num_frames, axis=0)
    return CameraTrack(K=K, R=R, t=t, conf=np.ones(num_frames),
                       width=intr.width, height=intr.height)
