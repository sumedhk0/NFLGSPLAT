"""Virtual-camera trajectory sampling.

Consumes a YAML like ``configs/trajectories/fly_through.yaml`` and emits
per-frame intrinsics + ``CameraPose`` list. Interpolation is piecewise-linear
in position and look-at; the resulting camera always has OpenCV convention
(``R``, ``t`` with ``x_cam = R @ x_world + t``).

Keyframe schema::

    trajectory:
      fps: 30
      mode: linear | bezier   (bezier currently falls through to linear)
      duration_s: 3.0          (optional; inferred from fps + num_frames if absent)
      num_frames: 90           (optional; else ceil(duration_s * fps))
      keyframes:
        - {t: 0.0, position_m: [...], look_at_m: [...], up: [...]}
        - ...                  (t in [0, 1] along the trajectory)

    intrinsics:
      fov_y_deg: 40.0
      width: 1920
      height: 1080
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf

from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV look-at (camera forward = +Z_cam, right = +X_cam, down = +Y_cam).

    Returns ``(R, t)`` with ``x_cam = R @ x_world + t``.
    """
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    R = np.stack([r, -u, f], axis=0)
    t = -R @ eye
    return R, t


def _interp_keyframes(kfs: Iterable[dict], ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kfs = sorted(list(kfs), key=lambda k: float(k["t"]))
    kf_ts = np.array([float(k["t"]) for k in kfs])
    pos = np.array([k["position_m"] for k in kfs], dtype=np.float64)
    tgt = np.array([k["look_at_m"]  for k in kfs], dtype=np.float64)
    up  = np.array([k.get("up", [0.0, 0.0, 1.0]) for k in kfs], dtype=np.float64)

    out_pos = np.empty((ts.shape[0], 3)); out_tgt = np.empty_like(out_pos); out_up = np.empty_like(out_pos)
    for i, t in enumerate(ts):
        t = float(np.clip(t, kf_ts[0], kf_ts[-1]))
        j = int(np.searchsorted(kf_ts, t, side="right") - 1)
        j = max(0, min(j, len(kf_ts) - 2))
        dt = kf_ts[j + 1] - kf_ts[j]
        alpha = 0.0 if dt <= 0 else (t - kf_ts[j]) / dt
        out_pos[i] = (1 - alpha) * pos[j] + alpha * pos[j + 1]
        out_tgt[i] = (1 - alpha) * tgt[j] + alpha * tgt[j + 1]
        out_up[i]  = (1 - alpha) * up[j]  + alpha * up[j + 1]
    return out_pos, out_tgt, out_up


def sample_trajectory(yaml_path: Path | str) -> tuple[CameraIntrinsics, list[CameraPose]]:
    cfg = OmegaConf.load(yaml_path)
    traj = cfg.trajectory
    intr_cfg = cfg.intrinsics
    fps = float(traj.get("fps", 30.0))
    if "num_frames" in traj:
        N = int(traj.num_frames)
    else:
        duration = float(traj.get("duration_s", 3.0))
        N = int(np.ceil(duration * fps))

    width = int(intr_cfg.width); height = int(intr_cfg.height)
    fov_y = float(intr_cfg.fov_y_deg) * np.pi / 180.0
    fy = 0.5 * height / np.tan(0.5 * fov_y)
    fx = fy
    intr = CameraIntrinsics(fx=fx, fy=fy,
                            cx=width / 2.0, cy=height / 2.0,
                            width=width, height=height)

    ts = np.linspace(0.0, 1.0, N)
    pos, tgt, up = _interp_keyframes(traj.keyframes, ts)
    poses: list[CameraPose] = []
    for p, g, u in zip(pos, tgt, up):
        R, t = _look_at(p, g, u)
        poses.append(CameraPose(R=R, t=t))
    return intr, poses
