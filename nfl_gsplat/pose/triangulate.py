"""Two-view triangulation of body joints with per-joint validity mask.

Inputs (one per camera, same ordering of frames and joints):

- ``uv      [T, J, 2]`` pixel observations
- ``conf    [T, J]``    per-joint regressor confidence

Shared:

- ``cameras {cam: (CameraIntrinsics, CameraPose)}``

Outputs:

- ``joints3d [T, J, 3]`` triangulated world-frame joints (NaN where invalid)
- ``valid    [T, J]``    bool mask
- ``reproj   [T, J, C]`` per-camera reprojection error in pixels

A (t, j) observation is **invalidated** if:
  * confidence in *any* contributing camera is below ``conf_min``, or
  * reprojection error in *any* contributing camera exceeds ``reproj_px_max``.

This conservative policy is what lets the downstream SMPL-X fit trust the
residuals — one bad joint can pull a whole pose off by tens of centimetres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from nfl_gsplat.utils.geometry import (
    CameraIntrinsics,
    CameraPose,
    project_points,
    triangulate_two_views,
)


@dataclass(frozen=True)
class TriangulationConfig:
    reproj_px_max: float = 20.0
    conf_min: float = 0.3


@dataclass(frozen=True)
class TriangulationResult:
    joints3d: np.ndarray   # [T, J, 3], NaN where invalid
    valid: np.ndarray      # [T, J] bool
    reproj: np.ndarray     # [T, J, C] pixels (NaN where cam not used)


def triangulate_joints_two_view(
    observations: Mapping[str, Mapping[str, np.ndarray]],
    cameras: Mapping[str, tuple[CameraIntrinsics, CameraPose]],
    cfg: TriangulationConfig,
) -> TriangulationResult:
    """Triangulate joints from exactly two synchronized cameras.

    ``observations`` maps ``cam_name → {"uv": [T, J, 2], "conf": [T, J]}``.
    Cameras present in ``observations`` must also be keys of ``cameras``.
    """
    cam_names = list(observations.keys())
    if len(cam_names) != 2:
        raise ValueError(
            f"triangulate_joints_two_view requires exactly 2 cameras, got {len(cam_names)}"
        )
    a, b = cam_names
    uv_a = np.asarray(observations[a]["uv"], dtype=np.float64)
    uv_b = np.asarray(observations[b]["uv"], dtype=np.float64)
    conf_a = np.asarray(observations[a]["conf"], dtype=np.float64)
    conf_b = np.asarray(observations[b]["conf"], dtype=np.float64)
    if uv_a.shape != uv_b.shape:
        raise ValueError(f"cam uv shapes disagree: {uv_a.shape} vs {uv_b.shape}")
    T, J = uv_a.shape[:2]

    intr_a, pose_a = cameras[a]
    intr_b, pose_b = cameras[b]
    Ka, Ra, ta = intr_a.K(), pose_a.R, pose_a.t
    Kb, Rb, tb = intr_b.K(), pose_b.R, pose_b.t
    Pa = Ka @ pose_a.extrinsic_3x4()
    Pb = Kb @ pose_b.extrinsic_3x4()

    joints3d = np.full((T, J, 3), np.nan, dtype=np.float64)
    valid = np.zeros((T, J), dtype=bool)
    reproj = np.full((T, J, 2), np.nan, dtype=np.float64)

    # Triangulate everything in one pass, then mask out by conf + reprojection.
    flat_a = uv_a.reshape(-1, 2)
    flat_b = uv_b.reshape(-1, 2)
    X = triangulate_two_views(flat_a, flat_b, Pa, Pb).reshape(T, J, 3)

    # Reprojection error per camera.
    flat_X = X.reshape(-1, 3)
    uv_a_pred = project_points(flat_X, Ka, Ra, ta).reshape(T, J, 2)
    uv_b_pred = project_points(flat_X, Kb, Rb, tb).reshape(T, J, 2)
    err_a = np.linalg.norm(uv_a_pred - uv_a, axis=-1)
    err_b = np.linalg.norm(uv_b_pred - uv_b, axis=-1)
    reproj[..., 0] = err_a
    reproj[..., 1] = err_b

    ok_conf = (conf_a >= cfg.conf_min) & (conf_b >= cfg.conf_min)
    ok_reproj = np.isfinite(err_a) & np.isfinite(err_b) \
        & (err_a <= cfg.reproj_px_max) & (err_b <= cfg.reproj_px_max)
    ok_in_front = np.isfinite(X).all(axis=-1)
    valid = ok_conf & ok_reproj & ok_in_front

    joints3d[~valid] = np.nan
    joints3d[valid] = X[valid]
    return TriangulationResult(joints3d=joints3d, valid=valid, reproj=reproj)
