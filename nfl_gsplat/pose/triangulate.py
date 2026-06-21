"""Two-view triangulation of body joints with per-joint validity mask.

Inputs (one per camera, same ordering of frames and joints):

- ``uv      [T, J, 2]`` pixel observations
- ``conf    [T, J]``    per-joint regressor confidence

Shared:

- ``cameras {cam: CameraTrack}``

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

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.utils.geometry import (
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
    cameras: Mapping[str, CameraTrack],
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

    joints3d = np.full((T, J, 3), np.nan, dtype=np.float64)
    valid = np.zeros((T, J), dtype=bool)
    reproj = np.full((T, J, 2), np.nan, dtype=np.float64)

    X = np.full((T, J, 3), np.nan, dtype=np.float64)
    err_a = np.full((T, J), np.nan, dtype=np.float64)
    err_b = np.full((T, J), np.nan, dtype=np.float64)
    for t in range(T):
        ia, pa = cameras[a].at(t)
        ib, pb = cameras[b].at(t)
        Ka, Ra, ta = ia.K(), pa.R, pa.t
        Kb, Rb, tb = ib.K(), pb.R, pb.t
        Pa = Ka @ pa.extrinsic_3x4()
        Pb = Kb @ pb.extrinsic_3x4()
        Xt = triangulate_two_views(uv_a[t], uv_b[t], Pa, Pb)      # [J, 3]
        X[t] = Xt
        err_a[t] = np.linalg.norm(project_points(Xt, Ka, Ra, ta) - uv_a[t], axis=-1)
        err_b[t] = np.linalg.norm(project_points(Xt, Kb, Rb, tb) - uv_b[t], axis=-1)
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
