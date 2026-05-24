"""3D Kalman filter for the ball trajectory.

State::

    x = [x, y, z, vx, vy, vz]      (world meters, m/s)

Dynamics — constant velocity + gravity::

    x_{k+1} = F @ x_k + g_vec
    g_vec   = [0, 0, -0.5 g dt², 0, 0, -g dt]    (positional drop + velocity loss)

Measurements — two modes, controlled per-frame by whether each camera has a
detection:

1. **Both cameras detect.** Triangulate the pair of 2D detections to a 3D
   world point, then do a standard KF position update: H = [I₃ | 0].
2. **Exactly one camera detects.** Use an EKF update constrained along the
   back-projected ray: measurement = the two off-ray perpendicular residuals
   (small pose error is fine because the football is tiny and noisy anyway).

If no cameras detect, the filter still propagates forward — this is what lets
the ball trajectory be reconstructed across brief occlusions.

Pure numpy + scipy; safe for CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from nfl_gsplat.utils.geometry import (
    CameraIntrinsics,
    CameraPose,
    project_points,
    triangulate_two_views,
)

GRAVITY = 9.81


@dataclass(frozen=True)
class BallKalmanConfig:
    fps: float = 30.0
    process_pos_std: float = 0.05       # m, per-step position noise
    process_vel_std: float = 0.5        # m/s, per-step velocity noise
    triangulated_pos_std: float = 0.10  # m, measurement sigma for 3D triangulation
    ray_residual_std: float = 0.3       # m, per-axis sigma when only one cam
    init_pos_std: float = 2.0
    init_vel_std: float = 5.0


def _transition_matrix(dt: float) -> np.ndarray:
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def _gravity_control(dt: float) -> np.ndarray:
    return np.array([0.0, 0.0, -0.5 * GRAVITY * dt * dt,
                     0.0, 0.0, -GRAVITY * dt], dtype=np.float64)


def _process_cov(cfg: BallKalmanConfig, dt: float) -> np.ndarray:
    Q = np.diag([
        cfg.process_pos_std ** 2, cfg.process_pos_std ** 2, cfg.process_pos_std ** 2,
        cfg.process_vel_std ** 2, cfg.process_vel_std ** 2, cfg.process_vel_std ** 2,
    ]) * dt
    return Q


def _triangulate_ball_frame(
    uv_a: np.ndarray, uv_b: np.ndarray,
    cam_a: tuple[CameraIntrinsics, CameraPose],
    cam_b: tuple[CameraIntrinsics, CameraPose],
) -> np.ndarray:
    Pa = cam_a[0].K() @ cam_a[1].extrinsic_3x4()
    Pb = cam_b[0].K() @ cam_b[1].extrinsic_3x4()
    return triangulate_two_views(uv_a.reshape(1, 2), uv_b.reshape(1, 2), Pa, Pb)[0]


def _ray_residual_and_jacobian(
    uv: np.ndarray,
    intr: CameraIntrinsics, pose: CameraPose,
    x_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r [2], H [2, 6]) where r is the projection residual at
    ``x_world`` for pixel ``uv`` and H is ∂(project(x))/∂x padded with zeros
    for velocity columns. Finite-difference Jacobian — small Jacobian, fast.
    """
    K, R, t = intr.K(), pose.R, pose.t
    uv_pred = project_points(x_world.reshape(1, 3), K, R, t)[0]
    r = uv_pred - uv
    eps = 1e-4
    H = np.zeros((2, 6), dtype=np.float64)
    for i in range(3):
        xp = x_world.copy(); xp[i] += eps
        xm = x_world.copy(); xm[i] -= eps
        up = project_points(xp.reshape(1, 3), K, R, t)[0]
        um = project_points(xm.reshape(1, 3), K, R, t)[0]
        H[:, i] = (up - um) / (2 * eps)
    return r, H


def run_kalman(
    detections_per_frame: Sequence[Mapping[str, np.ndarray]],
    cameras: Mapping[str, tuple[CameraIntrinsics, CameraPose]],
    cfg: BallKalmanConfig,
    *,
    return_velocity: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the 3D Kalman filter across ``T`` frames.

    ``detections_per_frame[t]`` is a dict ``{cam_name: uv [2]}``; missing
    cameras for a given frame indicate no ball detection. Cameras not in
    ``cameras`` are silently dropped.

    Returns ``(xyz [T, 3], visible [T])``, or ``(xyz, vel [T, 3], visible)`` when
    ``return_velocity=True``. The velocity (filter state ``x[3:6]``) is what
    orients the canonical football asset along its flight path; ``vel`` is NaN
    on frames where the filter has not emitted a position.
    """
    T = len(detections_per_frame)
    dt = 1.0 / cfg.fps
    F = _transition_matrix(dt)
    g = _gravity_control(dt)
    Q = _process_cov(cfg, dt)

    # Initialize from the first frame that has a triangulable detection.
    x = np.zeros(6, dtype=np.float64)
    P = np.diag([cfg.init_pos_std ** 2] * 3 + [cfg.init_vel_std ** 2] * 3)
    initialized = False

    xyz = np.full((T, 3), np.nan, dtype=np.float64)
    vel = np.full((T, 3), np.nan, dtype=np.float64)
    visible = np.zeros(T, dtype=bool)

    for t in range(T):
        det = detections_per_frame[t]
        cams_with_det = [c for c in det.keys() if c in cameras]

        if not initialized:
            if len(cams_with_det) >= 2:
                a, b = cams_with_det[0], cams_with_det[1]
                pos0 = _triangulate_ball_frame(
                    det[a], det[b], cameras[a], cameras[b]
                )
                if np.all(np.isfinite(pos0)):
                    x[:3] = pos0
                    x[3:] = 0.0
                    initialized = True
            if not initialized:
                # No measurement yet — don't emit.
                continue
        else:
            # Predict.
            x = F @ x + g
            P = F @ P @ F.T + Q

        # Update.
        if len(cams_with_det) >= 2:
            a, b = cams_with_det[0], cams_with_det[1]
            z = _triangulate_ball_frame(det[a], det[b], cameras[a], cameras[b])
            if np.all(np.isfinite(z)):
                H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
                R_meas = np.eye(3) * (cfg.triangulated_pos_std ** 2)
                y = z - H @ x
                S = H @ P @ H.T + R_meas
                K_kf = P @ H.T @ np.linalg.inv(S)
                x = x + K_kf @ y
                P = (np.eye(6) - K_kf @ H) @ P
        elif len(cams_with_det) == 1:
            c = cams_with_det[0]
            intr, pose = cameras[c]
            r, H = _ray_residual_and_jacobian(det[c], intr, pose, x[:3])
            R_meas = np.eye(2) * (cfg.ray_residual_std ** 2)
            y = -r                          # residual form: z - h(x)
            S = H @ P @ H.T + R_meas
            K_kf = P @ H.T @ np.linalg.inv(S)
            x = x + K_kf @ y
            P = (np.eye(6) - K_kf @ H) @ P

        xyz[t] = x[:3]
        vel[t] = x[3:]
        visible[t] = True

    if return_velocity:
        return xyz, vel, visible
    return xyz, visible
