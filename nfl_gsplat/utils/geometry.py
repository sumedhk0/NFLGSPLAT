"""CPU-only geometry helpers. No torch imports — safe from every conda env.

Conventions:
- World frame: right-handed, metric (meters), origin at field center,
  +X toward home endzone, +Z up.
- Camera pose: R (3,3) rotation world->camera, t (3,) translation world->camera.
  I.e. x_cam = R @ x_world + t.  Camera center c_world = -R.T @ t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Vec3 = np.ndarray       # shape (3,)
Mat3 = np.ndarray       # shape (3, 3)
Mat3x4 = np.ndarray     # shape (3, 4) = [R | t]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> Mat3:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraPose:
    R: Mat3                 # (3,3), world -> camera
    t: Vec3                 # (3,), world -> camera

    def center_world(self) -> Vec3:
        return -self.R.T @ self.t

    def extrinsic_3x4(self) -> Mat3x4:
        return np.concatenate([self.R, self.t.reshape(3, 1)], axis=1)


def project_points(points_w: np.ndarray, K: Mat3, R: Mat3, t: Vec3) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixel coords. Points behind the camera
    (z_cam <= 0) return (NaN, NaN)."""
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    x_cam = pts @ R.T + t.reshape(1, 3)
    z = x_cam[:, 2]
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    front = z > 1e-6
    xy = x_cam[front, :2] / z[front, None]
    hom = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)
    uv_front = hom @ K.T
    uv[front] = uv_front[:, :2]
    return uv


def backproject_ray(uv: np.ndarray, K: Mat3, R: Mat3, t: Vec3) -> tuple[Vec3, Vec3]:
    """Return (origin_world, direction_world_unit) for a pixel ray.

    origin = camera center; direction points *into* the scene.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    Kinv = np.linalg.inv(K)
    d_cam = Kinv @ np.array([uv[0], uv[1], 1.0])
    d_world = R.T @ d_cam
    d_world /= np.linalg.norm(d_world) + 1e-12
    origin = -R.T @ t
    return origin, d_world


def project_to_plane_z0(
    uv: np.ndarray,
    K: Mat3,
    R: Mat3,
    t: Vec3,
) -> np.ndarray:
    """Intersect pixel ray with the Z=0 world plane.

    Returns (x, y) in world meters, or (NaN, NaN) if the ray is ~parallel
    to the plane or intersects behind the camera.
    """
    origin, direction = backproject_ray(uv, K, R, t)
    if abs(direction[2]) < 1e-9:
        return np.array([np.nan, np.nan])
    s = -origin[2] / direction[2]
    if s <= 0:
        return np.array([np.nan, np.nan])
    hit = origin + s * direction
    return hit[:2]


def reprojection_rms(
    world_pts: np.ndarray,
    uv_gt: np.ndarray,
    K: Mat3,
    R: Mat3,
    t: Vec3,
) -> float:
    """Root-mean-square reprojection error in pixels."""
    uv_pred = project_points(world_pts, K, R, t)
    mask = np.isfinite(uv_pred).all(axis=1)
    err = uv_pred[mask] - uv_gt[mask]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1)))) if mask.any() else float("inf")


def foot_point_from_bbox(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Return (u, v) = bottom-center of an axis-aligned bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([0.5 * (x1 + x2), y2], dtype=np.float64)


def point_in_field_rect(xy: np.ndarray, length_m: float, width_m: float, buffer_m: float = 0.0) -> bool:
    """True iff (x, y) world meters lies inside the playing field + buffer."""
    hx = 0.5 * length_m + buffer_m
    hy = 0.5 * width_m + buffer_m
    return bool(abs(xy[0]) <= hx and abs(xy[1]) <= hy)


def triangulate_two_views(
    uv0: np.ndarray, uv1: np.ndarray,
    P0: Mat3x4, P1: Mat3x4,
) -> np.ndarray:
    """Linear two-view triangulation. Takes ``[u, v]`` pixel pairs and 3x4
    projection matrices ``P = K @ [R|t]``; returns Nx3 world points.
    """
    uv0 = np.asarray(uv0, dtype=np.float64).reshape(-1, 2)
    uv1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    N = uv0.shape[0]
    out = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        u0, v0 = uv0[i]
        u1, v1 = uv1[i]
        A = np.stack([
            u0 * P0[2] - P0[0],
            v0 * P0[2] - P0[1],
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out[i] = X[:3] / X[3]
    return out
