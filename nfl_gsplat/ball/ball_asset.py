"""Canonical football asset + per-frame orientation along the flight path.

A broadcast football is tiny and motion-blurred — a full 3DGS reconstruction
isn't realistic. Instead we author **one** canonical football (a Gaussian
prolate spheroid, brown with a white lace stripe, long axis = +X) once, store it
in the avatar library under ``__football__``, and per frame:

- translate it to the Kalman position ``xyz[t]``,
- rotate its long axis onto the velocity direction ``vel[t]``,
- optionally spin it about that axis (∝ speed · time) for a plausible spiral.

This reuses the library/LBS asset pattern and gives a real ball look + spin
"for free" from the Kalman state, with no per-frame learning.

CPU-only numpy; orientation quaternions reuse the helpers in
:mod:`nfl_gsplat.avatars.lbs_animate`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.avatars.lbs_animate import _quat_to_rotmat, _rotmat_to_quat

LONG_AXIS = np.array([1.0, 0.0, 0.0])    # canonical football points along +X


@dataclass(frozen=True)
class FootballAssetConfig:
    num_gaussians: int = 400
    half_length_m: float = 0.14          # ~0.28 m tip-to-tip
    radius_m: float = 0.085              # ~0.17 m diameter
    gaussian_scale_m: float = 0.012
    lace_color_rgb: tuple[float, float, float] = (0.9, 0.9, 0.9)
    leather_color_rgb: tuple[float, float, float] = (0.40, 0.22, 0.10)


def make_football_asset(cfg: FootballAssetConfig | None = None, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Build the canonical football Gaussian cloud (rest frame, long axis +X).

    Returns a dict with the :data:`nfl_gsplat.avatars.library.ASSET_KEYS`
    fields: ``xyz, rot, scale, opacity, sh`` (sh_degree 0).
    """
    cfg = cfg or FootballAssetConfig()
    n = cfg.num_gaussians
    rng = np.random.default_rng(seed)

    # Sample points on a prolate spheroid surface: x = L·cosθ along the long
    # axis, (y, z) = r·sinθ around it, plus a little inward jitter for volume.
    u = rng.uniform(-1.0, 1.0, n)                    # cosθ along long axis
    phi = rng.uniform(0.0, 2 * np.pi, n)
    ring = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    shell = rng.uniform(0.92, 1.0, n)                # thin shell
    x = cfg.half_length_m * u * shell
    y = cfg.radius_m * ring * np.cos(phi) * shell
    z = cfg.radius_m * ring * np.sin(phi) * shell
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    # Center on the origin so orient_ball's translation lands the centroid
    # exactly on the Kalman position.
    xyz -= xyz.mean(axis=0, keepdims=True)

    rot = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    scale = np.full((n, 3), np.log(cfg.gaussian_scale_m), dtype=np.float32)
    opacity = np.full((n,), 3.0, dtype=np.float32)

    # SH degree 0 (DC only): leather brown, with a white lace stripe along the
    # top seam (small |z|, positive y band near the crown).
    sh = np.zeros((n, 3, 1), dtype=np.float32)
    leather = np.array(cfg.leather_color_rgb, dtype=np.float32)
    lace = np.array(cfg.lace_color_rgb, dtype=np.float32)
    is_lace = (np.abs(z) < 0.18 * cfg.radius_m) & (y > 0.55 * cfg.radius_m)
    sh[:, :, 0] = np.where(is_lace[:, None], lace[None, :], leather[None, :])

    return {"xyz": xyz, "rot": rot, "scale": scale, "opacity": opacity, "sh": sh}


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit ``axis`` by ``angle`` (right-handed)."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _align_x_to(direction: np.ndarray) -> np.ndarray:
    """Rotation mapping the canonical long axis +X onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return np.eye(3)
    d = d / norm
    c = float(np.dot(LONG_AXIS, d))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        # Antiparallel: 180° about any axis ⟂ +X (use +Z).
        return _rodrigues(np.array([0.0, 0.0, 1.0]), np.pi)
    axis = np.cross(LONG_AXIS, d)
    return _rodrigues(axis, np.arccos(c))


def orient_ball(
    asset: dict[str, np.ndarray],
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    t: float = 0.0,
    spin_rate: float = 6.0,
) -> dict[str, np.ndarray]:
    """Pose the canonical football for one frame.

    ``spin_rate`` (rad per m·s⁻¹ per s) gives a speed-proportional spiral about
    the long axis; spin does not change which way the long axis points.
    Returns a new dict (``xyz, rot, scale, opacity, sh``); scale/opacity/sh are
    passed through unchanged.
    """
    position = np.asarray(position, dtype=np.float64).reshape(3)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(velocity))

    R_align = _align_x_to(velocity)
    if speed > 1e-6:
        spin_axis = velocity / speed
        R_total = _rodrigues(spin_axis, spin_rate * speed * t) @ R_align
    else:
        R_total = R_align

    xyz = np.asarray(asset["xyz"], dtype=np.float64)
    xyz_world = (xyz @ R_total.T) + position[None, :]

    R_canon = _quat_to_rotmat(np.asarray(asset["rot"], dtype=np.float64))   # [N,3,3]
    rot_world = _rotmat_to_quat(np.einsum("ij,njk->nik", R_total, R_canon))

    return {
        "xyz": xyz_world.astype(np.float32),
        "rot": rot_world.astype(np.float32),
        "scale": np.asarray(asset["scale"], dtype=np.float32),
        "opacity": np.asarray(asset["opacity"], dtype=np.float32),
        "sh": np.asarray(asset["sh"], dtype=np.float32),
    }


def principal_axis(xyz: np.ndarray) -> np.ndarray:
    """Unit eigenvector of the largest covariance eigenvalue (the long axis)."""
    pts = np.asarray(xyz, dtype=np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    w, v = np.linalg.eigh(cov)
    return v[:, int(np.argmax(w))]
