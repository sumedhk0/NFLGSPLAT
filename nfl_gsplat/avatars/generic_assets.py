"""Procedural generic avatars that aren't reconstructed per person.

Referees are not roster players, so we don't build a personal avatar for each.
Instead one **generic striped-shirt avatar** is authored once and posed by each
official's solved motion (like any cached avatar — same canonical schema +
``lbs_weights`` over the 22 body joints). The black/white vertical stripes are
encoded in the SH DC term by azimuth around the body axis.

Returns the canonical avatar dict consumed by
:meth:`nfl_gsplat.avatars.library.AvatarLibrary.put_referee_avatar`.
"""
from __future__ import annotations

import numpy as np


def make_referee_avatar(
    *,
    num_gaussians: int = 3000,
    num_joints: int = 22,
    sh_degree: int = 0,
    num_stripes: int = 8,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build a striped-shirt capsule avatar with a valid 22-joint LBS rig."""
    rng = np.random.default_rng(seed)
    xyz = np.column_stack([
        rng.normal(0.0, 0.18, num_gaussians),       # x lateral
        rng.normal(0.0, 0.12, num_gaussians),       # y front-back
        rng.uniform(-0.90, 0.70, num_gaussians),    # z pelvis ± ~1 m
    ]).astype(np.float32)

    rot = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (num_gaussians, 1))
    scale = np.full((num_gaussians, 3), np.log(0.04), dtype=np.float32)
    opacity = np.full((num_gaussians,), 2.0, dtype=np.float32)

    # Vertical black/white stripes by azimuth around the body's z axis.
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])                      # [-π, π]
    band = np.floor((azimuth + np.pi) / (2 * np.pi) * num_stripes).astype(int)
    is_white = (band % 2) == 0
    K_sh = (sh_degree + 1) ** 2
    sh = np.zeros((num_gaussians, 3, K_sh), dtype=np.float32)
    sh[:, :, 0] = np.where(is_white[:, None], 0.9, 0.1)

    # One-hot LBS to the nearest body joint by height (same scheme as the mock
    # avatar) so animate_gaussians poses it correctly.
    joint_z = np.linspace(-0.9, 0.9, num_joints)
    nearest = np.argmin(np.abs(xyz[:, 2, None] - joint_z[None, :]), axis=1)
    lbs = np.zeros((num_gaussians, num_joints), dtype=np.float32)
    lbs[np.arange(num_gaussians), nearest] = 1.0

    return {
        "canonical_xyz": xyz,
        "canonical_rot": rot,
        "canonical_scale": scale,
        "canonical_opacity": opacity,
        "canonical_sh": sh,
        "lbs_weights": lbs,
        "tier": np.array(["referee"]),
    }
