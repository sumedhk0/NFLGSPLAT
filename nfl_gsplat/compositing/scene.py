"""Per-frame scene composition: field + posed avatars + oriented football.

Pure numpy helpers (no torch / gsplat) so the compositing contract is unit- and
smoke-testable on CPU. The GPU render script (``scripts/05_render_novel_view``)
calls these to build the single :class:`~nfl_gsplat.compositing.merge_ply.GaussianBatch`
it hands to the rasterizer each frame.

An avatar (player or the generic referee) is a canonical Gaussian cloud +
``lbs_weights`` that we drive with that entity's per-frame joint transforms. The
football is the canonical asset oriented along the Kalman velocity. All sources
land in one merged batch.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.avatars.lbs_animate import animate_gaussians
from nfl_gsplat.ball.ball_asset import orient_ball
from nfl_gsplat.compositing.merge_ply import (
    GaussianBatch,
    batch_from_arrays,
    merge_batches,
)


def posed_avatar_batch(
    avatar: dict[str, np.ndarray],
    joint_tfms: np.ndarray,
) -> GaussianBatch:
    """LBS-pose a canonical avatar dict for one frame.

    ``avatar`` carries the canonical Gaussian schema (``canonical_xyz/rot/scale/
    opacity/sh``, ``lbs_weights``); ``joint_tfms`` is ``[J, 4, 4]`` canonical→world.
    Works for both per-player and the generic referee avatar.
    """
    xyz_w, rot_w = animate_gaussians(
        avatar["canonical_xyz"], avatar["canonical_rot"],
        avatar["lbs_weights"], joint_tfms,
    )
    sh = np.asarray(avatar["canonical_sh"], dtype=np.float32)
    return GaussianBatch(
        xyz=xyz_w.astype(np.float32),
        rot=rot_w.astype(np.float32),
        scale=np.asarray(avatar["canonical_scale"], dtype=np.float32),
        opacity=np.asarray(avatar["canonical_opacity"], dtype=np.float32),
        sh=sh,
        sh_degree=int(round(np.sqrt(sh.shape[-1])) - 1),
    )


def football_batch(
    asset: dict[str, np.ndarray],
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    t: float = 0.0,
    spin_rate: float = 6.0,
) -> GaussianBatch:
    """Orient the canonical football asset for one frame as a GaussianBatch."""
    posed = orient_ball(asset, position, velocity, t=t, spin_rate=spin_rate)
    return batch_from_arrays(
        xyz=posed["xyz"], rot=posed["rot"], scale=posed["scale"],
        opacity=posed["opacity"], sh=posed["sh"],
    )


def compose_frame(
    field: GaussianBatch,
    posed_avatars: list[GaussianBatch],
    ball: GaussianBatch | None = None,
) -> GaussianBatch:
    """Merge field + posed avatars (+ optional ball) into one batch.

    Composite-count contract:
    ``field_N + Σ avatar_N (players + referees) + (ball_N if present)``.
    """
    batches = [field, *posed_avatars]
    if ball is not None:
        batches.append(ball)
    return merge_batches(batches)
