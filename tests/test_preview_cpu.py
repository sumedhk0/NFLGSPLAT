"""Tests for the CPU splat preview.

This renderer exists so a splat can be INSPECTED on a machine with no CUDA
toolkit. The alternative that was actually reached for -- a generic mesh viewer
-- reports "faces" for a file that has none and shades the point cloud flat
grey, so a correct field is indistinguishable from a collapsed one. Twice during
bring-up that cost real time.
"""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.compositing.merge_ply import GaussianBatch
from nfl_gsplat.compositing.preview_cpu import (intrinsics, look_at,
                                                render_gaussians_cpu)

_SH_C0 = 0.28209479177387814


def _batch(xyz, colours, *, sigma=0.5, alpha=0.99):
    xyz = np.asarray(xyz, np.float32)
    n = len(xyz)
    sh = ((np.asarray(colours, np.float32) - 0.5) / _SH_C0)[:, :, None]
    return GaussianBatch(
        xyz=xyz,
        rot=np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1)),
        scale=np.full((n, 3), np.log(sigma), np.float32),
        opacity=np.full(n, np.log(alpha / (1 - alpha)), np.float32),
        sh=sh.astype(np.float32),
        sh_degree=0,
    )


def _render(batch, eye=(0.0, -10.0, 0.0), target=(0.0, 0.0, 0.0), size=64):
    rot, tvec = look_at(eye, target)
    return render_gaussians_cpu(batch, intrinsics(size, size), rot, tvec,
                                width=size, height=size)


def test_a_gaussian_in_front_is_drawn():
    img = _render(_batch([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]))
    assert img[:, :, 2].max() > 200, "red gaussian did not render"


def test_geometry_behind_the_camera_is_skipped():
    """Projecting a point behind the camera puts it somewhere plausible on
    screen with a negative depth; drawing it paints a ghost."""
    # Camera sits at y=-10 looking toward +Y, so BEHIND it is y < -10.
    behind = _batch([[0.0, -20.0, 0.0]], [[1.0, 0.0, 0.0]])
    img = _render(behind, eye=(0.0, -10.0, 0.0), target=(0.0, 0.0, 0.0))
    assert img[:, :, 2].max() < 100, "a gaussian behind the camera was drawn"


def test_nearer_gaussian_occludes_the_further_one():
    """Painter's algorithm: sorted far-to-near, the near one composites last."""
    batch = _batch([[0.0, 2.0, 0.0], [0.0, -2.0, 0.0]],
                   [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], sigma=1.0)
    img = _render(batch, eye=(0.0, -10.0, 0.0))
    centre = img[img.shape[0] // 2, img.shape[1] // 2]
    assert centre[0] > centre[2], "far (red) gaussian drew over the near (blue) one"


def test_transparent_gaussian_lets_the_background_through():
    opaque = _render(_batch([[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]], alpha=0.99))
    faint = _render(_batch([[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]], alpha=0.05))
    assert faint.max() < opaque.max(), "opacity had no effect on the composite"


def test_empty_view_returns_the_background_not_a_crash():
    img = _render(_batch([[0.0, 500.0, 0.0]], [[1.0, 0.0, 0.0]]))
    assert img.shape == (64, 64, 3)
    assert img.std() < 5, "expected a flat background"


def test_degenerate_up_vector_fails_loud():
    with pytest.raises(ValueError, match="parallel"):
        look_at((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))


# --- depth order must not depend on gaussian SIZE -----------------------------
# The splat loop groups by integer radius for vectorisation. That silently
# reordered compositing by SIZE rather than depth: on the first real scene
# (procedural field at sigma 0.075 m plus SMPL-X bodies at sigma 0.009 m) every
# player was painted over by the turf behind them, because the field's larger
# radius put it in a later group. Field-only and joints-only scenes never
# exposed it, since their gaussians are all one size.

def _one(xyz, sigma, rgb, opacity=0.995):
    from nfl_gsplat.compositing.merge_ply import GaussianBatch
    n = len(xyz)
    a = float(np.clip(opacity, 1e-4, 1 - 1e-4))
    return GaussianBatch(
        xyz=np.asarray(xyz, np.float32),
        rot=np.tile(np.array([1.0, 0, 0, 0], np.float32), (n, 1)),
        scale=np.tile(np.log(np.full(3, sigma, np.float32)), (n, 1)),
        opacity=np.full(n, np.log(a / (1 - a)), np.float32),
        sh=np.tile((np.asarray(rgb, np.float32) - 0.5) / 0.28209479177387814,
                   (n, 1))[:, :, None],
        sh_degree=0)


def test_a_small_near_gaussian_occludes_a_large_far_one():
    from nfl_gsplat.compositing.merge_ply import GaussianBatch
    from nfl_gsplat.compositing.preview_cpu import (intrinsics, look_at,
                                                    render_gaussians_cpu)
    # far + LARGE (like turf), near + SMALL (like a body vertex), same ray
    # Same ray through the image centre, different depths -- otherwise they
    # project to different pixels and the test proves nothing.
    far = _one([[0.0, 6.0, 3.0]], 0.60, (0.0, 1.0, 0.0))
    near = _one([[0.0, 0.0, 3.0]], 0.05, (1.0, 0.0, 0.0))
    both = GaussianBatch(
        xyz=np.vstack([far.xyz, near.xyz]),
        rot=np.vstack([far.rot, near.rot]),
        scale=np.vstack([far.scale, near.scale]),
        opacity=np.concatenate([far.opacity, near.opacity]),
        sh=np.vstack([far.sh, near.sh]), sh_degree=0)
    eye = np.array([0.0, -12.0, 3.0])
    rot, tvec = look_at(eye, np.array([0.0, 12.0, 3.0]))
    img = render_gaussians_cpu(both, intrinsics(160, 120, 50.0), rot, tvec,
                               width=160, height=120)
    centre = img[60, 80]          # BGR
    assert centre[2] > centre[1], (
        f"centre pixel {centre} is not red: the far, LARGER gaussian was "
        "composited over the near, smaller one")
