"""Tests for converting a body mesh into scene gaussians."""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.compositing.mesh_to_gaussians import (merge, mesh_to_gaussians,
                                                      vertex_normals)


def _quad():
    """Two triangles in the z=0 plane, so every normal must be +/-Z."""
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return verts, faces


def test_normals_are_perpendicular_to_a_flat_surface():
    verts, faces = _quad()
    n = vertex_normals(verts, faces)
    assert np.allclose(np.abs(n[:, 2]), 1.0)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_gaussians_are_flattened_along_the_normal():
    """A body surface is a SHELL. Isotropic gaussians inflate every limb by the
    radius and turn a 1.9 m player into a snowman."""
    verts, faces = _quad()
    b = mesh_to_gaussians(verts, faces)
    sigma = np.exp(b.scale)
    assert (sigma[:, 2] < sigma[:, 0] / 2).all(), "not flattened"
    assert np.allclose(sigma[:, 0], sigma[:, 1]), "in-plane axes should match"


def test_disc_is_rotated_into_the_surface():
    """The thin axis must point ALONG the normal, else the flattening is
    applied in the wrong direction and the shell reads as slabs."""
    verts, faces = _quad()
    b = mesh_to_gaussians(verts, faces)
    w, x, y, z = b.rot[0]
    # rotate +Z by the quaternion; for a z=0 surface it must stay on +/-Z
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    assert abs(abs((rot @ np.array([0.0, 0, 1]))[2]) - 1.0) < 1e-6


def test_extent_scales_with_the_mesh_not_a_fixed_radius():
    """A 2 m player and a 1.7 m player need different disc sizes for the shell
    to stay closed without over-inflating."""
    verts, faces = _quad()
    small = np.exp(mesh_to_gaussians(verts, faces).scale)[0, 0]
    big = np.exp(mesh_to_gaussians(verts * 3.0, faces).scale)[0, 0]
    assert big == pytest.approx(3 * small, rel=1e-6)


def test_colour_survives_the_sh_encoding():
    verts, faces = _quad()
    b = mesh_to_gaussians(verts, faces, colour=(0.2, 0.4, 0.9))
    rgb = 0.5 + 0.28209479177387814 * b.sh[:, :, 0]
    assert np.allclose(rgb[0], [0.2, 0.4, 0.9], atol=1e-5)


def test_merge_concatenates_and_preserves_counts():
    verts, faces = _quad()
    a = mesh_to_gaussians(verts, faces)
    b = mesh_to_gaussians(verts + 5.0, faces)
    m = merge([a, b])
    assert m.num_gaussians == a.num_gaussians + b.num_gaussians
    m.assert_no_nans()


def test_merge_refuses_mismatched_sh_degree():
    """Different SH widths cannot stack; failing here beats a shape error deep
    in the rasteriser."""
    verts, faces = _quad()
    a = mesh_to_gaussians(verts, faces)
    b = mesh_to_gaussians(verts, faces)
    b = type(b)(xyz=b.xyz, rot=b.rot, scale=b.scale, opacity=b.opacity,
                sh=np.repeat(b.sh, 4, axis=2), sh_degree=1)
    with pytest.raises(ValueError, match="SH degrees"):
        merge([a, b])


def test_bad_vertex_shape_is_rejected():
    _v, faces = _quad()
    with pytest.raises(ValueError, match=r"\[V, 3\]"):
        mesh_to_gaussians(np.zeros((4, 2)), faces)


def test_splat_extent_follows_the_local_vertex_spacing():
    # Two fans: a coarse one (spacing 0.10) and a fine one (spacing 0.01),
    # each a hub with a ring of 8 vertices.
    def fan(centre, r, offset):
        hub = np.array(centre)
        ring = hub + np.stack([r * np.cos(np.linspace(0, 2 * np.pi, 9)[:-1]),
                               r * np.sin(np.linspace(0, 2 * np.pi, 9)[:-1]), np.zeros(8)], axis=1)
        verts = np.vstack([hub, ring])
        faces = np.array([[0, 1 + i, 1 + (i + 1) % 8] for i in range(8)]) + offset
        return verts, faces
    v1, f1 = fan((0.0, 0.0, 0.0), 0.10, 0)
    v2, f2 = fan((5.0, 0.0, 0.0), 0.01, 9)
    batch = mesh_to_gaussians(np.vstack([v1, v2]), np.vstack([f1, f2]))
    sig = np.exp(batch.scale[:, 0])
    assert sig[0] > 5 * sig[9]                      # coarse hub vs fine hub
    assert np.all(sig[:9] > 0.05) and np.all(sig[9:] < 0.02)

