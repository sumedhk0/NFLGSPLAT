"""Stadium lighting on the avatars: kit colours shaded through the posed mesh's own normals.

WHY. The hi-fi render paints each body in flat kit colours (render.uniform); with no light, a player reads as a cutout
and the pile as one red mass. A night game under stadium lights has a strong key light from above: shading each splat
by how much it faces that light gives the bodies their volume back and separates overlapping players.

WHAT. ``lighting`` is ``ambient + diffuse * max(0, n . light) + bounce * max(0, -n_z)`` per outward normal (the bounce
is the turf lighting the undersides a little). ``shade`` applies it to per-vertex colours through the mesh's
area-weighted normals; ``lit_batch`` applies it to a built GaussianBatch through each splat's own normal -- the z axis
of its rotation, which mesh_to_gaussians and uniform.decal_gaussians both lay along the outward surface normal -- so a
body and the numbers on it are lit alike.

The key light sits high on the near-sideline (-y), behind-the-offense (+x) side: the sideline broadcast camera stands
at y = -102 m and the follow and skycam views look down the field from +x, so the faces every shipped view sees are lit
and the far sides fall off into shade.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from nfl_gsplat.compositing.mesh_to_gaussians import _SH_C0, vertex_normals

KEY_LIGHT = (0.35, -0.35, 0.87)      # toward the light: up, toward the near sideline and behind the offense (world z up)
AMBIENT = 0.58
DIFFUSE = 0.52
BOUNCE = 0.06


def lighting(normals, *, light=KEY_LIGHT, ambient: float = AMBIENT, diffuse: float = DIFFUSE,
             bounce: float = BOUNCE) -> np.ndarray:
    """``[N]`` brightness factor for unit outward ``normals`` ``[N, 3]`` under one directional ``light``."""
    n = np.asarray(normals, float)
    L = np.asarray(light, float)
    L = L / np.linalg.norm(L)
    return ambient + diffuse * np.clip(n @ L, 0.0, 1.0) + bounce * np.clip(-n[:, 2], 0.0, 1.0)


def shade(verts, faces, colour, **kw) -> np.ndarray:
    """``colour`` ``[V, 3]`` (or one RGB for every vertex) lit through the mesh's own normals, clipped to [0, 1]."""
    v = np.asarray(verts, float)
    c = np.asarray(colour, float)
    if c.ndim == 1:
        c = np.broadcast_to(c, v.shape)
    return np.clip(c * lighting(vertex_normals(v, faces), **kw)[:, None], 0.0, 1.0)


def splat_normals(rot) -> np.ndarray:
    """``[N, 3]`` z axes of the splats' rotations (quaternions w, x, y, z): the normal each disc was laid along."""
    q = np.asarray(rot, float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


def lit_batch(batch, **kw):
    """A degree-0 GaussianBatch with every splat's colour lit through its own normal (anything else untouched)."""
    if batch is None or batch.sh_degree != 0 or not len(batch.xyz):
        return batch
    rgb = batch.sh[:, :, 0].astype(float) * _SH_C0 + 0.5
    rgb = np.clip(rgb * lighting(splat_normals(batch.rot), **kw)[:, None], 0.0, 1.0)
    return dataclasses.replace(batch, sh=((rgb - 0.5) / _SH_C0)[:, :, None].astype(np.float32))
