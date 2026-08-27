"""Turn a posed SMPL-X mesh into gaussians that merge with the rest of the scene.

The scene is gaussians end to end -- the field is drawn as gaussians, and
``render_gsplat`` rasterises one merged batch -- so a body arriving as a
triangle mesh would need a second renderer and its own depth handling. Converting
it here keeps a single primitive and lets the rasteriser sort bodies against the
field for free.

A body surface is a SHELL, not a volume, so each gaussian is a flat disc lying in
the surface with its thin axis along the normal, exactly as the field's texels
are. Giving them isotropic extent instead inflates every limb by the gaussian
radius and makes a 1.9 m player read as a snowman.

Appearance here is a flat team colour. Real per-player appearance is the avatar
stage's job (LHM, feed-forward from a crop); this exists so bodies can be seen,
sized and placed correctly before that runs.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.compositing.merge_ply import GaussianBatch

_SH_C0 = 0.28209479177387814


def vertex_normals(vertices, faces):
    """Area-weighted vertex normals.

    Area weighting matters on SMPL-X: triangles around the hands and face are
    orders of magnitude smaller than those on the torso, and an unweighted mean
    lets that dense detail dominate the normal of every vertex it touches.
    """
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths < 1e-12, 1.0, lengths)


def _quat_from_z_to(normals):
    """Quaternions (w, x, y, z) rotating +Z onto each normal."""
    normals = np.asarray(normals, np.float64)
    z = np.array([0.0, 0.0, 1.0])
    dot = np.clip(normals @ z, -1.0, 1.0)
    axis = np.cross(np.broadcast_to(z, normals.shape), normals)
    axis_len = np.linalg.norm(axis, axis=1, keepdims=True)

    quats = np.zeros((len(normals), 4))
    quats[:, 0] = 1.0                                  # identity default
    ok = axis_len[:, 0] > 1e-9
    angle = np.arccos(dot[ok])
    unit = axis[ok] / axis_len[ok]
    quats[ok, 0] = np.cos(angle / 2)
    quats[ok, 1:] = unit * np.sin(angle / 2)[:, None]

    # Antiparallel: any perpendicular axis is a valid 180 degree rotation.
    flipped = (~ok) & (dot < 0)
    quats[flipped] = np.array([0.0, 1.0, 0.0, 0.0])
    return quats


def mesh_to_gaussians(vertices, faces, *, colour=(0.7, 0.7, 0.7),
                      opacity: float = 0.995,
                      thickness_ratio: float = 0.25) -> GaussianBatch:
    """One flat, surface-aligned gaussian per vertex.

    ``colour`` is linear RGB in [0, 1]. In-plane extent is derived from the mesh
    itself -- the mean distance between connected vertices -- so the shell stays
    closed whatever the body's size, rather than needing a hand-tuned radius per
    player.
    """
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be [V, 3], got {vertices.shape}")

    normals = vertex_normals(vertices, faces)

    tri = vertices[faces]
    edges = np.concatenate([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ])
    sigma_xy = float(np.mean(edges)) * 0.75      # overlap slightly, no gaps
    sigma_n = max(thickness_ratio * sigma_xy, 1e-4)

    n = len(vertices)
    scale = np.tile(np.log([sigma_xy, sigma_xy, sigma_n]).astype(np.float32), (n, 1))
    alpha = float(np.clip(opacity, 1e-4, 1 - 1e-4))
    rgb = np.tile(np.asarray(colour, np.float32), (n, 1))

    return GaussianBatch(
        xyz=vertices.astype(np.float32),
        rot=_quat_from_z_to(normals).astype(np.float32),
        scale=scale,
        opacity=np.full(n, np.log(alpha / (1 - alpha)), np.float32),
        sh=((rgb - 0.5) / _SH_C0)[:, :, None].astype(np.float32),
        sh_degree=0,
    )


def merge(batches) -> GaussianBatch:
    """Concatenate batches into one scene. All must share ``sh_degree``."""
    batches = [b for b in batches if b is not None and b.num_gaussians]
    if not batches:
        raise ValueError("nothing to merge")
    degrees = {b.sh_degree for b in batches}
    if len(degrees) != 1:
        raise ValueError(
            f"cannot merge batches with different SH degrees {sorted(degrees)}; "
            "their sh arrays have different widths and would not stack.")
    return GaussianBatch(
        xyz=np.vstack([b.xyz for b in batches]),
        rot=np.vstack([b.rot for b in batches]),
        scale=np.vstack([b.scale for b in batches]),
        opacity=np.concatenate([b.opacity for b in batches]),
        sh=np.concatenate([b.sh for b in batches]),
        sh_degree=batches[0].sh_degree,
    )
