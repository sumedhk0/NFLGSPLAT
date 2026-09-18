"""The football as gaussians: a prolate ellipsoid, 28 cm long and 17 cm across, brown with a white
lace line, pointed along its velocity in flight and along the carrier's travel otherwise.

The renderer draws bodies as one flat gaussian per mesh vertex (compositing.mesh_to_gaussians); the
ball uses the same builder on a subdivided octahedron so it shades and occludes like everything else.
The path comes from ``<play-dir>/ball.json`` (scripts/08y_ball_path.py), per frame ``xyz`` on the
field (metres, z up) and ``v`` (metres per frame)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LENGTH_M: float = 0.28
WIDTH_M: float = 0.17
BROWN = np.array([0.42, 0.22, 0.10])
LACE = np.array([0.95, 0.95, 0.92])


def _octasphere(level: int = 2):
    """A unit sphere from a subdivided octahedron: ``(vertices [V, 3], faces [F, 3])``."""
    verts = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(level):
        cache: dict = {}
        new_faces = []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.stack(verts), np.asarray(faces, np.int64)


_SPHERE = _octasphere(2)


def ball_mesh(xyz, v, *, length_m: float = LENGTH_M, width_m: float = WIDTH_M):
    """``(vertices [V, 3], faces, colours [V, 3])`` of the ball at ``xyz`` pointed along ``v``."""
    sv, sf = _SPHERE
    axis = np.asarray(v, float)
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    # a frame with the long axis first
    up = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b = np.cross(axis, up); b /= np.linalg.norm(b)
    c = np.cross(axis, b)
    local = sv * np.array([length_m / 2, width_m / 2, width_m / 2])
    verts = np.asarray(xyz, float) + local[:, 0:1] * axis + local[:, 1:2] * b + local[:, 2:3] * c
    colours = np.tile(BROWN, (len(sv), 1))
    lace = (np.abs(sv[:, 1]) < 0.12) & (sv[:, 2] > 0.6)          # a thin line along the top
    colours[lace] = LACE
    return verts, sf, colours


def load_ball(play_dir) -> dict:
    """``{frame: (xyz, v)}`` from ``<play-dir>/ball.json``, or {} when there is none."""
    f = Path(play_dir) / "ball.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {int(k): (np.asarray(r["xyz"], float), np.asarray(r["v"], float)) for k, r in d["frames"].items()}
