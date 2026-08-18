"""Homography <-> (K, R, t) for the planar (z=0) field.

A camera viewing the world z=0 plane satisfies, for a world point (X, Y, 0):
    s [u, v, 1]^T = K [r1 | r2 | t] [X, Y, 1]^T
so the field->image homography is H = K [r1 | r2 | t] (r1, r2 = first two
columns of R). Given H and the fixed-principal-point/unit-aspect intrinsic
model (same as solve_pnp), we recover the focal from the orthonormality of the
plane axes and rebuild a proper (K, R, t). Pure numpy; CPU-only.
"""
from __future__ import annotations

import numpy as np


def krt_to_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Field(z=0)->image homography H = K [r1 | r2 | t]."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    M = np.column_stack([R[:, 0], R[:, 1], t])
    H = np.asarray(K, dtype=np.float64) @ M
    return H / H[2, 2]


def _solve_focal(H: np.ndarray, cx: float, cy: float) -> float:
    H = np.asarray(H, dtype=np.float64)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    G = T @ H
    h1, h2 = G[:, 0], G[:, 1]
    denom_o = h1[2] * h2[2]
    num_o = -(h1[0] * h2[0] + h1[1] * h2[1])
    denom_n = (h1[2] ** 2 - h2[2] ** 2)
    num_n = (h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)
    # Both formulas divide by quantities that vanish at particular poses:
    # denom_o -> 0 when a plane axis is parallel to the image plane (a camera on
    # the field centerline, the nominal endzone rig pose). An ABSOLUTE gate is
    # meaningless because H carries an arbitrary scale, so a 0/0-shaped blowup
    # slipped through and poisoned the mean (measured: 1.14e6 averaged in, giving
    # f=1990 instead of 2600). Gate on RELATIVE conditioning, then weight what
    # survives by how well-conditioned it is.
    scale2 = max(abs(h1[2]), abs(h2[2]), 1e-30) ** 2
    cands, weights = [], []
    for num, den in ((num_o, denom_o), (num_n, denom_n)):
        if abs(den) / scale2 <= 1e-6:        # relatively degenerate: discard
            continue
        val = num / den
        if val > 0:
            cands.append(val)
            weights.append(abs(den))
    if not cands:
        raise ValueError("cannot recover focal from homography (degenerate view)")
    return float(np.sqrt(np.average(cands, weights=weights)))


def homography_to_krt(
    H: np.ndarray, *, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a field(z=0)->image homography into (K, R, t)."""
    H = np.asarray(H, dtype=np.float64)
    cx, cy = width / 2.0, height / 2.0
    f = _solve_focal(H, cx, cy)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], dtype=np.float64)
    B = np.linalg.inv(K) @ H
    scale = 1.0 / np.linalg.norm(B[:, 0])
    # Sign: the camera must see the field in front of it (positive depth).
    if (B[:, 2] * scale)[2] < 0:
        scale = -scale
    r1 = B[:, 0] * scale
    r2 = B[:, 1] * scale
    t = B[:, 2] * scale
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U @ Vt)
    R = U @ D @ Vt
    return K, R, t
