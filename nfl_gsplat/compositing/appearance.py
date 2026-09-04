"""Colour a body from the footage that showed it.

WHY. The avatar twin's geometry is a prior (SMPL-X) because two fixed cameras
give no parallax to learn it from. Its appearance need not be: with the mesh
fixing where every surface point is, colour is a well-posed question from even
one view -- project the vertex into the frame the calibration says it is in,
read the pixel. Team colours, numbers and skin come from the real video
instead of a palette.

WHAT IS SAMPLED, AND WHAT IS NOT. A vertex takes the pixel under it only if it
faces the camera (normal against the viewing ray) and lands inside the frame;
the rest are left NaN for the caller to fill -- from another view, another
frame, or a fallback colour. Nothing here decides which frame is best; that is
the caller's job, and the median over several frames is the honest way to do
it, since a single frame carries motion blur and whoever ran in front.

The samples are bilinear at the projected point. At 140 px per player the
texture is coarse; that is what the footage has.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.compositing.mesh_to_gaussians import vertex_normals

# A vertex must face the camera at least this much (cosine) to be sampled.
# 0.1 keeps grazing surfaces, which are what silhouettes are made of.
MIN_FACING: float = 0.1


def project(vertices, K, R, t):
    """``(uv [V, 2], depth [V])`` of world points in a camera."""
    cam = np.asarray(vertices, np.float64) @ np.asarray(R, np.float64).T + np.asarray(t, np.float64)
    z = cam[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.column_stack([u, v]), z


def sample_bilinear(image, uv):
    """Bilinear samples of ``image [H, W, C]`` at ``uv``; NaN outside."""
    image = np.asarray(image, np.float64)
    h, w = image.shape[:2]
    u, v = uv[:, 0], uv[:, 1]
    out = np.full((len(uv), image.shape[2]), np.nan)
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
    if not ok.any():
        return out
    u, v = u[ok], v[ok]
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bot = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    out[ok] = top * (1 - fy) + bot * fy
    return out


def vertex_colours_from_view(vertices, faces, K, R, t, image_rgb, *,
                             min_facing: float = MIN_FACING):
    """Per-vertex linear RGB in [0, 1] from one frame; NaN where unseen.

    ``image_rgb`` is ``[H, W, 3]`` uint8 or float in [0, 1], RGB order.
    """
    vertices = np.asarray(vertices, np.float64)
    K = np.asarray(K, np.float64)
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).reshape(3)
    normals = vertex_normals(vertices, faces)
    centre = -R.T @ t
    to_cam = centre[None, :] - vertices
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
    facing = (normals * to_cam).sum(1) >= min_facing
    uv, depth = project(vertices, K, R, t)
    uv[~facing | (depth <= 0)] = np.nan
    img = np.asarray(image_rgb)
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    return sample_bilinear(img, uv)


def turf_colour(image_rgb, *, step: int = 16):
    """Median colour of a subsampled frame: the field fills most of an All-22 frame."""
    return np.median(np.asarray(image_rgb, np.float32)[::step, ::step].reshape(-1, 3), axis=0)


def mask_turf(sample, turf, *, dist: float):
    """NaN out vertex samples within ``dist`` (RGB, 0..1) of the turf colour.

    At 140 px a limb is 3-5 px wide and most of its vertices sample pixels
    mixed with grass; a median over frames of those is khaki, not white or
    red (measured: 72-84 % of a body's vertices within 0.12 of turf, and the
    body rendered olive). A grass-coloured sample is not the body."""
    sample = np.array(sample, np.float32, copy=True)
    near = np.linalg.norm(sample - np.asarray(turf, np.float32)[None, :], axis=1) < dist
    sample[near] = np.nan
    return sample


def mesh_edges(faces) -> np.ndarray:
    """Unique undirected edges ``[E, 2]`` of a triangle mesh."""
    f = np.asarray(faces, np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def roughness(colours, faces) -> float:
    """Mean colour difference across mesh edges: speckle scores high, a
    jersey scores low. The measure for the smoothing below."""
    e = mesh_edges(faces)
    c = np.asarray(colours, np.float64)
    return float(np.abs(c[e[:, 0]] - c[e[:, 1]]).mean())


def smooth_colours(colours, faces, *, iters: int = 4, lam: float = 0.5) -> np.ndarray:
    """Laplacian smoothing of per-vertex colours over the mesh: each step
    moves a vertex ``lam`` of the way to the mean of its neighbours.

    WHY. At 140 px a vertex samples one to three valid pixels over the
    play and the per-vertex median is salt-and-pepper: red, white and
    green speckle across one jersey (the v3 fit, 2026-09-04). Four steps
    at 0.5 blend over a few centimetres, below the jersey-to-pants scale,
    and the speckle averages out. Not the fit's job: its smoothness term
    fights the data term on every noisy pixel."""
    c = np.array(colours, np.float64, copy=True)
    e = mesh_edges(faces)
    n = len(c)
    deg = np.bincount(e.ravel(), minlength=n).astype(np.float64)
    for _ in range(int(iters)):
        acc = np.zeros_like(c)
        np.add.at(acc, e[:, 0], c[e[:, 1]])
        np.add.at(acc, e[:, 1], c[e[:, 0]])
        mean = acc / np.maximum(deg, 1.0)[:, None]
        has = deg > 0
        c[has] = (1.0 - lam) * c[has] + lam * mean[has]
    return c


def median_colours(samples, *, fallback):
    """Median over ``[F, V, 3]`` frame samples, ignoring NaN; ``fallback`` where none."""
    samples = np.asarray(samples, np.float64)
    with np.errstate(all="ignore"):
        med = np.nanmedian(samples, axis=0)
    unseen = ~np.isfinite(med).all(1)
    out = med.copy()
    out[unseen] = np.asarray(fallback, np.float64).reshape(1, 3)
    return np.clip(out, 0.0, 1.0), unseen
