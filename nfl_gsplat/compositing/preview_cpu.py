"""CPU preview renderer for gaussian splats.

The real renderer is :mod:`nfl_gsplat.compositing.render_gsplat`, which drives
``gsplat.rasterization`` and needs a CUDA toolkit to JIT its kernels. That makes
it unavailable on a workstation without one, which is exactly where a splat most
often needs a quick look -- and the alternative, opening the PLY in a generic
mesh viewer, is actively misleading: those tools invent faces over the point
cloud and shade them flat grey, so a perfectly good field renders as a slab.

This is a small, dependency-light painter's-algorithm rasteriser: project each
gaussian's centre, sort back to front, and composite. It is a VERIFICATION tool,
not a substitute for the real thing -- it ignores anisotropic screen-space
covariance and evaluates spherical harmonics at degree 0 only. For opaque,
roughly isotropic content (the field, and avatars at a glance) it is faithful
enough to catch the errors worth catching: wrong pose, wrong scale, wrong
colour, wrong orientation, geometry in the wrong place.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.compositing.merge_ply import GaussianBatch

_SH_C0 = 0.28209479177387814


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def render_gaussians_cpu(batch: GaussianBatch, k_mat, rot, tvec, *,
                         width: int, height: int,
                         background=(0.10, 0.12, 0.14),
                         max_radius_px: int = 6) -> np.ndarray:
    """Render a batch from one camera. Returns ``uint8`` BGR ``[height, width, 3]``.

    ``k_mat``/``rot``/``tvec`` are the usual pinhole triple: world -> camera is
    ``rot @ x + tvec``.
    """
    xyz = np.asarray(batch.xyz, np.float64)
    cam = (np.asarray(rot, np.float64) @ xyz.T + np.asarray(tvec, np.float64)[:, None]).T
    depth = cam[:, 2]
    in_front = depth > 1e-6
    if not in_front.any():
        img = np.empty((height, width, 3), np.float64)
        img[:] = np.asarray(background, np.float64)[::-1]
        return (255 * np.clip(img, 0, 1)).astype(np.uint8)

    proj = (np.asarray(k_mat, np.float64) @ cam[in_front].T).T
    uv = proj[:, :2] / proj[:, 2:3]
    depth_f = depth[in_front]

    sigma = np.exp(np.asarray(batch.scale, np.float64)[in_front])
    # In-plane extent only; the flattened axis contributes no screen footprint
    # for a surface seen from anywhere but edge-on.
    sigma_xy = np.sort(sigma, axis=1)[:, 1]
    radius = np.clip(float(k_mat[0][0]) * sigma_xy / depth_f, 0.5, max_radius_px)

    colour = np.clip(0.5 + _SH_C0 * np.asarray(batch.sh, np.float64)[in_front, :, 0],
                     0.0, 1.0)
    alpha = _sigmoid(np.asarray(batch.opacity, np.float64)[in_front])

    on_screen = ((uv[:, 0] > -max_radius_px) & (uv[:, 0] < width + max_radius_px)
                 & (uv[:, 1] > -max_radius_px) & (uv[:, 1] < height + max_radius_px))
    uv, depth_f, radius = uv[on_screen], depth_f[on_screen], radius[on_screen]
    colour, alpha = colour[on_screen], alpha[on_screen]

    # Painter's algorithm: far to near, so nearer gaussians composite last.
    order = np.argsort(-depth_f)
    uv, radius = uv[order], radius[order]
    colour, alpha = colour[order], alpha[order]

    img = np.empty((height, width, 3), np.float64)
    img[:] = np.asarray(background, np.float64)[::-1]        # BGR

    cols = np.round(uv[:, 0]).astype(np.int64)
    rows = np.round(uv[:, 1]).astype(np.int64)
    bgr = colour[:, ::-1]

    # Splat as square footprints. Grouping by integer radius keeps this
    # vectorised -- a per-gaussian Python loop over ~1e6 points is minutes.
    for rad in np.unique(np.round(radius).astype(np.int64)):
        sel = np.round(radius).astype(np.int64) == rad
        if not sel.any():
            continue
        c_sel, r_sel = cols[sel], rows[sel]
        a_sel, col_sel = alpha[sel][:, None], bgr[sel]
        for d_r in range(-rad, rad + 1):
            for d_c in range(-rad, rad + 1):
                rr, cc = r_sel + d_r, c_sel + d_c
                ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                if not ok.any():
                    continue
                flat = rr[ok] * width + cc[ok]
                flat_img = img.reshape(-1, 3)
                # Later writes win, which is the painter's algorithm; the alpha
                # blend against what is already there keeps semi-transparent
                # gaussians from reading as opaque.
                flat_img[flat] = (a_sel[ok] * col_sel[ok]
                                  + (1.0 - a_sel[ok]) * flat_img[flat])

    return (255 * np.clip(img, 0, 1)).astype(np.uint8)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """``(rot, tvec)`` for a camera at ``eye`` looking at ``target``.

    Camera convention matches the rest of the project: +Z forward, +Y down.
    """
    eye = np.asarray(eye, np.float64)
    forward = np.asarray(target, np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, np.float64))
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        raise ValueError("camera up vector is parallel to the view direction")
    right /= norm
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])
    return rot, -rot @ eye


def intrinsics(width: int, height: int, fov_deg: float = 50.0):
    """Simple pinhole K for previewing."""
    focal = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]])
