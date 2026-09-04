"""A small differentiable Gaussian splatter in pure torch, for fitting appearance.

WHY. The photometric fit (docs/DESIGN_photometric_appearance.md) needs a
renderer that gives gradients to each Gaussian's colour, extent and opacity.
``gsplat``'s rasteriser needs a CUDA toolkit to JIT and there is none on this
machine; the CPU preview splatter is not differentiable. Bodies are small on
the footage (120-160 px), so a dense per-crop splatter is enough: no tiles, no
custom kernels, one body at a time.

WHAT. EWA splatting as in 3D Gaussian Splatting: each Gaussian's 3-D covariance
``R S S^T R^T`` is pushed through the camera's rotation and the projection's
Jacobian to a 2-D covariance, then composited front to back over the pixels of a
crop, ``C = sum_i T_i a_i c_i``, ``T_i = prod_{j<i} (1 - a_j)``. Gaussians are
processed in depth order in chunks so memory is ``pixels x chunk``.

WHAT IS OPTIMISED. Colour (linear RGB), a per-Gaussian log scale multiplier
and the opacity logit. Positions and orientations stay bound to the body: the
pose stage owns geometry (design decision, not a limitation of the maths).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nfl_gsplat.compositing.merge_ply import GaussianBatch, batch_from_arrays

SH_C0 = 0.28209479177387814
# 3DGS's low-pass: a third of a pixel added to every 2-D covariance so a
# Gaussian thinner than a pixel still touches one.
PIXEL_EPS = 0.3
MIN_DEPTH = 0.1
ALPHA_MAX = 0.99
ALPHA_CUT = 1.0 / 255.0
# A Gaussian is evaluated within this many standard deviations of its
# centre (3DGS uses 3), never beyond MAX_WINDOW_PX pixels.
WINDOW_SIGMAS = 3.0
MAX_WINDOW_PX = 40


@dataclass
class SceneParams:
    xyz: torch.Tensor              # [N, 3], fixed
    rot: torch.Tensor              # [N, 4] wxyz, fixed
    log_scale: torch.Tensor        # [N, 3], fixed base
    colour: torch.Tensor           # [N, 3] linear RGB, learnable
    log_scale_mult: torch.Tensor   # [N], learnable
    opacity_logit: torch.Tensor    # [N], learnable
    sh_k: int = 1

    @classmethod
    def from_batch(cls, batch: GaussianBatch, *, requires_grad: bool = False,
                   device: str | torch.device = "cpu") -> "SceneParams":
        dev = torch.device(device)
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)  # noqa: E731
        colour = 0.5 + SH_C0 * f(batch.sh[:, :, 0])
        n = batch.num_gaussians
        scene = cls(xyz=f(batch.xyz), rot=f(batch.rot), log_scale=f(batch.scale),
                    colour=colour.clone(), log_scale_mult=torch.zeros(n, device=dev),
                    opacity_logit=f(batch.opacity).clone(), sh_k=int(batch.sh.shape[-1]))
        if requires_grad:
            for name in ("colour", "log_scale_mult", "opacity_logit"):
                getattr(scene, name).requires_grad_(True)
        return scene

    def to_batch(self) -> GaussianBatch:
        n = self.xyz.shape[0]
        sh = np.zeros((n, 3, self.sh_k), np.float32)
        sh[:, :, 0] = ((self.colour.detach().cpu().numpy() - 0.5) / SH_C0)
        scale = (self.log_scale + self.log_scale_mult[:, None]).detach().cpu().numpy()
        return batch_from_arrays(self.xyz.cpu().numpy().astype(np.float32),
                                 self.rot.cpu().numpy().astype(np.float32),
                                 scale.astype(np.float32),
                                 self.opacity_logit.detach().cpu().numpy().astype(np.float32),
                                 sh)

    def parameters(self):
        return [self.colour, self.log_scale_mult, self.opacity_logit]


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """``[N, 4]`` wxyz (any norm) -> ``[N, 3, 3]``."""
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def covariance3d(scene: SceneParams) -> torch.Tensor:
    """``[N, 3, 3]`` world-frame covariance ``R S S^T R^T``."""
    Rm = quat_to_rotmat(scene.rot)
    s = torch.exp(scene.log_scale + scene.log_scale_mult[:, None])
    M = Rm * s[:, None, :]
    return M @ M.transpose(1, 2)


def project(scene: SceneParams, K, R, t):
    """``(means2d [N, 2], cov2d [N, 2, 2], depth [N])`` through a pinhole camera."""
    dev = scene.xyz.device
    # K may be a tensor with gradient (the fit's principal-point shift).
    K = (K.to(dev, torch.float32) if torch.is_tensor(K)
         else torch.as_tensor(np.asarray(K, np.float32), device=dev))
    Rc = torch.as_tensor(np.asarray(R, np.float32), device=dev)
    tc = torch.as_tensor(np.asarray(t, np.float32), device=dev).reshape(3)
    Xc = scene.xyz @ Rc.T + tc                                  # [N, 3]
    z = Xc[:, 2].clamp_min(MIN_DEPTH)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    zero = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([fx / z, zero, -fx * Xc[:, 0] / (z * z)], 1),
        torch.stack([zero, fy / z, -fy * Xc[:, 1] / (z * z)], 1),
    ], 1)                                                       # [N, 2, 3]
    Sigma_c = Rc @ covariance3d(scene) @ Rc.T                   # [N, 3, 3]
    cov2d = J @ Sigma_c @ J.transpose(1, 2)
    cov2d = cov2d + PIXEL_EPS * torch.eye(2, device=dev)
    return torch.stack([u, v], 1), cov2d, Xc[:, 2]


def render(scene: SceneParams, K, R, t, *, crop, background=(0.10, 0.12, 0.14),
           chunk: int = 2048, return_transmittance: bool = False):
    """``[h, w, 3]`` linear RGB of the crop ``(x0, y0, w, h)``, differentiable
    in colour, scale multiplier and opacity. Pixel ``(i, j)`` of the crop is
    image coordinate ``(x0 + j, y0 + i)``. ``background`` is an RGB triple or
    an ``[h, w, 3]`` image the body is composited over (the fit passes the
    frame crop itself, so uncovered pixels cost nothing).

    Sparse throughout: a Gaussian contributes only to the pixels within
    WINDOW_SIGMAS of its centre, the (pixel, Gaussian) pairs are sorted by
    pixel then depth, and the front-to-back transmittance is a segmented
    cumulative sum of ``log(1 - alpha)`` per pixel. A dense pixels x
    Gaussians pass was 226M entries for a 10k-vertex body in a 120x180 crop
    and took 0.3-0.5 s per render; the pairs number about 500k.
    """
    x0, y0, w, h = [int(v) for v in crop]
    dev = scene.xyz.device
    means, cov, depth = project(scene, K, R, t)
    keep = depth > MIN_DEPTH
    order = torch.argsort(depth[keep])
    idx = torch.nonzero(keep, as_tuple=True)[0][order]       # depth rank == position
    means, cov = means[idx], cov[idx]
    colour = scene.colour[idx].clamp(0.0, 1.0)
    opacity = torch.sigmoid(scene.opacity_logit[idx])
    n_g = means.shape[0]
    n_px = w * h

    a, b, c, d = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 0], cov[:, 1, 1]
    det = (a * d - b * c).clamp_min(1e-8)
    inv = torch.stack([torch.stack([d, -b], 1), torch.stack([-c, a], 1)], 1) / det[:, None, None]
    sig_max = torch.sqrt(torch.clamp(torch.maximum(a, d), min=1e-6)).detach()   # window size only

    # (pixel, gaussian, alpha) pairs inside the windows, built per chunk of
    # Gaussians so the window tensors stay small.
    pix_l, g_l, alpha_l = [], [], []
    for s0 in range(0, n_g, chunk):
        m, iv = means[s0:s0 + chunk], inv[s0:s0 + chunk]
        n_c = m.shape[0]
        r = int(min(MAX_WINDOW_PX,
                    max(1.0, float(torch.ceil(WINDOW_SIGMAS * sig_max[s0:s0 + chunk].max())))))
        offs = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], 1)      # [Wn, 2]
        pix = torch.round(m)[:, None, :] + offsets[None, :, :]          # [c, Wn, 2]
        dlt = pix - m[:, None, :]
        power = -0.5 * torch.einsum("cwi,cij,cwj->cw", dlt, iv, dlt)
        alpha_w = (opacity[s0:s0 + chunk, None] * torch.exp(power)).clamp(max=ALPHA_MAX)
        col = (pix[..., 0] - x0).long()
        row = (pix[..., 1] - y0).long()
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h) & (alpha_w >= ALPHA_CUT)
        gidx = (torch.arange(n_c, device=dev) + s0)[:, None].expand(-1, offsets.shape[0])
        pix_l.append((row * w + col)[inside])
        g_l.append(gidx[inside])
        alpha_l.append(alpha_w[inside])
    bg = (background.to(dev, torch.float32) if torch.is_tensor(background)
          else torch.as_tensor(np.asarray(background, np.float32), device=dev))
    bg = bg.reshape(-1, 3) if bg.ndim == 3 else bg[None, :].expand(n_px, 3)
    if not pix_l or sum(int(t_.numel()) for t_ in pix_l) == 0:
        empty = bg.reshape(h, w, 3).clone()
        return (empty, torch.ones(h, w, device=dev)) if return_transmittance else empty
    pidx = torch.cat(pix_l)
    gidx = torch.cat(g_l)
    alpha = torch.cat(alpha_l)

    # Sort by pixel, then depth (gidx is already the depth rank).
    key = pidx * n_g + gidx
    order = torch.argsort(key)
    pidx, gidx, alpha = pidx[order], gidx[order], alpha[order]
    log_t = torch.log1p(-alpha)                                          # log(1 - a)
    cs = torch.cumsum(log_t, 0)
    excl = cs - log_t                                                    # exclusive
    _vals, counts = torch.unique_consecutive(pidx, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    seg_start = torch.repeat_interleave(starts, counts)
    t_prev = torch.exp(excl - excl[seg_start])                          # transmittance before
    contrib = (t_prev * alpha)[:, None] * colour[gidx]
    C = torch.zeros(n_px, 3, device=dev).index_add(0, pidx, contrib)
    log_T = torch.zeros(n_px, device=dev).index_add(0, pidx, log_t)
    T = torch.exp(log_T)
    out = C + T[:, None] * bg
    if return_transmittance:
        return out.reshape(h, w, 3), T.reshape(h, w)
    return out.reshape(h, w, 3)
