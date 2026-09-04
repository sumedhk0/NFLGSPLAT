"""Fit one body's Gaussian appearance to the footage (M8, step 2).

See docs/DESIGN_photometric_appearance.md. One body, every frame it was
posed in, both views: the learnable colour, scale multiplier and opacity of
its vertex-bound Gaussians are shared across frames, and each frame's scene
is that body's placed vertices for the frame run through the usual binding
(mesh_to_gaussians). Loss is L1 between the body composited OVER the frame
crop and the crop itself -- so pixels the body does not cover cost nothing,
which is the mask -- plus a total-variation prior on colour over mesh edges,
plus a per-frame 2-D translation nuisance (a shift of the principal point)
so a few pixels of placement error are absorbed rather than painted in.

Positions are never optimised: the pose stage owns geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from nfl_gsplat.compositing import splat_torch as st
from nfl_gsplat.compositing.merge_ply import GaussianBatch
from nfl_gsplat.compositing.mesh_to_gaussians import mesh_to_gaussians, vertex_normals
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass
class FrameObs:
    image: np.ndarray        # [H, W, 3] RGB, float in [0, 1] or uint8
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    vertices: np.ndarray     # [V, 3] world, placed for this frame


@dataclass
class FitConfig:
    iters: int = 200
    lr: float = 0.02
    lr_shift: float = 0.1
    tv_weight: float = 0.01
    margin_px: int = 6
    translation: bool = True
    device: str = "cpu"
    log_every: int = 50
    # Turf bleed: a silhouette pixel is half grass, and an L1 over the crop
    # paints the edge vertices green (play 2's bodies came out olive). Each
    # pixel's error is weighted by the body's coverage there, (1 - T)^p, so
    # interior pixels decide the colours and edges barely count; and a weak
    # prior keeps every vertex near its starting colour.
    coverage_power: float = 2.0
    colour_prior: float = 0.02


@dataclass
class BodyAppearance:
    colour: torch.Tensor          # [V, 3]
    log_scale_mult: torch.Tensor  # [V]
    opacity_logit: torch.Tensor   # [V]
    shift: torch.Tensor | None = None   # [F, 2] per-frame principal-point shift, px
    history: dict = field(default_factory=dict)

    def apply(self, batch: GaussianBatch) -> GaussianBatch:
        """A body's batch (any frame's placement) with the fitted appearance."""
        scene = st.SceneParams.from_batch(batch)
        scene.colour = self.colour.detach().clone()
        scene.log_scale_mult = self.log_scale_mult.detach().clone()
        scene.opacity_logit = self.opacity_logit.detach().clone()
        return scene.to_batch()


def _image01(img) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def crop_for(ob: FrameObs, *, margin_px: int = 6):
    """``(x0, y0, w, h)`` around the body's projected vertices, inside the image."""
    q = (ob.K @ (ob.R @ np.asarray(ob.vertices, float).T + np.asarray(ob.t, float).reshape(3, 1))).T
    ok = q[:, 2] > st.MIN_DEPTH
    uv = q[ok, :2] / q[ok, 2:]
    Hh, Ww = ob.image.shape[:2]
    x0 = int(max(0, np.floor(uv[:, 0].min()) - margin_px))
    y0 = int(max(0, np.floor(uv[:, 1].min()) - margin_px))
    x1 = int(min(Ww, np.ceil(uv[:, 0].max()) + margin_px + 1))
    y1 = int(min(Hh, np.ceil(uv[:, 1].max()) + margin_px + 1))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def seen_vertices(faces, obs: list[FrameObs]) -> np.ndarray:
    """``[V]`` bool: faces some camera (normal toward it) and projects in frame."""
    seen = None
    for ob in obs:
        v = np.asarray(ob.vertices, float)
        n = vertex_normals(v, np.asarray(faces))
        cam = -np.asarray(ob.R).T @ np.asarray(ob.t).reshape(3)
        facing = np.einsum("ij,ij->i", n, cam[None, :] - v) > 0
        q = (ob.K @ (ob.R @ v.T + np.asarray(ob.t).reshape(3, 1))).T
        uv = q[:, :2] / np.maximum(q[:, 2:], 1e-9)
        Hh, Ww = ob.image.shape[:2]
        inside = (q[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < Ww) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh)
        s = facing & inside
        seen = s if seen is None else (seen | s)
    return seen


def mesh_edges(faces) -> np.ndarray:
    f = np.asarray(faces, np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def fit_body(colour0, faces, obs: list[FrameObs], cfg: FitConfig | None = None):
    """Fit the body's appearance to ``obs``; returns ``(BodyAppearance, history)``."""
    cfg = cfg or FitConfig()
    dev = torch.device(cfg.device)
    faces = np.asarray(faces, np.int64)
    colour0_t = torch.as_tensor(np.asarray(colour0, np.float32), device=dev)
    colour = colour0_t.clone().requires_grad_(True)
    n_v = colour.shape[0]

    frames = []
    for ob in obs:
        batch = mesh_to_gaussians(np.asarray(ob.vertices, float), faces)
        base = st.SceneParams.from_batch(batch, device=dev)
        x0, y0, w, h = crop_for(ob, margin_px=cfg.margin_px)
        target = torch.as_tensor(_image01(ob.image)[y0:y0 + h, x0:x0 + w], device=dev)
        frames.append((base, (x0, y0, w, h), target,
                       torch.as_tensor(np.asarray(ob.K, np.float32), device=dev),
                       ob.R, ob.t))
    log_scale_mult = torch.zeros(n_v, device=dev, requires_grad=True)
    opacity_logit = frames[0][0].opacity_logit.clone().requires_grad_(True)
    shift = torch.zeros(len(frames), 2, device=dev, requires_grad=cfg.translation)

    groups = [{"params": [colour, log_scale_mult, opacity_logit], "lr": cfg.lr}]
    if cfg.translation:
        groups.append({"params": [shift], "lr": cfg.lr_shift})
    opt = torch.optim.Adam(groups)
    edges = torch.as_tensor(mesh_edges(faces), device=dev)
    hist = {"loss": []}
    n_f = len(frames)
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        # One frame's graph at a time: summing every render into one loss kept
        # tens of GB alive for backward (54 GB on a 60-frame body). Gradients
        # accumulate across frames; one optimiser step per iteration.
        for fi, (base, crop, target, K, R, t) in enumerate(frames):
            scene = st.SceneParams(xyz=base.xyz, rot=base.rot, log_scale=base.log_scale,
                                   colour=colour, log_scale_mult=log_scale_mult,
                                   opacity_logit=opacity_logit, sh_k=base.sh_k)
            Kf = K
            if cfg.translation:
                Kf = K + torch.zeros_like(K).index_put(
                    (torch.tensor([0, 1], device=dev), torch.tensor([2, 2], device=dev)), shift[fi])
            img, T = st.render(scene, Kf, R, t, crop=crop, background=target,
                               return_transmittance=True)
            if not img.requires_grad:                # no Gaussian touched this crop
                continue
            weight = (1.0 - T.detach()).clamp(0, 1) ** cfg.coverage_power
            part = ((img - target).abs().mean(-1) * weight).sum() / (weight.sum() + 1e-6) / n_f
            part.backward()
            total += part.item()
        if cfg.tv_weight > 0:
            tv = cfg.tv_weight * (colour[edges[:, 0]] - colour[edges[:, 1]]).abs().mean()
            tv.backward()
            total += tv.item()
        if cfg.colour_prior > 0:
            prior = cfg.colour_prior * (colour - colour0_t).abs().mean()
            prior.backward()
            total += prior.item()
        opt.step()
        with torch.no_grad():
            colour.clamp_(0.0, 1.0)
        hist["loss"].append(total)
        if cfg.log_every and it % cfg.log_every == 0:
            _LOG.info("fit: iter %d loss %.4f", it, total)
    hist["shift"] = shift.detach().cpu().numpy().tolist()
    return BodyAppearance(colour.detach(), log_scale_mult.detach(), opacity_logit.detach(),
                          shift.detach() if cfg.translation else None, hist), hist
