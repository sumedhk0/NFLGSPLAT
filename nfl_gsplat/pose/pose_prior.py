"""A pretrained pose prior (VPoser) as a ruler for "weird joints" and as the shift toward likely poses.

WHY. The user asked for a model that flags implausible joint configurations and moves them where they are likely
to be. Training one on play 1's own fits would teach it play 1's own mistakes; VPoser (Pavlakos et al. 2019, V02_05
retrained on AMASS for SMPL-X) already knows what human bodies do. It is a variational autoencoder over the 21-joint
SMPL-X body pose (63 axis-angle numbers) with a 32-d latent and a unit-Gaussian prior, so two things fall out
without training: the latent's distance from the origin scores how far a pose sits from the mocap manifold
(the plausibility ruler), and decode(encode(pose)) is the nearest plausible pose (the shift).

WHAT. ``load`` reads the checkpoint from data/body_models/vposer/V02_05 (human_body_prior, torch, CPU is enough).
``scores`` gives the latent norm per body-frame; ``project`` the projected body poses; ``blend`` moves a pose toward
its projection by a weight, which the caller gates on the keypoint residual so a pose the film supports is not
overwritten. Nothing here touches the fit; the timeline's body_pose rows go in and come out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VPOSER_DIR: str = "data/body_models/vposer/V02_05"
NORM_HI: float = 8.0            # a latent norm above this reads as implausible on play 1 (set by the score distribution)


def load(vposer_dir: str | Path = VPOSER_DIR):
    """The VPoser model (eval, no grad) from its experiment directory (snapshots/*.ckpt + *.yaml). Loaded by hand:
    human_body_prior's load_model splits paths on '/', which on Windows leaves it unable to find the yaml."""
    import glob
    import os

    import torch
    from human_body_prior.models.vposer_model import VPoser
    from omegaconf import OmegaConf

    d = Path(vposer_dir)
    yamls = sorted(glob.glob(str(d / "*.yaml")))
    ckpts = sorted(glob.glob(str(d / "snapshots" / "*.ckpt")), key=os.path.getmtime)
    if not yamls or not ckpts:
        raise FileNotFoundError(f"VPoser needs a *.yaml and snapshots/*.ckpt under {d}")
    cfg = OmegaConf.load(yamls[0])
    vp = VPoser(cfg)
    sd = torch.load(ckpts[-1], map_location="cpu", weights_only=False)["state_dict"]
    sd = {(k[len("vp_model."):] if k.startswith("vp_model.") else k): v for k, v in sd.items()}
    vp.load_state_dict(sd, strict=False)
    for p in vp.parameters():
        p.requires_grad_(False)
    vp.eval()
    return vp


def encode(vp, body_pose: np.ndarray) -> np.ndarray:
    """Latent means ``[N, 32]`` for body poses ``[N, 21, 3]`` (or ``[N, 63]``) axis-angle."""
    import torch

    bp = np.asarray(body_pose, np.float32).reshape(-1, 63)
    with torch.no_grad():
        q = vp.encode(torch.from_numpy(bp))
    return q.mean.cpu().numpy()


def decode(vp, z: np.ndarray) -> np.ndarray:
    """Body poses ``[N, 21, 3]`` axis-angle decoded from latents ``[N, 32]``."""
    import torch

    with torch.no_grad():
        out = vp.decode(torch.from_numpy(np.asarray(z, np.float32)))
    return out["pose_body"].cpu().numpy().reshape(-1, 21, 3)


def scores(vp, body_pose: np.ndarray) -> np.ndarray:
    """The plausibility ruler: the latent mean's norm per pose ``[N]`` (0 = the mean pose; mocap sits mostly under
    ~5; the fits' tail is what the ruler is for)."""
    return np.linalg.norm(encode(vp, body_pose), axis=1)


def project(vp, body_pose: np.ndarray) -> np.ndarray:
    """decode(encode(pose)): the nearest plausible pose per body-frame ``[N, 21, 3]``."""
    return decode(vp, encode(vp, body_pose))


def blend(body_pose: np.ndarray, projected: np.ndarray, w) -> np.ndarray:
    """Move each pose toward its projection by ``w`` (scalar or ``[N]``) per joint on the rotation vectors."""
    bp = np.asarray(body_pose, float).reshape(-1, 21, 3)
    pr = np.asarray(projected, float).reshape(-1, 21, 3)
    w = np.asarray(w, float).reshape(-1, 1, 1) if np.ndim(w) else float(w)
    return (1 - w) * bp + w * pr


def weights_from_scores(s: np.ndarray, *, lo: float = None, hi: float = None) -> np.ndarray:
    """A per-pose blend weight from the score: 0 at or under ``lo``, 1 at or over ``hi``, linear between (the shift
    is spent on the implausible tail, never on a pose the prior already likes)."""
    lo = NORM_HI * 0.75 if lo is None else float(lo)
    hi = NORM_HI if hi is None else float(hi)
    return np.clip((np.asarray(s, float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
