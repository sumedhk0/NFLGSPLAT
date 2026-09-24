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


def canonical(body_pose: np.ndarray) -> np.ndarray:
    """The same rotations with every axis-angle vector folded under pi (``r -> r * (|r| - 2 pi) / |r|`` while
    ``|r| > pi``), shape kept. The timeline unwraps rotation vectors along a track for continuity and holds the
    last record forward, so a rendered body can carry ``|r| > pi``; VPoser was trained on canonical vectors and
    read such a hip as 27 where the same rotation reads 6 (2026-09-24). Every entry to the prior goes through
    here."""
    r = np.asarray(body_pose, float)
    out = r.reshape(-1, 3).copy()
    n = np.linalg.norm(out, axis=1)
    for _ in range(4):                                   # |r| < 3 pi in practice; loop for safety
        m = n > np.pi
        if not m.any():
            break
        out[m] *= ((n[m] - 2 * np.pi) / n[m])[:, None]
        n = np.linalg.norm(out, axis=1)
    return out.reshape(r.shape)


def encode(vp, body_pose: np.ndarray) -> np.ndarray:
    """Latent means ``[N, 32]`` for body poses ``[N, 21, 3]`` (or ``[N, 63]``) axis-angle (canonicalised first)."""
    import torch

    bp = canonical(np.asarray(body_pose, np.float32)).astype(np.float32).reshape(-1, 63)
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
    bp = canonical(np.asarray(body_pose, float)).reshape(-1, 21, 3)
    pr = np.asarray(projected, float).reshape(-1, 21, 3)
    w = np.asarray(w, float).reshape(-1, 1, 1) if np.ndim(w) else float(w)
    return (1 - w) * bp + w * pr


def weights_from_scores(s: np.ndarray, *, lo: float = None, hi: float = None) -> np.ndarray:
    """A per-pose blend weight from the score: 0 at or under ``lo``, 1 at or over ``hi``, linear between (the shift
    is spent on the implausible tail, never on a pose the prior already likes)."""
    lo = NORM_HI * 0.75 if lo is None else float(lo)
    hi = NORM_HI if hi is None else float(hi)
    return np.clip((np.asarray(s, float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


# ---- the encoder in numpy, for the fit ----------------------------------------------------------------------------
class NumpyEncoder:
    """VPoser's encoder mean as numpy: BatchNorm(63) -> Linear -> LeakyReLU(0.01) -> BatchNorm(512) -> Linear -> Linear
    -> mu. Built once from the torch model's state; a least-squares fit calls it ~70 times an iteration."""

    def __init__(self, vp):
        sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in vp.state_dict().items() if k.startswith("encoder_net")}
        def bn(i):
            g, b, m, v = sd[f"encoder_net.{i}.weight"], sd[f"encoder_net.{i}.bias"], sd[f"encoder_net.{i}.running_mean"], sd[f"encoder_net.{i}.running_var"]
            scale = g / np.sqrt(v + 1e-5)
            return scale, b - m * scale
        self.bn1 = bn(1); self.bn4 = bn(4)
        self.w2, self.b2 = sd["encoder_net.2.weight"], sd["encoder_net.2.bias"]
        self.w6, self.b6 = sd["encoder_net.6.weight"], sd["encoder_net.6.bias"]
        self.w7, self.b7 = sd["encoder_net.7.weight"], sd["encoder_net.7.bias"]
        self.wmu, self.bmu = sd["encoder_net.8.mu.weight"], sd["encoder_net.8.mu.bias"]

    def __call__(self, body_pose) -> np.ndarray:
        """Latent means ``[N, 32]`` (or ``[32]`` for one pose) from body poses ``[N, 63]`` / ``[63]`` / ``[N, 21, 3]``."""
        x = canonical(np.asarray(body_pose, np.float64))
        one = x.ndim == 1 or (x.ndim == 2 and x.shape == (21, 3))
        x = x.reshape(-1, 63)
        x = x * self.bn1[0] + self.bn1[1]
        x = x @ self.w2.T + self.b2
        x = np.where(x > 0, x, 0.01 * x)
        x = x * self.bn4[0] + self.bn4[1]
        x = x @ self.w6.T + self.b6
        x = x @ self.w7.T + self.b7
        z = x @ self.wmu.T + self.bmu
        return z[0] if one else z


_ENC = None


def encoder(vposer_dir: str | Path = VPOSER_DIR) -> NumpyEncoder:
    """The module's cached numpy encoder (loaded from the torch checkpoint on first use)."""
    global _ENC
    if _ENC is None:
        _ENC = NumpyEncoder(load(vposer_dir))
    return _ENC


# ---- the shift on a timeline -----------------------------------------------------------------------------------
# Measured on play 1 (2026-09-23, v105 as rendered, blend 8 -> 12 per frame): score p99 15.9 -> 9.7, the share over
# 8.0 6.8 -> 5.0 %, keypoint residual p50 6.3 -> 6.7 px, joint jitter p90 0.064 -> 0.062 but p99 0.185 -> 0.224:
# a per-frame weight switches between neighbouring frames and the blend's edges jump. The weight is therefore
# smoothed along each man's frames (SHIFT_SIGMA) before the blend.
SHIFT: bool = False             # apply the shift in the render scripts (read at call time); off until the film says yes
SHIFT_LO: float = 8.0           # the score where the blend toward the projection starts ...
SHIFT_HI: float = 12.0          # ... and where it is complete
SHIFT_SIGMA: float = 3.0        # frames: Gaussian smoothing of the per-frame weight along each man's run (0 = none)


def smooth_weights(w: np.ndarray, sigma: float) -> np.ndarray:
    """``w [T]`` smoothed along the frames by a Gaussian of ``sigma`` frames (edges held), so a moved frame's
    neighbours move part of the way and the blend has no edge to jump over; unchanged for sigma <= 0."""
    w = np.asarray(w, float)
    if sigma <= 0 or len(w) < 2:
        return w.copy()
    from scipy.ndimage import gaussian_filter1d
    return np.clip(gaussian_filter1d(w, sigma, mode="nearest"), 0.0, 1.0)


SUPPORT_LO: float = 8.0         # px: a body-frame whose keypoint residual is under this is film-supported: no shift ...
SUPPORT_HI: float = 16.0        # ... over this the shift is unrestricted; linear between (a low pass-protection crouch
                                # scored 11 on play 1 and sat on its keypoints; the sprinter's flung arms did not)


ARM_ROWS: tuple = (13, 14, 15, 16, 17, 18, 19, 20)   # collars, shoulders, elbows, wrists
SUPPORT_ARM_LO: float = 4.0     # px: the arm rows' ramp -- arm keypoints on a 100-px sprinter are the least reliable
SUPPORT_ARM_HI: float = 10.0    # (motion blur put the detector's elbow on a flung-back arm at 1.5-8 px, v107p), so an
                                # arm is freed sooner than a leg


def support_weights(resid, *, lo=None, hi=None) -> np.ndarray:
    """A multiplier in 0..1 from a per-frame keypoint residual (px): 0 at or under ``lo`` (the film backs the pose),
    1 at or over ``hi``; NaN (no keypoints) = 1. ``lo`` / ``hi`` may be arrays broadcast against ``resid``."""
    lo = SUPPORT_LO if lo is None else np.asarray(lo, float)
    hi = SUPPORT_HI if hi is None else np.asarray(hi, float)
    r = np.asarray(resid, float)
    w = np.clip((r - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    return np.where(np.isfinite(r), w, 1.0)


def row_ramps() -> tuple:
    """``(lo [21], hi [21])`` per body_pose row: the arm rows' ramp (SUPPORT_ARM_LO / HI) and the body's (SUPPORT_LO
    / HI) elsewhere, read at call time."""
    lo = np.full(21, float(SUPPORT_LO)); hi = np.full(21, float(SUPPORT_HI))
    for r in ARM_ROWS:
        lo[r], hi[r] = float(SUPPORT_ARM_LO), float(SUPPORT_ARM_HI)
    return lo, hi


# body_pose row r is SMPL-X joint r + 1; the joint at the END of each row's bone (its child) is the one whose keypoint
# residual says whether that rotation is right: rows without a keypointed child take the body's median residual (-1)
ROW_CHILD_COCO: dict = {
    0: 13, 1: 14,            # L/R hip rows -> the knees (COCO 13, 14)
    3: 15, 4: 16,            # L/R knee rows -> the ankles (15, 16)
    15: 7, 16: 8,            # L/R shoulder rows -> the elbows (7, 8)
    17: 9, 18: 10,           # L/R elbow rows -> the wrists (9, 10)
}


def row_support(resid_by_coco: dict, body_median: float) -> np.ndarray:
    """``[21]`` residual px per body_pose row from ``{coco_joint: px}`` (missing = NaN): each row takes its bone's
    child joint's residual (ROW_CHILD_COCO), the other rows the body median."""
    out = np.full(21, float(body_median) if np.isfinite(body_median) else np.nan)
    for r, c in ROW_CHILD_COCO.items():
        v = resid_by_coco.get(c, np.nan)
        out[r] = float(v) if v is not None and np.isfinite(v) else out[r]
    return out


def load_row_support(path) -> dict | None:
    """``{(pid, frame): [21] px}`` from 09j's pose_support.json, or None when the file is absent."""
    import json
    from pathlib import Path as _P
    path = _P(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {}
    for pid, byf in d.get("rows", {}).items():
        for f, rs in byf.items():
            out[(int(pid), int(f))] = np.array([np.nan if v is None else float(v) for v in rs])
    return out


def shift_timeline(tl, vp, *, lo=None, hi=None, sigma=None, lo_frame=None, hi_frame=None, support=None,
                   row_support_by=None) -> dict:
    """Blend every drawn body's pose toward its VPoser projection where its score is high: per id, over each run of
    consecutive frames, the weights from the scores (``lo``..``hi``), times the support multiplier where
    ``support`` ``{(pid, frame): keypoint residual px}`` is given (a pose the film backs is not moved), smoothed by
    ``sigma`` frames, then the blend in place. None knobs = the module's SHIFT_LO / SHIFT_HI / SHIFT_SIGMA at call
    time. Returns ``{"scored", "moved", "full", "mean_move_rad"}``."""
    lo = SHIFT_LO if lo is None else float(lo)
    hi = SHIFT_HI if hi is None else float(hi)
    sigma = SHIFT_SIGMA if sigma is None else float(sigma)
    by: dict = {}
    for f, states in tl.states.items():
        if (lo_frame is not None and f < lo_frame) or (hi_frame is not None and f > hi_frame):
            continue
        for s in states:
            by.setdefault(int(s.pid), {})[int(f)] = s
    rep = {"scored": 0, "moved": 0, "full": 0, "mean_move_rad": 0.0}
    moves = []
    for pid, byf in by.items():
        fs = sorted(byf)
        runs, run = [], [fs[0]]
        for f in fs[1:]:
            if f == run[-1] + 1:
                run.append(f)
            else:
                runs.append(run)
                run = [f]
        runs.append(run)
        for run in runs:
            bps = np.array([byf[f].body_pose.reshape(21, 3) for f in run])
            w = weights_from_scores(scores(vp, bps), lo=lo, hi=hi)
            if support is not None:
                w = w * support_weights([support.get((pid, f), np.nan) for f in run])
            w = smooth_weights(w, sigma)
            rep["scored"] += len(run)
            if not (w > 0).any():
                continue
            if row_support_by is not None:
                # per row: the frame's weight times the row's own support (its bone's child joint on or off its
                # keypoint), smoothed along the run row by row
                r_lo, r_hi = row_ramps()
                rs = np.array([support_weights(row_support_by.get((pid, f), np.full(21, np.nan)), lo=r_lo, hi=r_hi)
                               for f in run])
                wr = np.stack([smooth_weights(w * rs[:, r], sigma) for r in range(21)], axis=1)   # [T, 21]
                pr = project(vp, bps)
                new = (1 - wr[:, :, None]) * bps + wr[:, :, None] * pr
                weff = wr.max(axis=1)
            else:
                new = blend(bps, project(vp, bps), w)
                weff = w
            for f, bp, wi, old in zip(run, new, weff, bps):
                if wi > 0:
                    byf[f].body_pose = bp.reshape(21, 3)
                    rep["moved"] += 1
                    rep["full"] += int(wi >= 1.0)
                    moves.append(float(np.abs(bp - old).mean()))
    rep["mean_move_rad"] = float(np.mean(moves)) if moves else 0.0
    return rep
