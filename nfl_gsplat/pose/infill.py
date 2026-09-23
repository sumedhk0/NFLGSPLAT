"""A self-supervised joint in-filler, fitted to one play.

WHY. The user's ask (2026-09-23): a model of this play's own motion that fills the joints the cameras did not see with
what the play shows elsewhere -- the same runner's stride repeats, a lineman's stance repeats, the body's joints move
together. Overfitting to the play is the point: the model is a per-play prior like the field splat, not a model of
football. What it must never do is learn our own fits' habits at the joints they got wrong, so it trains ONLY on the
joints the play is sure of (a confident keypoint in a camera, see ``sure_mask``) and is scored on sure joints it was
not shown (``holdout_score``), against the fill in use today (SLERP between the neighbouring sure keyframes, the same
thing timeline.gate_low_confidence does). Only if it beats that on held-out joints does it touch the play.

WHAT. Per (player, keyframe) a body pose is 21 axis-angle rows. A window of ``2 * HALF + 1`` keyframes is the model's
input: the rows (the masked rows carrying today's SLERP fill), a per-row "sure" flag and a "masked" flag. A small 1-D
convolutional network over the window predicts the CORRECTION over the SLERP fill at the centre keyframe, so the model
starts where today's fill is and can only add what the play's context supports (a first version that predicted the
rows outright lost to SLERP on a smooth synthetic signal, 2.8 against 1.1 degrees). Training masks runs of sure rows
the way occlusion hides them; inference masks the unsure rows. Numpy in and out; torch only in ``train`` / ``predict``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

J: int = 21                 # body_pose rows
HALF: int = 4               # keyframes of context either side of the target
SURE_CONF: float = 0.6      # a row is sure when its driving keypoint's confidence is at least this
UNSURE_CONF: float = 0.3    # ... and filled when under this (the gate's threshold)


@dataclass
class Sequence:
    """One player's keyframes: ``frames [K]`` sorted, ``pose [K, J, 3]`` axis-angles, ``conf [K, J]`` (nan = unknown)."""
    pid: int
    frames: np.ndarray
    pose: np.ndarray
    conf: np.ndarray


@dataclass
class Windows:
    x: np.ndarray            # [N, W, J*3 + J + J]: the rows (SLERP-filled where masked), the sure flags, the masked flags
    y: np.ndarray            # [N, J, 3] the true rows at the centre keyframe
    base: np.ndarray         # [N, J, 3] the SLERP fill at the centre keyframe (the model predicts y - base)
    mask: np.ndarray         # [N, J] rows to predict (True) at the centre
    sure: np.ndarray         # [N, J] the centre's sure flags
    who: list = field(default_factory=list)   # (pid, frame) per window


def sure_mask(conf: np.ndarray, *, sure_conf: float = SURE_CONF) -> np.ndarray:
    """``[K, J]`` True where the confidence is known and at least ``sure_conf``."""
    c = np.asarray(conf, float)
    return np.isfinite(c) & (c >= sure_conf)


def unsure_mask(conf: np.ndarray, *, unsure_conf: float = UNSURE_CONF) -> np.ndarray:
    """``[K, J]`` True where the confidence is known and under ``unsure_conf`` (the rows to fill)."""
    c = np.asarray(conf, float)
    return np.isfinite(c) & (c < unsure_conf)


def slerp_fill(frames: np.ndarray, pose: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """The fill in use today: per row, the rows where ``keep`` is False are SLERPed between the nearest kept
    keyframes either side; edge runs take the nearest kept value. ``frames [K]``, ``pose [K, J, 3]``, ``keep [K, J]``."""
    out = np.array(pose, float, copy=True)
    fk = np.asarray(frames, float)
    for j in range(pose.shape[1]):
        idx = np.flatnonzero(keep[:, j])
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            out[:, j] = pose[idx[0], j]
            continue
        rot = Rotation.from_rotvec(pose[idx, j])
        s = Slerp(fk[idx], rot)
        inside = (fk >= fk[idx[0]]) & (fk <= fk[idx[-1]])
        out[inside, j] = s(fk[inside]).as_rotvec()
        out[fk < fk[idx[0]], j] = pose[idx[0], j]
        out[fk > fk[idx[-1]], j] = pose[idx[-1], j]
    return out


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between axis-angle rotations ``a`` and ``b`` (broadcast over leading dims, last dim 3)."""
    a = np.asarray(a, float).reshape(-1, 3)
    b = np.asarray(b, float).reshape(-1, 3)
    d = Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)
    return np.degrees(np.linalg.norm(d.as_rotvec(), axis=1))


def build_windows(seqs: list[Sequence], masks: list[np.ndarray], *, half: int = HALF) -> Windows:
    """Windows for every keyframe of every sequence whose ``masks[i]`` row set at that keyframe is non-empty. The
    masked rows are replaced, in every keyframe of the sequence, by the SLERP fill from the sure rows that are not
    masked (so a masked run reads as SLERP would draw it); the input carries the rows, the sure flags (masked rows
    read as not sure) and the masked flags. Edge keyframes pad with the edge value and a zero sure flag."""
    xs, ys, bs, ms, ss, who = [], [], [], [], [], []
    W = 2 * half + 1
    for s, m in zip(seqs, masks):
        K = len(s.frames)
        sure = sure_mask(s.conf) & ~m
        base = slerp_fill(s.frames, s.pose, sure)
        # every row that is not sure (masked, unsure, unknown) reads as SLERP draws it, so the model never learns to
        # read an unsure row's value: in training the hidden rows would otherwise carry their true values as context
        # while at test time the same rows carry the fill (that mismatch cost 3.7 vs 1.5 deg on the synthetic long hole)
        rows_all = np.where(sure[:, :, None], s.pose, base)
        for k in range(K):
            if not m[k].any():
                continue
            span = np.arange(k - half, k + half + 1)
            idx = np.clip(span, 0, K - 1)
            rows = rows_all[idx].copy()                          # [W, J, 3]
            sflag = sure[idx].astype(float)                      # [W, J]
            sflag[(span < 0) | (span >= K)] = 0.0
            mflag = m[idx].astype(float)
            xs.append(np.concatenate([rows.reshape(W, J * 3), sflag, mflag], axis=1))
            ys.append(s.pose[k])
            bs.append(base[k])
            ms.append(m[k])
            ss.append(sure[k])
            who.append((s.pid, int(s.frames[k])))
    if not xs:
        z = np.zeros((0, J, 3))
        return Windows(np.zeros((0, W, J * 5)), z, z, np.zeros((0, J), bool), np.zeros((0, J), bool), [])
    return Windows(np.stack(xs), np.stack(ys), np.stack(bs), np.stack(ms), np.stack(ss), who)


RUN: int = 4                # keyframes a masked run lasts (occlusion hides a limb for a stretch, not a frame)
# What occlusion hides is a LIMB: its rows go together. Masks (training and hold-out alike) hide one limb's rows over
# a run; the model then fills a limb from the other limbs and from time, which is the question at inference.
LIMBS = {"L arm": [15, 17, 19], "R arm": [16, 18, 20], "L leg": [0, 3, 6], "R leg": [1, 4, 7]}


def run_masks(seqs: list[Sequence], *, share: float = 0.15, run: int = RUN, seed: int = 0) -> list[np.ndarray]:
    """Training masks: per limb, random runs of ``run`` keyframes over its SURE rows until about ``share`` of them
    are masked."""
    rng = np.random.default_rng(seed)
    out = []
    for s in seqs:
        sure = sure_mask(s.conf)
        K = len(s.frames)
        m = np.zeros_like(sure)
        for rows in LIMBS.values():
            n_sure = int(sure[:, rows].sum())
            if n_sure == 0:
                continue
            target = max(1, int(round(share * n_sure)))
            tries = 0
            while m[:, rows].sum() < target and tries < 10 * K:
                a = int(rng.integers(0, K))
                m[a:a + run, rows] |= sure[a:a + run, rows]
                tries += 1
        out.append(m)
    return out


def holdout_split(seqs: list[Sequence], *, every: int = 20, run: int = RUN, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``(train_masks, test_masks)``: at every ``every``-th keyframe one limb (cycling through LIMBS) has its sure rows
    held out for ``run`` keyframes (never a training target, never seen as sure -- see ``hide``); the rest get run
    training masks."""
    train = run_masks(seqs, run=run, seed=seed)
    limbs = list(LIMBS.values())
    test = []
    for s, m in zip(seqs, train):
        sure = sure_mask(s.conf)
        held = np.zeros_like(sure)
        for i, a in enumerate(range(every - 1, len(s.frames), every)):
            rows = limbs[i % len(limbs)]
            held[a:a + run, rows] = sure[a:a + run, rows]
        m[held] = False
        test.append(held)
    return train, test


def hide(seqs: list[Sequence], test_masks: list[np.ndarray]) -> list[Sequence]:
    """Copies of ``seqs`` whose held-out rows carry an unknown confidence (nan): the model can neither train on
    them nor see them as sure context."""
    out = []
    for s, held in zip(seqs, test_masks):
        c = np.array(s.conf, float, copy=True)
        c[held] = np.nan
        out.append(Sequence(s.pid, s.frames, s.pose, c))
    return out


def train(win: Windows, *, epochs: int = 60, lr: float = 2e-3, width: int = 128, seed: int = 0, device: str = "cpu",
          val_every: int = 7):
    """Fit the in-filler on ``win`` (masked rows -> the correction over SLERP at the centre). The correction head
    starts at zero, so epoch 0 IS today's fill; every ``val_every``-th window is validation, the epoch with the
    lowest validation error is kept, and if none beats epoch 0 the model returns the zero correction. Returns the
    torch model (eval mode) with ``net.best_epoch`` (0 = SLERP wins) and ``net.val_curve``."""
    import copy

    import torch
    from torch import nn

    torch.manual_seed(seed)
    W, C = win.x.shape[1], win.x.shape[2]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(C, width, 3, padding=1), nn.GELU(),
                nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
                nn.Conv1d(width, width, 3, padding=1), nn.GELU())
            self.head = nn.Linear(width * W, J * 3)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):                      # x [N, W, C]
            h = self.conv(x.transpose(1, 2))       # [N, width, W]
            return self.head(h.flatten(1)).view(-1, J, 3)

    net = Net().to(device)
    x = torch.tensor(win.x, dtype=torch.float32, device=device)
    y = torch.tensor(win.y - win.base, dtype=torch.float32, device=device)     # the correction over SLERP
    m = torch.tensor(win.mask, dtype=torch.float32, device=device)[:, :, None]
    n = len(x)
    is_val = torch.zeros(n, dtype=torch.bool, device=device)
    if val_every and n >= 2 * val_every:
        is_val[val_every - 1::val_every] = True
    tr_idx = torch.nonzero(~is_val).flatten()
    va_idx = torch.nonzero(is_val).flatten()

    def val_err():
        if len(va_idx) == 0:
            return float("nan")
        with torch.no_grad():
            pred = net(x[va_idx])
            return float((((pred - y[va_idx]) ** 2 * m[va_idx]).sum() / m[va_idx].sum().clamp(min=1.0)).sqrt())

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    bs = 256
    curve = [val_err()]                       # epoch 0: the zero correction = SLERP
    best, best_state, best_ep = curve[0], copy.deepcopy(net.state_dict()), 0
    for ep in range(1, epochs + 1):
        net.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            pred = net(x[b])
            loss = ((pred - y[b]) ** 2 * m[b]).sum() / m[b].sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        v = val_err()
        curve.append(v)
        if len(va_idx) == 0 or v < best - 1e-6:
            best, best_state, best_ep = v, copy.deepcopy(net.state_dict()), ep
    net.load_state_dict(best_state)
    net.eval()
    net.best_epoch = best_ep
    net.val_curve = curve
    return net


def predict(net, win: Windows, *, device: str = "cpu") -> np.ndarray:
    """``[N, J, 3]`` the SLERP fill plus the model's correction at the centre keyframe of every window (only the
    masked rows mean anything)."""
    import torch

    with torch.no_grad():
        x = torch.tensor(win.x, dtype=torch.float32, device=device)
        out = []
        for i in range(0, len(x), 1024):
            out.append(net(x[i:i + 1024]).cpu().numpy())
    res = np.concatenate(out) if out else np.zeros((0, J, 3))
    return win.base + res


def holdout_score(seqs: list[Sequence], *, every: int = 20, run: int = RUN, seed: int = 0, half: int = HALF, **train_kw) -> dict:
    """Train on the play with runs of ``run`` keyframes (every ``every``-th) hidden, then score the model and today's
    SLERP fill on recovering the hidden sure rows: mean and p90 geodesic error in degrees, per limb group too."""
    train_m, test_m = holdout_split(seqs, every=every, run=run, seed=seed)
    hidden = hide(seqs, test_m)
    win_train = build_windows(hidden, train_m, half=half)
    net = train(win_train, seed=seed, **train_kw)
    win_test = build_windows(hidden, test_m, half=half)
    pred = predict(net, win_test)
    err_model, err_slerp, rows = [], [], []
    for i, (pid, f) in enumerate(win_test.who):
        for j in np.flatnonzero(win_test.mask[i]):
            err_model.append(geodesic_deg(pred[i, j], win_test.y[i, j])[0])
            err_slerp.append(geodesic_deg(win_test.base[i, j], win_test.y[i, j])[0])   # today's fill of the same rows
            rows.append(j)
    em, es, rows = np.asarray(err_model), np.asarray(err_slerp), np.asarray(rows)
    groups = {"arms": [15, 16, 17, 18, 19, 20], "legs": [0, 1, 3, 4, 6, 7]}
    by = {}
    for name, js in groups.items():
        sel = np.isin(rows, js)
        if sel.any():
            by[name] = {"model_mean": float(em[sel].mean()), "slerp_mean": float(es[sel].mean()), "n": int(sel.sum())}
    return {"n": int(len(em)), "best_epoch": int(getattr(net, "best_epoch", -1)),
            "model_mean": float(em.mean()) if len(em) else float("nan"),
            "model_p90": float(np.percentile(em, 90)) if len(em) else float("nan"),
            "slerp_mean": float(es.mean()) if len(es) else float("nan"),
            "slerp_p90": float(np.percentile(es, 90)) if len(es) else float("nan"),
            "by_group": by, "net": net}


def fill(seqs: list[Sequence], net, *, half: int = HALF, unsure_conf: float = UNSURE_CONF) -> dict:
    """``{pid: {frame: pose[J, 3]}}`` with the unsure rows replaced by the model's prediction, the rest untouched."""
    masks = [unsure_mask(s.conf, unsure_conf=unsure_conf) for s in seqs]
    win = build_windows(seqs, masks, half=half)
    pred = predict(net, win)
    out = {s.pid: {int(f): np.array(s.pose[k], float, copy=True) for k, f in enumerate(s.frames)} for s in seqs}
    for i, (pid, f) in enumerate(win.who):
        rows = np.flatnonzero(win.mask[i])
        out[pid][f][rows] = pred[i, rows]
    return out
