"""Which way a man moves relative to his chest: a per-play classifier (forward, backward, sideways).

WHY. A one-view fit (the sideline camera alone) cannot tell which way a man faces when he is seen side-on, and the
regressor on a small crop cannot either; both put a sprinting receiver's chest 90-180 degrees off his run on play 1
(the motion receiver at 434-494, film: sprinting downfield). A rule "face where you run" was measured and REJECTED:
the fast backward movers on the film are real -- safeties backpedalling, linemen in their pass sets, the quarterback's
drop. What decides the facing is WHO is moving WHERE and WHEN: a receiver running downfield after the snap runs
forward, a safety moving toward his own end zone in the first second backpedals, a tackle sliding back sets.

WHAT. The play labels itself. Where both cameras see a man the two-view fit's facing is trustworthy (film-checked),
and its angle to the man's motion gives the class: FWD (within 45 degrees), BACK (beyond 135), LAT (between), SLOW
under V_MIN. A small classifier learns those classes from features that never look at the facing -- speed, the motion
against the attack direction, side and position group, time since the snap and the ball's phase, depth behind the
line, the ball's bearing against the motion, the nearest opponent, and what the sideline camera sees of his face
(nose confidence, the shoulders' left/right order) against the camera's bearing -- and is scored held out by PLAYER
(grouped folds), then predicts the one-view frames. Torch only (no scikit-learn in the venvs); numpy in, numpy out.
"""
from __future__ import annotations

import numpy as np

FPS: float = 59.94               # play frame numbers are the clip's 59.94 fps frames (the export's fps is its sample rate)
V_MIN: float = 1.5               # m/s: slower than this has no direction to judge
FWD, BACK, LAT, SLOW = 0, 1, 2, 3
CLASSES = (FWD, BACK, LAT)
CLASS_NAMES = {FWD: "forward", BACK: "backward", LAT: "sideways", SLOW: "slow"}
ROLES = ("OL", "DL", "LB", "DB", "WR", "TE", "RB", "QB")
PHASES = ("pre", "flight", "after")
FEATURE_NAMES = (["log_speed", "down_cos", "down_sin", "offence"] + [f"role_{r}" for r in ROLES]
                 + ["t_snap", "phase_pre", "phase_flight", "phase_after", "depth", "ball_cos", "ball_sin",
                    "opp_near", "nose", "cam_cos", "cam_sin", "nose_cam_cos", "lr_cam_cos"])


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi


def relative_class(facing: float, velocity, v_min: float = None) -> int:
    """FWD / BACK / LAT from the angle between a facing (radians, ground plane) and a velocity (m/s, 2D); SLOW under
    ``v_min`` (default V_MIN)."""
    v = np.asarray(velocity, float)
    if float(np.linalg.norm(v)) < (V_MIN if v_min is None else v_min):
        return SLOW
    d = abs(float(wrap(facing - np.arctan2(v[1], v[0]))))
    if d <= np.radians(45):
        return FWD
    if d >= np.radians(135):
        return BACK
    return LAT


def track_motion(xy: dict, f: int, half: int = 4):
    """``(speed m/s, heading rad)`` of a ground track ``{frame: (x, y)}`` at play frame ``f`` from the positions
    ``half`` frames either side (59.94 fps); None when either is missing."""
    a, b = xy.get(f - half), xy.get(f + half)
    if a is None or b is None:
        return None
    v = (np.asarray(b, float) - np.asarray(a, float)) / (2 * half / FPS)
    return float(np.linalg.norm(v)), float(np.arctan2(v[1], v[0]))


def feature_vector(row: dict) -> np.ndarray:
    """One sample's features (FEATURE_NAMES) from a dict: speed, heading (rad), attack_sign (+1 when the offence
    attacks +x), offence (bool), role, t_snap (s), phase, depth (m behind his own side of the line), ball_bearing
    (rad, field), nearest_opp (m), nose (sideline nose confidence, 0 when unseen), cam_bearing (rad, from the man to
    the sideline camera), lr_sign (+1 the sideline shoulders say he faces the camera, -1 his back, 0 unknown)."""
    h = float(row["heading"])
    down = wrap(h - (0.0 if row["attack_sign"] > 0 else np.pi))          # motion against the offence's attack
    side = 1.0 if row["offence"] else -1.0
    roles = [1.0 if row.get("role") == r else 0.0 for r in ROLES]
    ph = [1.0 if row.get("phase") == p else 0.0 for p in PHASES]
    bb = row.get("ball_bearing")
    ball = (0.0, 0.0) if bb is None else (float(np.cos(wrap(bb - h))), float(np.sin(wrap(bb - h))))
    cb = row.get("cam_bearing")
    cam = (0.0, 0.0) if cb is None else (float(np.cos(wrap(cb - h))), float(np.sin(wrap(cb - h))))
    nose = float(row.get("nose") or 0.0)
    lr = float(row.get("lr_sign") or 0.0)
    x = ([np.log1p(float(row["speed"])), float(np.cos(down)), float(np.sin(down)), side] + roles
         + [float(np.clip(row.get("t_snap", 0.0), -1.0, 6.0))] + ph
         + [float(np.clip(row.get("depth", 0.0), -10.0, 25.0)) / 10.0, ball[0], ball[1],
            float(np.clip(row.get("nearest_opp", 10.0), 0.0, 10.0)) / 10.0, nose, cam[0], cam[1],
            # the face toward the camera AND the camera ahead of the motion = moving forward (and the shoulders' order)
            nose * cam[0], lr * cam[0]])
    return np.asarray(x, float)


def expected_facing(cls: int, heading: float):
    """The facing a class implies for a motion heading: FWD the heading, BACK its opposite, else None."""
    if cls == FWD:
        return float(wrap(heading))
    if cls == BACK:
        return float(wrap(heading + np.pi))
    return None


def _net(n_in: int, hidden: int = 32):
    import torch

    return torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                               torch.nn.ReLU(), torch.nn.Linear(hidden, len(CLASSES)))


def train(X, y, *, seed: int = 0, epochs: int = 400, weight_decay: float = 1e-3, lr: float = 1e-2):
    """A small MLP (two hidden layers of 32) on standardised features, class-balanced cross entropy; returns
    ``(net, mean, std)``."""
    import torch

    X = np.asarray(X, float); y = np.asarray(y, int)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    torch.manual_seed(seed)
    net = _net(X.shape[1])
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
    return net, mu, sd


def predict_proba(model, X) -> np.ndarray:
    import torch

    net, mu, sd = model
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor((np.asarray(X, float) - mu) / sd, dtype=torch.float32)), dim=1)
    return p.numpy().astype(float)


def grouped_cv_accuracy(X, y, groups, *, folds: int = 5, seed: int = 0, per_class: bool = False, **kw):
    """Accuracy with whole groups (player ids) held out per fold -- a model scored on men it never saw."""
    X = np.asarray(X, float); y = np.asarray(y, int); groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    pred = np.full(len(y), -1)
    for k in range(folds):
        hold = np.isin(groups, ug[k::folds])
        if hold.all() or not hold.any():
            continue
        m = train(X[~hold], y[~hold], seed=seed, **kw)
        pred[hold] = np.argmax(predict_proba(m, X[hold]), axis=1)
    ok = pred >= 0
    acc = float(np.mean(pred[ok] == y[ok]))
    if not per_class:
        return acc
    return acc, {CLASS_NAMES[c]: float(np.mean(pred[ok & (y == c)] == c)) if np.any(ok & (y == c)) else float("nan")
                 for c in CLASSES}, pred
