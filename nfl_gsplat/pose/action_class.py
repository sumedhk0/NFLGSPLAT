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


# ---- the tracking-learnt prior applied to a play's pose records --------------------------------------------------
# A model trained on measured orientation (scripts/09l_facing_prior.py, BDB) predicts, from where a man moves and who he
# is, whether he runs forward, backpedals or moves sideways. Where it is confident about forward or backward on a
# frame only one camera sees, a pose record facing more than PRIOR_MAX_OFF_DEG off the facing that class implies is a
# one-view front/back or side-on error -- play 1's motion receiver, drawn facing the far sideline at 430-490 while the
# film has him sprinting downfield -- and is dropped, but only when a record of his within PRIOR_REACH frames agrees
# (nothing is invented: the timeline turns him to the records that were right).
PRIOR_MIN_P: float = 0.8
PRIOR_MAX_OFF_DEG: float = 90.0
PRIOR_REACH: int = 12
PRIOR_FRAME_SLACK: int = 2
COVERED_ROLES = ("WR", "TE", "RB", "QB", "DB", "LB")


def load_model(path):
    """``(net, mu, sd)`` saved by scripts/09l_facing_prior.py (--model-out)."""
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if list(blob.get("features", FEATURE_NAMES)) != list(FEATURE_NAMES):
        raise ValueError(f"{path}: trained on other features than this module's FEATURE_NAMES")
    net = _net(len(FEATURE_NAMES))
    net.load_state_dict(blob["state"])
    net.eval()
    return net, np.asarray(blob["mu"], float), np.asarray(blob["sd"], float)


def prior_table(xy_by_pid: dict, *, teams: dict, roles: dict, snap: int, release: int, passer: int, los_x: float,
                attack: float, model, half: int = 4) -> dict:
    """``{(pid, frame): (probs[3], heading)}`` for every moving sample from the snap to the release of the position
    groups the tracking covers; ``xy_by_pid``: pid -> {frame: (x, y)} ground track (metres, play frames)."""
    offence = teams.get(passer)
    keys, rows = [], []
    by_frame: dict = {}
    for p, tr in xy_by_pid.items():
        for f, v in tr.items():
            by_frame.setdefault(f, {})[p] = np.asarray(v, float)
    for p, tr in xy_by_pid.items():
        role = roles.get(p)
        if role not in COVERED_ROLES or teams.get(p) is None:
            continue
        is_off = teams[p] == offence
        for f in sorted(tr):
            if not snap <= f < release:
                continue
            m = track_motion(tr, f, half=half)
            if m is None or m[0] < V_MIN:
                continue
            here = np.asarray(tr[f], float)
            opp = [np.linalg.norm(q - here) for p2, q in by_frame.get(f, {}).items()
                   if teams.get(p2) is not None and teams.get(p2) != teams[p]]
            qb = by_frame.get(f, {}).get(passer)
            rows.append(feature_vector(dict(
                speed=m[0], heading=m[1], attack_sign=attack, offence=is_off, role=role, t_snap=(f - snap) / FPS,
                phase="pre", depth=attack * (here[0] - los_x) * (-1.0 if is_off else 1.0),
                ball_bearing=None if (qb is None or p == passer) else float(np.arctan2(qb[1] - here[1], qb[0] - here[0])),
                nearest_opp=min(opp) if opp else 10.0, nose=0.0, cam_bearing=None, lr_sign=0.0)))
            keys.append((p, f, m[1]))
    if not rows:
        return {}
    pr = predict_proba(model, np.stack(rows))
    return {(p, f): (pr[i], h) for i, (p, f, h) in enumerate(keys)}


def _record_yaw(rec) -> float:
    from scipy.spatial.transform import Rotation

    fw = Rotation.from_rotvec(np.asarray(rec[1], float).reshape(3)).apply([0.0, 0.0, 1.0])
    return float(np.arctan2(fw[1], fw[0]))


def drop_against_prior(poses_by_pid: dict, prior: dict, *, one_view=None, min_p: float = None, max_off_deg: float = None,
                       reach: int = None, frame_slack: int = None) -> tuple[dict, list]:
    """``poses_by_pid`` (pid -> {frame: (body_pose, orient, betas, source)}) without the records the prior outvotes
    (see PRIOR_MIN_P); ``prior``: prior_table's output; ``one_view``: {(frame, pid)} the rule may act on (None = all).
    Returns ``(poses, [(pid, frame), ...])``. Knobs default to the module's values at call time."""
    min_p = PRIOR_MIN_P if min_p is None else float(min_p)
    lim = np.radians(PRIOR_MAX_OFF_DEG if max_off_deg is None else float(max_off_deg))
    reach = PRIOR_REACH if reach is None else int(reach)
    slack = PRIOR_FRAME_SLACK if frame_slack is None else int(frame_slack)
    out: dict = {}
    dropped: list = []
    for pid, recs in poses_by_pid.items():
        fs = sorted(int(f) for f in recs)
        yaw = {f: _record_yaw(recs[f]) for f in fs}
        expect = {}
        for f in fs:
            hit = None
            for d in sorted(range(-slack, slack + 1), key=abs):
                hit = prior.get((pid, f + d))
                if hit is not None:
                    break
            if hit is None:
                continue
            probs, heading = hit
            c = int(np.argmax(probs))
            if probs[c] < min_p or c not in (FWD, BACK):
                continue
            expect[f] = expected_facing(c, heading)
        cand = [f for f, e in expect.items() if (one_view is None or (f, pid) in one_view)
                and abs(float(wrap(yaw[f] - e))) > lim]
        cset = set(cand)
        gone = [f for f in cand if any(abs(g - f) <= reach and g not in cset and abs(float(wrap(yaw[g] - expect[f]))) <= lim
                                       for g in fs if g != f)]
        kept = {f: r for f, r in recs.items() if int(f) not in set(gone)}
        dropped += [(pid, f) for f in sorted(gone)]
        if kept:
            out[pid] = kept
    return out, dropped
