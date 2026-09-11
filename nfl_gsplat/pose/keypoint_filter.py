"""Temporal outlier rejection on 2-D keypoint tracks before any fit.

WHY. The detector's wrists jump: on play 1's sideline keypoints (confident
ones only) a wrist moves 1.6 px between frames at the median and 44 px at
the p99 -- a left/right swap or a miss for one frame -- and every fit that
follows the keypoints throws the arm there and back ("arms all over the
place", the user on v24). Wrists are also the least confident joints
(30 % under 0.5) where hips and knees never are.

WHAT. Per (camera, player, joint), in frame order, a confident point is
judged against the midpoints of its confident neighbours in pairs (the
k-th before with the k-th after, k up to ``window``): constant motion,
however fast, sits on every midpoint, a one-frame spike sits ``gate_px``
off the median of them. Outliers get confidence 0 so the fits ignore them
and the regressor's prior holds the joint for that frame. Track ends
(no neighbour on one side) are not judged. Nothing is smoothed: the fit's
temporal term and the timeline's filter do that, and smoothing a real
stride here would lag it. Play 1: 1.8 % of confident keypoints rejected
(wrists 3.6 %), wrist frame-to-frame motion p99 44 -> 15 px.

TRIED AND REJECTED (2026-09-09, the runner on play 1's footage overlay):
judging by the NEAREST pair only, worst offender first (a real stride
swing is 3-5 px off its +-1 midpoint, 15-40 px off the +-2/+-3 ones, and
this rule threw a runner's ankles away on 6 of 27 frames). It kept the
ankles -- and let through the detector's left/right label flips, which
last two to four frames on a player running at the camera (10 % of his
frames): the arms flailed worse than before. The window-3 median catches
a multi-frame flip because the +-2 / +-3 pairs straddle it. A flip-aware
rule (swap the sides when the swapped labels continue the previous frames)
is the open item; a distance test alone cannot tell a flip from legs or
arms crossing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GATE_PX: float = 18.0
WINDOW: int = 3
MIN_CONF: float = 0.3


def outlier_mask(frames, xy, conf, *, gate_px: float = GATE_PX, window: int = WINDOW,
                 min_conf: float = MIN_CONF) -> np.ndarray:
    """Boolean mask of outliers among one joint's track (frames sorted)."""
    f = np.asarray(frames, int)
    xy = np.asarray(xy, float)
    ok = np.asarray(conf, float) >= min_conf
    out = np.zeros(len(f), bool)
    idx = np.flatnonzero(ok)
    if len(idx) < 3:
        return out
    pos = {int(i): k for k, i in enumerate(idx)}
    for i in idx:
        k = pos[int(i)]
        mids = []
        for step in range(1, window + 1):
            a, b = k - step, k + step
            if a < 0 or b >= len(idx):
                break
            ia, ib = idx[a], idx[b]
            if f[i] - f[ia] > window or f[ib] - f[i] > window:
                break
            mids.append(0.5 * (xy[ia] + xy[ib]))
        if len(mids) < 1:
            continue
        pred = np.median(np.stack(mids), axis=0)
        if np.linalg.norm(xy[i] - pred) > gate_px:
            out[i] = True
    return out


def reject_outliers(df: pd.DataFrame, *, gate_px: float = GATE_PX, window: int = WINDOW,
                    min_conf: float = MIN_CONF):
    """``(df, n_rejected)``: the keypoint table (columns cam, global_player_id,
    joint, frame, x, y, conf) with outliers' ``conf`` set to 0."""
    out = df.copy()
    conf = out["conf"].to_numpy(float).copy()
    rejected = 0
    loc = {ix: k for k, ix in enumerate(out.index)}
    order = out.sort_values(["cam", "global_player_id", "joint", "frame"])
    for _, g in order.groupby(["cam", "global_player_id", "joint"], sort=False):
        m = outlier_mask(g["frame"].to_numpy(int), g[["x", "y"]].to_numpy(float), g["conf"].to_numpy(float),
                         gate_px=gate_px, window=window, min_conf=min_conf)
        for ix in g.index.to_numpy()[m]:
            conf[loc[ix]] = 0.0
            rejected += 1
    out["conf"] = conf
    return out, rejected


# --------------------------------------------------------------------------- left/right flips
# The detector's left and right labels swap for a few frames at a time on a player running
# at the camera (play 1's runner: 11 % of his frames for the arms, 8 % for the legs, 2-4
# frames at a time). A distance test against the previous frame alone cannot tell a flip
# from limbs crossing (legs pass each other every stride); against the positions the last
# two frames PREDICT, a crossing continues its motion and a flip contradicts it.
LR_GROUPS = {"arms": ((5, 6), (7, 8), (9, 10)), "legs": ((13, 14), (15, 16))}
LR_MARGIN_FRAC: float = 0.3      # the swapped labels must beat the kept ones by this fraction ...
LR_MARGIN_PX: float = 8.0        # ... plus this many pixels, summed over the group
LR_MAX_GAP: int = 3              # frames; further back the prediction is not trusted


def _predict(prev_a, prev_b, fa, fb, f):
    """Linear prediction of a joint at frame f from its positions at fa (older) and fb (newer)."""
    if prev_a is None:
        return prev_b
    v = (prev_b - prev_a) / max(fb - fa, 1)
    return prev_b + v * (f - fb)


def lr_flip_mask(frames, xy_by_joint, conf_by_joint, pairs, *, min_conf: float = MIN_CONF,
                 margin_frac: float = LR_MARGIN_FRAC, margin_px: float = LR_MARGIN_PX, max_gap: int = LR_MAX_GAP):
    """Frames (indices into ``frames``) whose left/right labels of one group should be swapped.
    ``xy_by_joint[j]`` and ``conf_by_joint[j]`` are [T, 2] / [T] over the same frames. The
    decision at each frame uses the positions AFTER the earlier frames' swaps."""
    f = np.asarray(frames, int)
    T = len(f)
    joints = [j for pr in pairs for j in pr]
    pos = {j: np.asarray(xy_by_joint[j], float).copy() for j in joints}
    ok = {j: np.asarray(conf_by_joint[j], float) >= min_conf for j in joints}
    swapped = np.zeros(T, bool)
    last = {j: [] for j in joints}                    # per joint: [(index, xy)] of the last two confident frames
    for t in range(T):
        keep = swap = 0.0
        n = 0
        for a, b in pairs:
            if not (ok[a][t] and ok[b][t]):
                continue
            preds = {}
            for j in (a, b):
                hist = [(i, p) for i, p in last[j] if f[t] - f[i] <= max_gap]
                if not hist:
                    break
                if len(hist) == 2:
                    preds[j] = _predict(hist[0][1], hist[1][1], f[hist[0][0]], f[hist[1][0]], f[t])
                else:
                    preds[j] = hist[-1][1]
            if len(preds) < 2:
                continue
            keep += np.linalg.norm(pos[a][t] - preds[a]) + np.linalg.norm(pos[b][t] - preds[b])
            swap += np.linalg.norm(pos[b][t] - preds[a]) + np.linalg.norm(pos[a][t] - preds[b])
            n += 1
        if n >= 1 and swap < keep * (1.0 - margin_frac) - margin_px:
            swapped[t] = True
            for a, b in pairs:
                pa, pb = pos[a][t].copy(), pos[b][t].copy()
                pos[a][t], pos[b][t] = pb, pa
                ok[a][t], ok[b][t] = ok[b][t], ok[a][t]
        for j in joints:
            if ok[j][t]:
                last[j] = (last[j] + [(t, pos[j][t].copy())])[-2:]
    return swapped


def fix_lr_flips(df: pd.DataFrame, *, groups=LR_GROUPS, min_conf: float = MIN_CONF):
    """``(df, n_frames_swapped)``: the keypoint table with the left/right labels of a limb
    group swapped on the frames where the swapped labels continue the previous frames'
    motion and the given ones contradict it. Runs BEFORE reject_outliers."""
    out = df.copy()
    cols = ["x", "y", "conf"]
    n_swapped = 0
    for (cam, pid), g in out.groupby(["cam", "global_player_id"], sort=False):
        piv = {c: g.pivot_table(index="frame", columns="joint", values=c) for c in cols}
        frames = piv["x"].index.to_numpy(int)
        for name, pairs in groups.items():
            joints = [j for pr in pairs for j in pr]
            if any(j not in piv["x"].columns for j in joints):
                continue
            xy = {j: np.column_stack([piv["x"][j].to_numpy(float), piv["y"][j].to_numpy(float)]) for j in joints}
            conf = {j: np.nan_to_num(piv["conf"][j].to_numpy(float), nan=0.0) for j in joints}
            mask = lr_flip_mask(frames, xy, conf, pairs, min_conf=min_conf)
            if not mask.any():
                continue
            n_swapped += int(mask.sum())
            bad_frames = set(frames[mask].tolist())
            rows = g[g["frame"].isin(bad_frames) & g["joint"].isin(joints)]
            for a, b in pairs:
                ra = rows[rows["joint"] == a]
                rb = rows[rows["joint"] == b]
                common = ra.set_index("frame").index.intersection(rb.set_index("frame").index)
                # swap by index: both rows exist for the frames in ``common``
                ia_idx = ra[ra["frame"].isin(common)].sort_values("frame").index
                ib_idx = rb[rb["frame"].isin(common)].sort_values("frame").index
                va = out.loc[ia_idx, cols].to_numpy().copy()
                vb = out.loc[ib_idx, cols].to_numpy().copy()
                out.loc[ia_idx, cols] = vb
                out.loc[ib_idx, cols] = va
    return out, n_swapped
