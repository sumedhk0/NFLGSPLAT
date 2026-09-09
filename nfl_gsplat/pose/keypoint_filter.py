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
