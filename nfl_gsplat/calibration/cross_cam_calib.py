"""Calibrate the endzone camera from players shared with the calibrated
sideline camera.

Field markings fail on the endzone view (steep down-field perspective makes
yard-line identity ambiguous — measured labels 67-210 px wrong). Players don't:
the sideline camera turns each player's foot pixel into a field point
(X, Y, 0); paired with the same player's endzone foot pixel that is exactly the
(world, uv) correspondence the fixed-center joint solve consumes, and players
spread across the whole endzone image give the conditioning the markings lack.
Correspondence is bootstrapped geometrically (hypothesize an endzone camera,
project, Hungarian-match, solve, re-match) — no jersey OCR, no new deps.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.utils.geometry import project_points


def match_frame(world_xyz, endzone_uv, K, R, t, *, max_px):
    """Project world points through a hypothesized endzone camera and optimally
    assign them to endzone foot detections; keep assignments within max_px.
    Returns (matched_world (k,3), matched_uv (k,2))."""
    from scipy.optimize import linear_sum_assignment
    world_xyz = np.asarray(world_xyz, np.float64).reshape(-1, 3)
    endzone_uv = np.asarray(endzone_uv, np.float64).reshape(-1, 2)
    if len(world_xyz) == 0 or len(endzone_uv) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))
    pred = project_points(world_xyz, K, R, t)                  # (N,2), NaN if behind
    ok = np.isfinite(pred).all(axis=1)
    if not ok.any():
        return np.zeros((0, 3)), np.zeros((0, 2))
    idx = np.where(ok)[0]
    cost = np.linalg.norm(pred[idx][:, None, :] - endzone_uv[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= max_px
    wsel = idx[rows[keep]]
    return world_xyz[wsel], endzone_uv[cols[keep]]
