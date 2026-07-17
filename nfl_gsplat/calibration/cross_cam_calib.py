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


def sideline_field_by_frame(tracks_df, sideline_track, *, cam="sideline"):
    """{frame: world (N,3), Z=0} from projecting the sideline cam's foot points
    to the field plane. Drops NaN / off-field-magnitude rows."""
    from nfl_gsplat.calibration.cameras_io import CameraTrack  # noqa: F401 (type)
    from nfl_gsplat.tracking.cross_cam_reid import project_foot_points_to_field
    proj = project_foot_points_to_field(tracks_df[tracks_df["cam"] == cam],
                                        {cam: sideline_track})
    out: dict[int, np.ndarray] = {}
    for fr, grp in proj.groupby("frame"):
        xy = grp[["foot_x_m", "foot_y_m"]].to_numpy()
        ok = np.isfinite(xy).all(axis=1) & (np.abs(xy[:, 0]) <= 60) & (np.abs(xy[:, 1]) <= 30)
        if ok.any():
            w = np.column_stack([xy[ok], np.zeros(ok.sum())])
            out[int(fr)] = w.astype(np.float64)
    return out


def endzone_feet_by_frame(tracks_df, *, cam="endzone", deg, orig_wh):
    """{frame: uv (M,2)} of endzone foot points, mapped into the rotated
    working frame by rotate_uv."""
    from nfl_gsplat.calibration.view_rotation import rotate_uv
    ez = tracks_df[tracks_df["cam"] == cam]
    out: dict[int, np.ndarray] = {}
    for fr, grp in ez.groupby("frame"):
        uv = grp[["foot_u", "foot_v"]].to_numpy()
        rot = np.array([rotate_uv(u, v, deg, orig_wh) for (u, v) in uv], np.float64)
        if len(rot):
            out[int(fr)] = rot
    return out


def _look_at_R(C, target):
    fwd = np.asarray(target, float) - np.asarray(C, float)
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return None
    fwd = fwd / n
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        return None
    right = right / rn
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _hyp_cam(C, world_pts, image_size, f):
    """Per-frame hypothesized endzone (K,R,t): look at the frame's player
    centroid with focal f."""
    R = _look_at_R(C, np.asarray(world_pts, float).mean(axis=0))
    if R is None:
        return None
    t = -R @ np.asarray(C, np.float64)
    K = np.array([[f, 0, image_size[0] / 2.0], [0, f, image_size[1] / 2.0], [0, 0, 1.0]])
    return K, R, t


def _interp_cam_for(res, image_size):
    """cam_for(fidx) that falls back to predicting (r, f) from the nearest
    SOLVED neighbors (sharing the one fixed C) for frames solve_fixed_center
    didn't rescue last round, instead of skipping them outright.

    A frame can miss the AUDIT_DROP_PX cut on one round purely because its
    bootstrap corrs carried a stray mismatch (identity noise from _hyp_cam's
    approximate per-frame gaze, see solve_endzone_cross_camera) -- the shared
    C and that frame's true pose are still fine. Once neighboring frames ARE
    solved, their smoothly-varying (r, f) (the joint solve's own smoothness
    prior assumes exactly this) predicts the missing frame's camera far more
    accurately than the crude centroid look-at, letting a tight re-match gate
    recover it. Interpolates between bracketing solved frames; linearly
    EXTRAPOLATES from the nearest two solved frames past either end of the
    solved range (a hard skip there left an entire tail of frames -- e.g. a
    contiguous run past the last solved id -- permanently unrecoverable, since
    nothing downstream of an unsolved frame could ever seed it). A bad
    extrapolation is harmless: the caller still gates the resulting match on
    pixel distance, so a wrong guess just fails to match rather than
    poisoning anything."""
    import cv2
    solved_ids = [i for i, r in enumerate(res) if r is not None]
    if not solved_ids:
        return lambda fidx: None
    C = np.asarray(res[solved_ids[0]].pose.center_world(), np.float64)
    ids_arr = np.array(solved_ids, dtype=float)
    rstack = np.stack([cv2.Rodrigues(res[i].pose.R)[0].ravel() for i in solved_ids])
    fstack = np.array([res[i].intrinsics.fx for i in solved_ids])

    def _predict(fidx):
        if len(ids_arr) == 1 or ids_arr[0] <= fidx <= ids_arr[-1]:
            rv = np.array([np.interp(fidx, ids_arr, rstack[:, k]) for k in range(3)])
            return rv, float(np.interp(fidx, ids_arr, fstack))
        i0, i1 = (0, 1) if fidx < ids_arr[0] else (len(ids_arr) - 2, len(ids_arr) - 1)
        x0, x1 = ids_arr[i0], ids_arr[i1]
        w = (fidx - x0) / (x1 - x0)
        return rstack[i0] + w * (rstack[i1] - rstack[i0]), float(fstack[i0] + w * (fstack[i1] - fstack[i0]))

    def cam_for(fidx):
        r = res[fidx] if fidx < len(res) else None
        if r is not None:
            return r.intrinsics.K(), r.pose.R, r.pose.t
        rv, f = _predict(float(fidx))
        R = cv2.Rodrigues(rv)[0]
        t = -R @ C
        K = np.array([[f, 0, image_size[0] / 2.0], [0, f, image_size[1] / 2.0], [0, 0, 1.0]])
        return K, R, t
    return cam_for


def _build_corrs(field_by, feet_by, cam_for_frame, image_size, max_px):
    """cam_for_frame(fidx) -> (K,R,t) | None. Returns {fidx: (world(N,3), uv(N,2))}
    with >=4 matches, plus total matched count."""
    out, total = {}, 0
    for fidx, world in field_by.items():
        feet = feet_by.get(fidx)
        krt = cam_for_frame(fidx)
        if feet is None or krt is None:
            continue
        mw, mu = match_frame(world, feet, *krt, max_px=max_px)
        total += len(mw)
        if len(mw) >= 4:
            out[fidx] = (mw, mu)
    return out, total


def solve_endzone_cross_camera(field_by_frame, feet_by_frame, image_size, *,
                               init_C, view_deg=90, max_rounds=3,
                               focal_guesses=(1500.0, 2500.0, 3500.0),
                               match_px=(200.0, 30.0, 20.0)):
    """Bootstrap an endzone camera from a multi-start over candidate centers
    (+ focal sweep), match players, solve_fixed_center, re-match with the
    solved cameras, iterate. Returns a per-frame [CalibrationResult|None] in
    the rotated working frame."""
    from nfl_gsplat.calibration.joint_solve import (
        _GRID_EZ_X, _GRID_EZ_Y, _GRID_EZ_Z, solve_fixed_center)
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.errors import CalibrationError
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    T = (max(field_by_frame) if field_by_frame else 0) + 1

    # Bootstrap: a single init_C can be tens of meters off the truth, which
    # leaves projected players too far from endzone detections to gate-match
    # -- a sparse/mismatched bootstrap starves solve_fixed_center's rescue.
    # Multi-start over init_C + the behind-endzone grid (same grid the joint
    # solver trusts for this geometry), scoring each (center, focal) pair by
    # total gate-matched players and keeping the best. The round-0 gate is
    # loose (match_px[0], default 200px) because _hyp_cam's per-frame
    # look-at-the-players'-centroid is only an approximation of the camera's
    # true gaze -- even the TRUE center can carry 100+ px of systematic
    # reprojection error from that approximation alone (measured on the
    # synthetic: median 90-150px at the true C). A tight gate here starves
    # even a correct candidate while a wrong candidate can accidentally gate-
    # match a handful of players, so the round-0 gate has to be loose enough
    # to admit genuine (if systematically offset) matches; identity stays
    # correct because Hungarian assignment is driven by RELATIVE, not
    # absolute, pixel distance.
    def _score(C0, f):
        def cam_for(fidx, _C0=C0, _f=f):
            w = field_by_frame.get(fidx)
            return _hyp_cam(_C0, w, image_size, _f) if w is not None and len(w) else None
        return _build_corrs(field_by_frame, feet_by_frame, cam_for,
                            image_size, match_px[0])

    total_possible = sum(min(len(w), len(feet_by_frame.get(fidx, ())))
                         for fidx, w in field_by_frame.items())

    best = None   # (total, corrs, C0)
    for f in focal_guesses:
        corrs, total = _score(init_C, f)
        if best is None or total > best[0]:
            best = (total, corrs, np.asarray(init_C, np.float64))
    # A good measured init_C that already matches nearly every player short-
    # circuits the grid scan (cheap win, saves ~55x the bootstrap work).
    if total_possible <= 0 or best[0] < 0.9 * total_possible:
        candidate_centers = [np.array([X, Y, Z]) for X in _GRID_EZ_X
                             for Y in _GRID_EZ_Y for Z in _GRID_EZ_Z]
        for C0 in candidate_centers:
            for f in focal_guesses:
                corrs, total = _score(C0, f)
                if total > best[0]:
                    best = (total, corrs, C0)
    _total, corrs, C_boot = best
    if sum(len(w) for (w, _u) in corrs.values()) < 20 or len(corrs) < 5:
        raise CalibrationError(
            "endzone: cross-camera matching did not converge (too few player "
            "matches at bootstrap) — check frame sync and player detections.")

    # Anchor solve_fixed_center's OWN multi-start at the bootstrap winner.
    # solve_fixed_center's coarse plausibility grid (30-60m spacing) gets a
    # real ~50-iteration solve only at a "plausible anchor" candidate (see
    # init_from_results); everywhere else it only spends a cheap 12-iteration
    # scoring pass, which -- given the bootstrap corrs carry a few percent of
    # identity mismatches from the approximate gaze above -- is not enough to
    # relocate the camera precisely. Feeding the bootstrap winner in as an
    # anchor (>=3 identical entries, the threshold init_from_results requires)
    # gives it the deep solve it needs.
    anchor = CalibrationResult(
        intrinsics=CameraIntrinsics(2000.0, 2000.0, 0.0, 0.0, 1, 1),
        pose=CameraPose(R=np.eye(3), t=-np.asarray(C_boot, np.float64)),
        rms_px=0.0, num_correspondences=0, refined_with_ba=False)
    init_results = [anchor if i < 3 else None for i in range(T)]

    results = None
    for rnd in range(max_rounds):
        res, _mirrored = solve_fixed_center(
            corrs_by_frame=None, image_size=image_size,
            init_results=init_results, view_deg=view_deg, _frame_data_override=corrs)
        results = res
        init_results = res      # warm-start the next round with real per-frame cameras
        # re-match using the solved per-frame cameras (interpolating over any
        # frame this round's rescue dropped -- see _interp_cam_for), tightening
        # the gate as the camera estimate improves round to round.
        gate = match_px[min(rnd + 1, len(match_px) - 1)]
        cam_for = _interp_cam_for(res, image_size)
        new_corrs, total = _build_corrs(field_by_frame, feet_by_frame, cam_for,
                                        image_size, gate)
        if len(new_corrs) < 5:
            break
        corrs = new_corrs
    if results is None or all(r is None for r in results):
        raise CalibrationError(
            "endzone: cross-camera solve produced no usable frames.")
    return results
