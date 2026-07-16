# Cross-Camera Endzone Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the endzone camera from players shared with the already-calibrated sideline camera — project sideline foot-points to the field, match them to endzone foot detections through a hypothesize-project-match ICP loop, and feed the resulting `(world, uv)` pairs to the existing `solve_fixed_center`.

**Architecture:** New module `nfl_gsplat/calibration/cross_cam_calib.py`: (1) per-frame data prep from `tracks.parquet` + the sideline `CameraTrack`; (2) `match_frame` projects hypothesized-camera world points and Hungarian-matches to endzone feet; (3) `solve_endzone_cross_camera` bootstraps a camera, builds correspondences, calls `solve_fixed_center` via its `_frame_data_override` seam, re-matches, and iterates. Orchestration loads the sideline `cameras.npz`, solves endzone, and merges `endzone_*` back without touching sideline.

**Tech Stack:** numpy, scipy (`linear_sum_assignment`, `least_squares` inside the reused solver), pandas (parquet), existing calibration modules. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-15-cross-camera-endzone-calibration-design.md`

## Global Constraints

- Endzone camera model + solve are the EXISTING `solve_fixed_center` (fixed center, per-frame pan/zoom); this cycle only supplies cross-camera correspondences and the ICP loop around it. `view_deg=90` for the endzone (its working frame is the rotated one).
- Sideline camera is a trusted, FIXED input — read from `cameras.npz`, never modified. Endzone results MERGE into `cameras.npz` (`endzone_*`), preserving `sideline_*`.
- Correspondences are `(world_xyz (3,), (u,v))` where world is a player's field point `(X, Y, 0)` from the sideline projection and `uv` is that player's endzone foot pixel IN THE ROTATED WORKING FRAME; endzone results are de-rotated back to original pixels before assembly (reuse `view_rotation.derotate_result`).
- Fail loud (`SetupError`/`CalibrationError` with pointer): sideline camera absent/low-conf → SetupError; missing `tracks.parquet` → SetupError; ICP not converging / too few matches → `CalibrationError` naming `endzone` ("cross-camera matching did not converge"). No silent fallback.
- Videos assumed frame-synchronized (same index = same instant); the same-field-point acceptance check verifies it.
- `pytest -m "not gpu and not slow"` green; `ruff check nfl_gsplat tests scripts` clean. Synthetic tests build tracks DataFrames / arrays directly (no YOLO, no video, no parquet round-trip — local pyarrow is present but keep unit tests I/O-free). NEVER commit real NFL imagery.
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `match_frame` — project a hypothesized camera and Hungarian-match to endzone feet

**Files:**
- Create: `nfl_gsplat/calibration/cross_cam_calib.py`
- Test: `tests/test_cross_cam_calib.py` (new)

**Interfaces:**
- Consumes: `project_points(points_w, K, R, t)` (`nfl_gsplat/utils/geometry.py`), `scipy.optimize.linear_sum_assignment`.
- Produces (Task 3 relies on this): `match_frame(world_xyz, endzone_uv, K, R, t, *, max_px) -> tuple[np.ndarray, np.ndarray]` — returns `(matched_world (k,3), matched_uv (k,2))`: projects each world point through `(K,R,t)`, optimally assigns to the `endzone_uv` rows, keeps assignments within `max_px`. Empty `(0,3)/(0,2)` arrays when nothing matches.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cross_cam_calib.py
"""Cross-camera endzone calibration. Synthetic 2-camera scenes, no video/YOLO."""
import numpy as np
import pytest

from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points


def _look_at(C, target):
    fwd = target - C; fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _cam(C, target, f=2000.0, wh=(1920, 1080)):
    R = _look_at(np.asarray(C, float), np.asarray(target, float))
    t = -R @ np.asarray(C, float)
    K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
    return K, R, t


def test_match_frame_recovers_correct_pairs():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0], [-30.0, -8.0, 0.0], [-20.0, 2.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    uv = project_points(world, K, R, t)                 # exact endzone pixels
    # shuffle the detections; matcher must re-pair them to the right world pts
    perm = [2, 0, 1]
    mw, mu = match_frame(world, uv[perm], K, R, t, max_px=5.0)
    assert mw.shape == (3, 3) and mu.shape == (3, 2)
    # each returned world point's projection equals its matched uv
    proj = project_points(mw, K, R, t)
    assert np.abs(proj - mu).max() < 1e-6


def test_match_frame_distance_gate_drops_outlier():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0], [-30.0, -8.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    uv = project_points(world, K, R, t)
    uv = np.vstack([uv, [10.0, 10.0]])                  # a detection with no world match
    mw, mu = match_frame(world, uv, K, R, t, max_px=5.0)
    assert mw.shape == (2, 3)                            # the 2 real players only


def test_match_frame_far_projection_no_match():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    mw, mu = match_frame(world, np.array([[5.0, 5.0]]), K, R, t, max_px=5.0)
    assert mw.shape == (0, 3) and mu.shape == (0, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cross_cam_calib.py -v`
Expected: FAIL — `ModuleNotFoundError: nfl_gsplat.calibration.cross_cam_calib`

- [ ] **Step 3: Implement**

```python
# nfl_gsplat/calibration/cross_cam_calib.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_cam_calib.py -v`
Expected: 3 PASS

- [ ] **Step 5: Ruff + full suite + commit**

Run: `python -m ruff check nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py` — clean.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py
git commit -m "feat(calibration): match_frame projects+Hungarian-matches players to endzone feet

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Per-frame data prep — sideline field points + endzone feet

**Files:**
- Modify: `nfl_gsplat/calibration/cross_cam_calib.py` (add two functions)
- Test: `tests/test_cross_cam_calib.py` (append)

**Interfaces:**
- Consumes: `cross_cam_reid.project_foot_points_to_field(df, cameras)` (adds `foot_x_m, foot_y_m`); `view_rotation.rotate_uv(u, v, deg, orig_wh)`; a `CameraTrack` (`cameras_io`).
- Produces (Task 3 relies on):
  - `sideline_field_by_frame(tracks_df, sideline_track, *, cam="sideline") -> dict[int, np.ndarray]` — `{frame: world (N,3)}` with Z=0, from projecting that cam's foot points; drops NaN / off-field-magnitude (|X|≤60, |Y|≤30) rows.
  - `endzone_feet_by_frame(tracks_df, *, cam="endzone", deg, orig_wh) -> dict[int, np.ndarray]` — `{frame: uv (M,2)}`, each `(foot_u, foot_v)` mapped by `rotate_uv(...,deg,orig_wh)` into the rotated working frame.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _tracks_df(rows):
    import pandas as pd
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    df = pd.DataFrame(rows)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c not in ("cam",) else ""
    return df


def _sideline_track(C, target, n_frames=3, f=2600.0, wh=(1920, 1080)):
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    R = _look_at(np.asarray(C, float), np.asarray(target, float))
    t = -R @ np.asarray(C, float)
    K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
    return CameraTrack(K=np.repeat(K[None], n_frames, 0), R=np.repeat(R[None], n_frames, 0),
                       t=np.repeat(t[None], n_frames, 0), conf=np.ones(n_frames),
                       width=wh[0], height=wh[1])


def test_sideline_field_by_frame_projects_feet_to_z0():
    from nfl_gsplat.calibration.cross_cam_calib import sideline_field_by_frame
    sl = _sideline_track([-3.6, 80.0, 36.0], [0.0, 0.0, 0.0])
    # a player truly at field (X=-10, Y=4, 0): find its sideline foot pixel
    K, R, t = sl.at(0)[0].K(), sl.at(0)[1].R, sl.at(0)[1].t
    uv = project_points(np.array([[-10.0, 4.0, 0.0]]), K, R, t)[0]
    df = _tracks_df([{"frame": 0, "cam": "sideline", "foot_u": uv[0], "foot_v": uv[1],
                      "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1}])
    fb = sideline_field_by_frame(df, sl)
    assert 0 in fb
    assert np.allclose(fb[0][0, :2], [-10.0, 4.0], atol=0.2) and fb[0][0, 2] == 0.0


def test_endzone_feet_by_frame_rotates():
    from nfl_gsplat.calibration.cross_cam_calib import endzone_feet_by_frame
    from nfl_gsplat.calibration.view_rotation import rotate_uv
    df = _tracks_df([{"frame": 0, "cam": "endzone", "foot_u": 100.0, "foot_v": 200.0,
                      "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1},
                     {"frame": 0, "cam": "sideline", "foot_u": 5.0, "foot_v": 6.0,
                      "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1}])
    fb = endzone_feet_by_frame(df, deg=90, orig_wh=(1920, 1080))
    assert list(fb) == [0]
    assert np.allclose(fb[0][0], rotate_uv(100.0, 200.0, 90, (1920, 1080)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cross_cam_calib.py -v -k "field_by_frame or feet_by_frame"`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement** (append to cross_cam_calib.py)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_cam_calib.py -v`
Expected: all PASS

- [ ] **Step 5: Ruff + full suite + commit**

Run: `python -m ruff check nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py` — clean.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py
git commit -m "feat(calibration): per-frame sideline field points + rotated endzone feet

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `solve_endzone_cross_camera` — the bootstrap + ICP loop

**Files:**
- Modify: `nfl_gsplat/calibration/cross_cam_calib.py` (add the solve + helpers)
- Test: `tests/test_cross_cam_calib.py` (append)

**Interfaces:**
- Consumes: Task 1 `match_frame`; `solve_fixed_center(corrs_by_frame, image_size, *, init_results, view_deg, _frame_data_override)` returning `(results, mirrored)` (`joint_solve.py`); `CalibrationResult` fields `intrinsics.K()`, `pose.R`, `pose.t`.
- Produces (Task 4 relies on):
  - `solve_endzone_cross_camera(field_by_frame, feet_by_frame, image_size, *, init_C, view_deg=90, max_rounds=3, focal_guesses=(1500.0, 2500.0, 3500.0), match_px=(60.0, 30.0, 20.0)) -> list`
    — returns a `[CalibrationResult|None]` list aligned to `max(frame index)+1`, the endzone camera per frame (still in the rotated working frame; the caller de-rotates). Raises `CalibrationError` if the bootstrap finds too few matches or the ICP does not converge.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_solve_endzone_cross_camera_recovers_synthetic():
    # Ground truth: a fixed endzone camera + ~20 players moving over 40 frames.
    # Sideline gives the players' true field points; endzone sees their feet.
    # The solve must recover the endzone camera from a grid/measured init that
    # is NOT the truth.
    from nfl_gsplat.calibration.cross_cam_calib import solve_endzone_cross_camera
    from nfl_gsplat.calibration.view_rotation import rotated_wh
    rng = np.random.default_rng(0)
    ow = (1920, 1080)
    ez_wh = rotated_wh(90, ow)                      # endzone works in rotated frame
    C_true = np.array([-110.0, -8.0, 45.0])
    n_frames, n_players = 40, 20
    field_by, feet_by = {}, {}
    f_by = {}
    for i in range(n_frames):
        # players scattered on the field, drifting frame to frame
        base = rng.uniform([-45, -22], [10, 22], size=(n_players, 2))
        world = np.column_stack([base, np.zeros(n_players)])
        field_by[i] = world
        tx = -20.0 + 30.0 * i / (n_frames - 1)
        R = _look_at(C_true, np.array([tx, 0.0, 0.0]))
        t = -R @ C_true
        f = 2400.0 + 400.0 * i / (n_frames - 1)
        K = CameraIntrinsics(f, f, ez_wh[0] / 2, ez_wh[1] / 2, ez_wh[0], ez_wh[1]).K()
        uv = project_points(world, K, R, t)
        ok = np.isfinite(uv).all(axis=1)
        feet_by[i] = uv[ok] + rng.normal(0, 0.5, uv[ok].shape)
        f_by[i] = f
    results = solve_endzone_cross_camera(
        field_by, feet_by, ez_wh, init_C=np.array([-90.0, 0.0, 35.0]), view_deg=90)
    solved = [r for r in results if r is not None]
    assert len(solved) >= 30
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_true) < 1.0
    for r in solved:
        assert np.allclose(r.pose.center_world(), C_rec)     # one fixed center


def test_solve_endzone_cross_camera_fails_loud_no_matches():
    from nfl_gsplat.calibration.cross_cam_calib import solve_endzone_cross_camera
    from nfl_gsplat.errors import CalibrationError
    # players at (X,Y) that no plausible endzone camera can align to random feet
    field_by = {i: np.array([[-30.0, 0.0, 0.0]]) for i in range(15)}
    feet_by = {i: np.array([[9999.0, 9999.0]]) for i in range(15)}
    with pytest.raises(CalibrationError, match="cross-camera"):
        solve_endzone_cross_camera(field_by, feet_by, (1080, 1920),
                                   init_C=np.array([-90.0, 0.0, 35.0]), view_deg=90)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_cam_calib.py -v -k solve_endzone`
Expected: FAIL — function not defined.

- [ ] **Step 3: Implement** (append to cross_cam_calib.py)

```python
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
                               match_px=(60.0, 30.0, 20.0)):
    """Bootstrap an endzone camera from init_C (+ focal sweep), match players,
    solve_fixed_center, re-match with the solved cameras, iterate. Returns a
    per-frame [CalibrationResult|None] in the rotated working frame."""
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    from nfl_gsplat.errors import CalibrationError

    T = (max(field_by_frame) if field_by_frame else 0) + 1

    # Bootstrap: pick the focal guess that yields the most matches under init_C.
    best = None
    for f in focal_guesses:
        def cam_for(fidx, _f=f):
            w = field_by_frame.get(fidx)
            return _hyp_cam(init_C, w, image_size, _f) if w is not None and len(w) else None
        corrs, total = _build_corrs(field_by_frame, feet_by_frame, cam_for,
                                    image_size, match_px[0])
        if best is None or total > best[0]:
            best = (total, corrs)
    corrs = best[1]
    if sum(len(w) for (w, _u) in corrs.values()) < 20 or len(corrs) < 5:
        raise CalibrationError(
            "endzone: cross-camera matching did not converge (too few player "
            "matches at bootstrap) — check frame sync and player detections.")

    results = None
    for rnd in range(max_rounds):
        res, _mirrored = solve_fixed_center(
            corrs_by_frame=None, image_size=image_size,
            init_results=[None] * T, view_deg=view_deg, _frame_data_override=corrs)
        results = res
        # re-match using the solved per-frame cameras, tightening the gate
        gate = match_px[min(rnd + 1, len(match_px) - 1)]

        def cam_for(fidx, _res=res):
            r = _res[fidx] if fidx < len(_res) else None
            return None if r is None else (r.intrinsics.K(), r.pose.R, r.pose.t)
        new_corrs, total = _build_corrs(field_by_frame, feet_by_frame, cam_for,
                                        image_size, gate)
        if len(new_corrs) < 5:
            break
        corrs = new_corrs
    if results is None or all(r is None for r in results):
        raise CalibrationError(
            "endzone: cross-camera solve produced no usable frames.")
    return results
```

NOTE: `solve_fixed_center` runs its own multi-start over the candidate grid
(which includes the behind-endzone positions) each round; the ICP only improves
the CORRESPONDENCES fed to it. If the synthetic recovery test is slow (> ~30 s),
reduce `n_frames` in the test to 30 — the point is camera recovery, not scale.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_cam_calib.py -v`
Expected: all PASS (recovery within 1 m; fail-loud raises).

- [ ] **Step 5: Ruff + full suite + commit**

Run: `python -m ruff check nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py` — clean.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py
git commit -m "feat(calibration): solve_endzone_cross_camera bootstrap+ICP over solve_fixed_center

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Orchestration + CLI — build endzone from sideline, merge into cameras.npz

**Files:**
- Modify: `nfl_gsplat/calibration/run_autocalib.py` (add `build_endzone_from_sideline`)
- Modify: `scripts/02_autocalibrate.py` (add `cross-endzone` mode)
- Test: `tests/test_run_autocalib.py` (append)

**Interfaces:**
- Consumes: Task 2/3 functions; `load_camera_track(path)` + `write_camera_track(path, tracks, fps)` + `CameraTrack` (`cameras_io`); `assemble_track_from_results` + `derotate_result` + `_camera_rotation` (already in run_autocalib); `boxes`/tracks via `pandas.read_parquet`.
- Produces:
  - `build_endzone_from_sideline(*, play_dir, tracks_path, cameras_npz, endzone_video, fps, init_C=(-111.0,-20.9,63.8), sideline_cam="sideline", endzone_cam="endzone", rotations=None) -> Path`
    — loads the sideline `CameraTrack` from `cameras_npz` (SetupError if absent/low-conf), reads `tracks_path`, builds field points + rotated endzone feet, runs `solve_endzone_cross_camera`, de-rotates results, assembles an endzone `CameraTrack`, MERGES it into `cameras_npz` (preserving sideline), returns the path.
  - CLI: `python scripts/02_autocalibrate.py --play-dir P --mode cross-endzone` (reads `<play_dir>/cameras.npz` + `<play_dir>/tracks.parquet`).

- [ ] **Step 1: Write the failing test** (append to tests/test_run_autocalib.py)

```python
def test_build_endzone_from_sideline_merges(monkeypatch, tmp_path):
    # sideline camera present in cameras.npz; endzone solved from cross-camera
    # stub; result merged as endzone_* without dropping sideline_*.
    import numpy as np
    import pandas as pd
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.cameras_io import CameraTrack, load_camera_track, write_camera_track

    T = 3
    sl = CameraTrack(K=np.repeat(np.eye(3)[None], T, 0), R=np.repeat(np.eye(3)[None], T, 0),
                     t=np.zeros((T, 3)), conf=np.ones(T), width=1920, height=1080)
    npz = tmp_path / "cameras.npz"
    write_camera_track(npz, {"sideline": sl}, fps=30.0)

    tracks = tmp_path / "tracks.parquet"
    pd.DataFrame({"frame": [0], "cam": ["endzone"], "foot_u": [1.0], "foot_v": [2.0],
                  "bbox_x1": [0.0], "bbox_y1": [0.0], "bbox_x2": [1.0], "bbox_y2": [1.0]}
                 ).to_parquet(tracks, index=False)

    # stub the heavy solve: return T frames of a fixed endzone camera
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
    def fake_solve(field_by, feet_by, image_size, *, init_C, view_deg=90, **kw):
        R = np.eye(3); t = np.array([100.0, 0.0, 40.0])
        return [CalibrationResult(intrinsics=CameraIntrinsics(2000, 2000, 540, 960, 1080, 1920),
                                  pose=CameraPose(R=R, t=t), rms_px=1.0,
                                  num_correspondences=8, refined_with_ba=True)] * T
    monkeypatch.setattr(ra, "solve_endzone_cross_camera", fake_solve, raising=False)
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda v: type("M", (), {"num_frames": T, "width": 1920, "height": 1080})())

    out = ra.build_endzone_from_sideline(
        play_dir=str(tmp_path), tracks_path=str(tracks), cameras_npz=str(npz),
        endzone_video="e.mp4", fps=30.0)
    cams = load_camera_track(out)
    assert "sideline" in cams and "endzone" in cams          # merged, sideline kept
    assert cams["endzone"].num_frames == T


def test_build_endzone_missing_sideline_fails_loud(tmp_path):
    from nfl_gsplat.calibration.run_autocalib import build_endzone_from_sideline
    from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
    from nfl_gsplat.errors import SetupError
    import numpy as np
    npz = tmp_path / "cameras.npz"
    write_camera_track(npz, {"endzone": CameraTrack(
        K=np.repeat(np.eye(3)[None], 2, 0), R=np.repeat(np.eye(3)[None], 2, 0),
        t=np.zeros((2, 3)), conf=np.ones(2), width=1920, height=1080)}, fps=30.0)
    with pytest.raises(SetupError, match="sideline"):
        build_endzone_from_sideline(play_dir=str(tmp_path), tracks_path=str(tmp_path / "t.parquet"),
                                    cameras_npz=str(npz), endzone_video="e.mp4", fps=30.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_autocalib.py -v -k endzone_from_sideline`
Expected: FAIL — `build_endzone_from_sideline` not defined.

- [ ] **Step 3: Implement** — in `run_autocalib.py`, module-level import:

```python
from nfl_gsplat.calibration.cross_cam_calib import (
    endzone_feet_by_frame, sideline_field_by_frame, solve_endzone_cross_camera,
)
```

and the function:

```python
def build_endzone_from_sideline(*, play_dir, tracks_path, cameras_npz, endzone_video,
                                fps, init_C=(-111.0, -20.9, 63.8),
                                sideline_cam="sideline", endzone_cam="endzone",
                                rotations=None):
    """Calibrate the endzone camera from players shared with the calibrated
    sideline camera; merge endzone_* into cameras.npz (keep sideline)."""
    import numpy as np
    import pandas as pd

    from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta

    cams = load_camera_track(cameras_npz)
    sl = cams.get(sideline_cam)
    if sl is None or float(np.mean(sl.conf > 0)) < 0.2:
        raise SetupError(
            f"no calibrated {sideline_cam!r} camera in {cameras_npz} — run the "
            "sideline pretrained calibration first (02_autocalibrate --mode pretrained).")
    if not Path(tracks_path).exists():
        raise SetupError(
            f"player tracks not found: {tracks_path} — run scripts/03b_detect_players.py first.")
    df = pd.read_parquet(tracks_path)

    meta = ffprobe_meta(endzone_video)
    deg = _camera_rotation(endzone_cam, rotations)
    orig_wh = (meta.width, meta.height)
    from nfl_gsplat.calibration.view_rotation import rotated_wh
    work_wh = rotated_wh(deg, orig_wh)

    field_by = sideline_field_by_frame(df, sl, cam=sideline_cam)
    feet_by = endzone_feet_by_frame(df, cam=endzone_cam, deg=deg, orig_wh=orig_wh)
    results = solve_endzone_cross_camera(field_by, feet_by, work_wh,
                                         init_C=np.asarray(init_C, float), view_deg=deg)
    results = [derotate_result(r, deg, orig_wh) if r is not None else None for r in results]
    # pad/trim to the endzone frame count for assembly
    T = meta.num_frames
    results = (results + [None] * T)[:T]
    cams[endzone_cam] = assemble_track_from_results(results, width=meta.width,
                                                    height=meta.height, max_gap=30)
    return write_camera_track(Path(cameras_npz), cams, fps=fps)
```

- [ ] **Step 4: Wire the CLI** (`scripts/02_autocalibrate.py`) — add `cross_endzone = "cross-endzone"` to `CalibMode`; branch:

```python
    if mode is CalibMode.cross_endzone:
        from nfl_gsplat.calibration.run_autocalib import build_endzone_from_sideline
        rotations = {}
        for pair in (p.strip() for p in rotate.split(",") if p.strip()):
            cam_name, _, deg_s = pair.partition("=")
            if deg_s not in ("0", "90", "180", "270"):
                raise typer.BadParameter(f"--rotate {pair!r}: rotation must be 0/90/180/270.")
            rotations[cam_name.strip()] = int(deg_s)
        out = build_endzone_from_sideline(
            play_dir=pd.dir, tracks_path=pd.dir / "tracks.parquet",
            cameras_npz=pd.dir / "cameras.npz", endzone_video=pd.video("endzone"),
            fps=meta.fps, rotations=rotations or None)
    elif mode is CalibMode.pretrained:
        ...  # unchanged
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_run_autocalib.py tests/test_cross_cam_calib.py -v`
Expected: all PASS.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.
Run: `python -m ruff check nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py tests/test_run_autocalib.py
git commit -m "feat(calibration): --mode cross-endzone builds endzone from sideline players

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Real-footage acceptance (manual gate)

**Files:** none. Prereqs present: sideline `cameras.npz`, `tracks.parquet` (both from earlier cycles), local env (pyarrow/torch/ultralytics) already installed.

- [ ] **Step 1: Run cross-endzone calibration**

```
python scripts/02_autocalibrate.py --play-dir data\2025\week_04\SEA_at_AZ\play_001 --mode cross-endzone
```
Expected: merges `endzone_*` into `cameras.npz`, or fails loud "cross-camera matching did not converge" (record the round counts / matched totals if so).

- [ ] **Step 2: Verify the endzone camera is plausible**

```
python -c "import numpy as np; d=np.load(r'data\2025\week_04\SEA_at_AZ\play_001\cameras.npz'); \
R=d['endzone_R']; t=d['endzone_t']; c=d['endzone_conf']; \
C=np.einsum('nij,ni->nj', R, -t); ok=c>0; \
print('endzone kept', int(ok.sum()), 'center', C[ok].mean(0).round(1))"
```
Assert center |X| 50–150, |Y| ≤ 40, Z 10–80. Confirm `sideline_*` unchanged
(center ≈ (−3.6, 80.5, 35.9)).

- [ ] **Step 3: The same-field-point proof (the real acceptance)**

For a few sampled frames, take a sideline player's field point `(X,Y,0)` and its
matched endzone detection; project the point through the de-rotated endzone
camera and confirm it lands on that endzone foot pixel (a few px), AND that the
endzone detection back-projected to Z=0 lands within a few meters of the
sideline player's `(X,Y)`. Adapt from `scripts/diag_pretrained.py`'s overlay
pattern; write overlays OUTSIDE the repo. This proves the two cameras agree on
the same physical players — the definition of a correct cross-camera solve.

- [ ] **Step 4: Record the outcome** in the branch progress notes and, on merge,
  update the endzone memory: cross-camera result (kept frames, center) or the
  measured non-convergence and its cause.
