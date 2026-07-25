# Jersey-Identity Endzone Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the endzone camera from jersey-identity player correspondences (multi-play, shared center) instead of the failed geometric foot-point matching.

**Architecture:** Reuse the existing `nfl_gsplat/identity` subsystem to give every track a `player_uid` from jersey+team (geometry-free); join sideline↔endzone on `player_uid`; feed the correct correspondences (sideline foot→field point ↔ endzone foot pixel) into the existing `solve_fixed_center`, aggregating multiple first-half plays under one shared center C. Work in **native endzone pixels** (`view_deg=0`, no rotation, no roll).

**Tech Stack:** Python, NumPy, SciPy, pandas, pytest. Reuses `joint_solve.solve_fixed_center`, `identity.registry.resolve_tracks`, `identity.team_color`, `tracking.jersey_ocr`, `tracking.detect_track.detect_and_track`, `cross_cam_reid.project_foot_points_to_field`. No new runtime deps.

## Global Constraints

- Work in **native endzone pixel coordinates**: `view_deg=0`, no `view_rotation`, no roll. (The 90° rotation only served the field-marking detector, which this bypasses.)
- Player world points are arbitrary `(X, Y, 0)`, NOT named `NFL_LANDMARKS`. Therefore the solve MUST use `solve_fixed_center(corrs_by_frame=None, _frame_data_override=frame_data, ...)` where `frame_data = {frame: (world (N,3) float64, uv (N,2) float64)}` — never the named-landmark `corrs_by_frame` path.
- Cross-camera identity is geometry-free: `identity.registry.resolve_tracks(df, candidates, cfg, id_col="track_id")` per camera, joined on `player_uid`. Never use `cross_cam_reid.reid_pipeline` (it needs both cameras calibrated).
- `OTHER_UID = ""` (empty string) and `REFEREE_UID = "__referee__"` are non-player; exclude them from correspondences.
- Fail loud with `CalibrationError`/`SetupError` + an actionable pointer. No silent fallback that changes numerical results.
- `solve_fixed_center` changes must be **default-preserving**: existing callers (sideline path) get byte-identical behavior when the new params are unset.
- GPU/PaddleOCR precompute runs in the `nfl_smplx` conda env on PACE Phoenix `embers` partition (account `paceship-pso`). Unit tests never import PaddleOCR — they mock OCR / pass tracks with `jersey_number_ocr` pre-filled.
- Never commit real NFL video/frames; diagnostic images go to `C:\Users\sumedh\diag\` or scratchpad, OUTSIDE the repo.
- `pytest -m "not gpu and not slow"` green; ruff clean. New commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Branch: `jersey-identity-endzone` (already created off main; the spec is committed there as `2578835`).

---

## File Structure

- `nfl_gsplat/calibration/joint_solve.py` — MODIFY: add `center_bounds` + `audit_drop_px` params to `solve_fixed_center` (default-preserving).
- `nfl_gsplat/calibration/endzone_identity.py` — NEW: build identity correspondences (join on `player_uid`).
- `nfl_gsplat/calibration/endzone_multiplay.py` — NEW: `EndzonePrior` + `solve_endzone_identity` (multi-play shared-C driver).
- `nfl_gsplat/calibration/identity_precompute.py` — NEW: pure identity-column assignment (global team split + `player_uid`), reused by the script.
- `scripts/03c_identity_tracks.py` — NEW: orchestrate detect+track + jersey OCR + identity columns → `tracks.parquet` (PACE).
- `nfl_gsplat/calibration/run_autocalib.py` — MODIFY: add `build_endzone_identity_from_plays(...)`.
- `scripts/02_autocalibrate.py` — MODIFY: add `--mode identity-endzone` (multi-play + prior).
- Tests: `tests/test_endzone_identity.py`, `tests/test_endzone_multiplay.py`, `tests/test_identity_precompute.py`, and additions to `tests/test_joint_solve.py` (or the existing joint-solve test file).

---

## Task 1: `solve_fixed_center` gains `center_bounds` + `audit_drop_px`

**Files:**
- Modify: `nfl_gsplat/calibration/joint_solve.py` (`solve_fixed_center` at line 462; `_candidate_centers` at line 342; module const `AUDIT_DROP_PX` at line 45).
- Test: `tests/test_joint_solve.py` (create if absent).

**Interfaces:**
- Consumes: existing `solve_fixed_center(corrs_by_frame, image_size, *, init_results, max_rounds=2, _frame_data_override=None, view_deg=0)`; `_candidate_centers(init_results)`.
- Produces: `solve_fixed_center(..., center_bounds=None, audit_drop_px=None)` where `center_bounds` is `((xmin,xmax),(ymin,ymax),(zmin,zmax))` or `None`, and `audit_drop_px` is a float or `None` (falls back to `AUDIT_DROP_PX`). New helper `_filter_centers(cands, center_bounds) -> list[np.ndarray]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_joint_solve.py`:

```python
import numpy as np
from nfl_gsplat.calibration import joint_solve as js


def test_filter_centers_keeps_only_in_box():
    cands = [np.array([-115.0, 0.0, 25.0]), np.array([120.0, 20.0, 35.0]),
             np.array([-90.0, 0.0, 35.0])]
    box = ((-150.0, -50.0), (-10.0, 10.0), (10.0, 80.0))
    kept = js._filter_centers(cands, box)
    got = {tuple(c) for c in kept}
    assert (-115.0, 0.0, 25.0) in got            # in box
    assert (-90.0, 0.0, 35.0) in got             # in box
    assert (120.0, 20.0, 35.0) not in got        # reflected, excluded


def test_filter_centers_none_is_identity():
    cands = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    assert js._filter_centers(cands, None) == cands
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_joint_solve.py::test_filter_centers_keeps_only_in_box -v`
Expected: FAIL — `module 'joint_solve' has no attribute '_filter_centers'`.

- [ ] **Step 3: Implement `_filter_centers` and thread the params**

Add near `_candidate_centers` in `joint_solve.py`:

```python
def _filter_centers(cands, center_bounds):
    """Keep only candidate centers inside center_bounds
    ((xmin,xmax),(ymin,ymax),(zmin,zmax)); None -> unchanged. Always keeps at
    least the first candidate (the anchor) so the solve never starves."""
    if center_bounds is None:
        return cands
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = center_bounds
    kept = [c for c in cands
            if xlo <= c[0] <= xhi and ylo <= c[1] <= yhi and zlo <= c[2] <= zhi]
    return kept if kept else cands[:1]
```

Change `solve_fixed_center`'s signature and body. Signature:

```python
def solve_fixed_center(corrs_by_frame, image_size, *, init_results,
                       max_rounds: int = 2, _frame_data_override=None,
                       view_deg: int = 0, center_bounds=None, audit_drop_px=None):
```

Right after `candidates = _candidate_centers(init_results)` (line 488) insert:

```python
    candidates = _filter_centers(candidates, center_bounds)
```

Replace the two `AUDIT_DROP_PX` usages inside this function (lines 504 and 515, the `m <= AUDIT_DROP_PX` / `med <= AUDIT_DROP_PX` gates, and the message on line 519) with a local:

```python
    drop_px = AUDIT_DROP_PX if audit_drop_px is None else audit_drop_px
```
and use `drop_px` in the two comparisons and the f-string message. Do NOT change the module constant.

- [ ] **Step 4: Run the filter test to verify it passes**

Run: `python -m pytest tests/test_joint_solve.py -v`
Expected: both `_filter_centers` tests PASS.

- [ ] **Step 5: Add a default-preserving regression test**

Add to `tests/test_joint_solve.py` a test that an existing sideline-style solve is unchanged. Reuse the synthetic solve fixture already used elsewhere if present; otherwise assert the no-arg path still runs on a tiny synthetic named-landmark set. Minimal version:

```python
def test_solve_fixed_center_defaults_unchanged(monkeypatch):
    # center_bounds=None and audit_drop_px=None must not alter candidate set
    seen = {}
    real = js._filter_centers
    def spy(cands, cb):
        seen["cb"] = cb
        return real(cands, cb)
    monkeypatch.setattr(js, "_filter_centers", spy)
    # a trivially-failing solve is fine; we only assert the bounds were None
    try:
        js.solve_fixed_center({}, (1920, 1080), init_results=[None] * 3)
    except Exception:
        pass
    assert seen.get("cb", "unset") is None
```

- [ ] **Step 6: Run the whole joint-solve suite + ruff**

Run: `python -m pytest tests/test_joint_solve.py -v` and `python -m pytest -m "not gpu and not slow" -q -k joint`
Expected: PASS. Run: `python -m ruff check nfl_gsplat/calibration/joint_solve.py` → clean.

- [ ] **Step 7: Commit**

```bash
git add nfl_gsplat/calibration/joint_solve.py tests/test_joint_solve.py
git commit -m "feat(calibration): solve_fixed_center center_bounds + audit_drop_px (default-preserving)"
```

---

## Task 2: `endzone_identity.py` — identity correspondences

**Files:**
- Create: `nfl_gsplat/calibration/endzone_identity.py`
- Test: `tests/test_endzone_identity.py`

**Interfaces:**
- Consumes: `cross_cam_reid.project_foot_points_to_field(df, {cam: CameraTrack})` (adds `foot_x_m,foot_y_m`); `cameras_io.CameraTrack`; columns `frame, cam, track_id, player_uid, foot_u, foot_v, conf`.
- Produces:
  - `field_positions_by_uid(tracks_df, sideline_track, *, cam="sideline", smooth_window=15) -> {player_uid: {frame: (X, Y)}}`
  - `endzone_pixels_by_uid(tracks_df, *, cam="endzone") -> {player_uid: {frame: (u, v)}}`
  - `identity_correspondences(tracks_df, sideline_track, *, sideline_cam="sideline", endzone_cam="endzone", smooth_window=15) -> {frame: (world (N,3) float64, uv (N,2) float64)}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_endzone_identity.py`:

```python
import numpy as np
import pandas as pd
from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points


def _sideline_track(C, target, n_frames, f=6000.0, wh=(1920, 1080)):
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    fwd = np.asarray(target, float) - np.asarray(C, float); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); t = -R @ np.asarray(C, float)
    K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
    return CameraTrack(K=np.repeat(K[None], n_frames, 0), R=np.repeat(R[None], n_frames, 0),
                       t=np.repeat(t[None], n_frames, 0), conf=np.ones(n_frames),
                       width=wh[0], height=wh[1])


def _rows(recs):
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    df = pd.DataFrame(recs)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c not in ("cam",) else ""
    if "player_uid" not in df.columns:
        df["player_uid"] = ""
    return df


def test_identity_correspondences_joins_on_uid():
    sl = _sideline_track([-3.6, 80.0, 36.0], [0, 0, 0], n_frames=2)
    K, R, t = sl.at(0)[0].K(), sl.at(0)[1].R, sl.at(0)[1].t
    # two players at known field points; get their sideline foot pixels
    p_a = np.array([[-10.0, 4.0, 0.0]]); p_b = np.array([[-15.0, -6.0, 0.0]])
    ua = project_points(p_a, K, R, t)[0]; ub = project_points(p_b, K, R, t)[0]
    recs = [
        {"frame": 0, "cam": "sideline", "player_uid": "2025_A_58", "foot_u": ua[0], "foot_v": ua[1], "conf": 1},
        {"frame": 0, "cam": "sideline", "player_uid": "2025_A_20", "foot_u": ub[0], "foot_v": ub[1], "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "2025_A_58", "foot_u": 800.0, "foot_v": 500.0, "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "2025_A_20", "foot_u": 900.0, "foot_v": 520.0, "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "",          "foot_u": 10.0,  "foot_v": 10.0,  "conf": 1},  # OTHER: excluded
    ]
    from nfl_gsplat.calibration.endzone_identity import identity_correspondences
    corr = identity_correspondences(_rows(recs), sl, smooth_window=1)
    assert set(corr) == {0}
    world, uv = corr[0]
    assert world.shape == (2, 3) and uv.shape == (2, 2)
    # #58's field point recovered near (-10, 4), matched to its endzone pixel (800,500)
    i58 = np.argmin(np.linalg.norm(uv - [800.0, 500.0], axis=1))
    assert np.allclose(world[i58, :2], [-10.0, 4.0], atol=0.3) and world[i58, 2] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_endzone_identity.py -v`
Expected: FAIL — module `endzone_identity` not found.

- [ ] **Step 3: Implement `endzone_identity.py`**

```python
"""Cross-camera endzone correspondences from jersey IDENTITY (player_uid).

The sideline camera (calibrated) turns each player's foot pixel into a field
point (X, Y, 0); the endzone camera sees the same player's foot pixel. Pairing
is by player_uid (from the jersey/team identity stack) -- not geometry -- so the
correspondences are correct regardless of the unknown endzone camera. Native
endzone pixels (no rotation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import CameraTrack  # noqa: F401 (type)
from nfl_gsplat.identity.registry import OTHER_UID, REFEREE_UID


def _is_player(uid) -> bool:
    return isinstance(uid, str) and uid not in (OTHER_UID, REFEREE_UID)


def field_positions_by_uid(tracks_df, sideline_track, *, cam="sideline", smooth_window=15):
    """{player_uid: {frame: (X, Y)}} from projecting the sideline cam's feet to
    Z=0, temporally smoothed per uid with a centered rolling median."""
    from nfl_gsplat.tracking.cross_cam_reid import project_foot_points_to_field
    sl = tracks_df[tracks_df["cam"] == cam]
    proj = project_foot_points_to_field(sl, {cam: sideline_track})
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for uid, grp in proj.groupby("player_uid"):
        if not _is_player(uid):
            continue
        g = grp.sort_values("frame")
        xy = g[["foot_x_m", "foot_y_m"]].to_numpy(float)
        ok = np.isfinite(xy).all(axis=1)
        g = g[ok]; xy = xy[ok]
        if not len(g):
            continue
        w = max(1, int(smooth_window))
        sx = pd.Series(xy[:, 0]).rolling(w, center=True, min_periods=1).median().to_numpy()
        sy = pd.Series(xy[:, 1]).rolling(w, center=True, min_periods=1).median().to_numpy()
        out[str(uid)] = {int(fr): (float(a), float(b))
                         for fr, a, b in zip(g["frame"].to_numpy(), sx, sy)}
    return out


def endzone_pixels_by_uid(tracks_df, *, cam="endzone"):
    """{player_uid: {frame: (u, v)}} of native endzone foot pixels."""
    ez = tracks_df[tracks_df["cam"] == cam]
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for uid, grp in ez.groupby("player_uid"):
        if not _is_player(uid):
            continue
        out[str(uid)] = {int(fr): (float(u), float(v))
                         for fr, u, v in zip(grp["frame"], grp["foot_u"], grp["foot_v"])}
    return out


def identity_correspondences(tracks_df, sideline_track, *, sideline_cam="sideline",
                             endzone_cam="endzone", smooth_window=15):
    """{frame: (world (N,3), uv (N,2))} pairing sideline field points to endzone
    foot pixels by shared player_uid, per co-observed frame."""
    field = field_positions_by_uid(tracks_df, sideline_track, cam=sideline_cam,
                                    smooth_window=smooth_window)
    pix = endzone_pixels_by_uid(tracks_df, cam=endzone_cam)
    per_frame: dict[int, tuple[list, list]] = {}
    for uid, fmap in field.items():
        emap = pix.get(uid)
        if not emap:
            continue
        for fr, (x, y) in fmap.items():
            if fr not in emap:
                continue
            w_list, u_list = per_frame.setdefault(int(fr), ([], []))
            w_list.append([x, y, 0.0]); u_list.append(list(emap[fr]))
    return {fr: (np.asarray(w, np.float64), np.asarray(u, np.float64))
            for fr, (w, u) in per_frame.items()}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_endzone_identity.py -v`
Expected: PASS.

- [ ] **Step 5: Add a smoothing + exclusion test**

```python
def test_excludes_non_players_and_requires_both_cams():
    from nfl_gsplat.calibration.endzone_identity import identity_correspondences
    sl = _sideline_track([-3.6, 80.0, 36.0], [0, 0, 0], n_frames=1)
    recs = [
        {"frame": 0, "cam": "sideline", "player_uid": "2025_A_58", "foot_u": 960.0, "foot_v": 540.0, "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "__referee__", "foot_u": 5.0, "foot_v": 5.0, "conf": 1},
    ]
    # #58 only seen in sideline; referee only in endzone -> no correspondences
    assert identity_correspondences(_rows(recs), sl, smooth_window=1) == {}
```

Run: `python -m pytest tests/test_endzone_identity.py -v` → PASS.

- [ ] **Step 6: ruff + commit**

Run: `python -m ruff check nfl_gsplat/calibration/endzone_identity.py`
```bash
git add nfl_gsplat/calibration/endzone_identity.py tests/test_endzone_identity.py
git commit -m "feat(calibration): identity correspondences joined on player_uid"
```

---

## Task 3: `endzone_multiplay.py` — `EndzonePrior` + shared-C solve

**Files:**
- Create: `nfl_gsplat/calibration/endzone_multiplay.py`
- Test: `tests/test_endzone_multiplay.py`

**Interfaces:**
- Consumes: `solve_fixed_center(corrs_by_frame=None, image_size, *, init_results, _frame_data_override, view_deg, center_bounds, audit_drop_px)` (Task 1); `solve_pnp.CalibrationResult`; `utils.geometry.CameraIntrinsics, CameraPose`.
- Produces:
  - `EndzonePrior` dataclass: `x_range: tuple[float,float]`, `y_range`, `z_range`, `focal_range: tuple[float,float]`; property `center_bounds -> ((xlo,xhi),(ylo,yhi),(zlo,zhi))` and `center0 -> np.ndarray` (box midpoint).
  - `solve_endzone_identity(corrs_by_play: list[dict[int, tuple[np.ndarray, np.ndarray]]], image_size, prior: EndzonePrior, *, audit_drop_px=15.0, min_frames=10) -> list[list[CalibrationResult | None]]` (one result-list per play, aligned to that play's frame indices).

- [ ] **Step 1: Write the failing test**

Create `tests/test_endzone_multiplay.py`:

```python
import numpy as np
from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points


def _look_at(C, target):
    fwd = np.asarray(target, float) - np.asarray(C, float); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _play_corrs(C_true, rng, n_frames=25, n_players=8, wh=(1920, 1080)):
    corrs = {}
    for i in range(n_frames):
        tx = -20.0 + 30.0 * i / (n_frames - 1)
        xs = tx + rng.uniform(-8, 8, n_players); ys = rng.uniform(-22, 22, n_players)
        world = np.column_stack([xs, ys, np.zeros(n_players)])
        R = _look_at(C_true, [tx, 0.0, 0.0]); t = -R @ C_true
        f = 2500.0 + 300.0 * i / (n_frames - 1)
        K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
        uv = project_points(world, K, R, t)
        ok = np.isfinite(uv).all(axis=1)
        corrs[i] = (world[ok], uv[ok] + rng.normal(0, 0.5, uv[ok].shape))
    return corrs


def test_solve_endzone_identity_multiplay_recovers():
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior, solve_endzone_identity
    rng = np.random.default_rng(0)
    C_true = np.array([-112.0, 0.0, 24.0])
    plays = [_play_corrs(C_true, rng) for _ in range(3)]
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15), z_range=(10, 60),
                         focal_range=(1500, 3500))
    per_play = solve_endzone_identity(plays, (1920, 1080), prior, audit_drop_px=8.0)
    solved = [r for pl in per_play for r in pl if r is not None]
    assert len(solved) >= 30
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_true) < 1.0        # near truth, not the +112 mirror
    assert C_rec[0] < 0                                 # correct side (prior box)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_endzone_multiplay.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `endzone_multiplay.py`**

```python
"""Multi-play, shared-center endzone solve from identity correspondences.

The endzone camera is a fixed tripod: its center C is identical across all
first-half plays. We concatenate every play's frames into one frame set (offset
so indices never collide), share one C, and let each frame keep its own
rotation+focal -- exactly solve_fixed_center's model. The EndzonePrior bounds C
to the correct side so the Y-reflection is unreachable. Native pixels, view_deg=0."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.errors import CalibrationError

_PLAY_STRIDE = 1_000_000     # frame-index offset per play; assumes < 1e6 frames/play


@dataclass(frozen=True)
class EndzonePrior:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    focal_range: tuple[float, float]

    @property
    def center_bounds(self):
        return (self.x_range, self.y_range, self.z_range)

    @property
    def center0(self) -> np.ndarray:
        return np.array([sum(self.x_range) / 2, sum(self.y_range) / 2,
                         sum(self.z_range) / 2], float)


def solve_endzone_identity(corrs_by_play, image_size, prior, *,
                           audit_drop_px=15.0, min_frames=10):
    """corrs_by_play[p] = {frame: (world(N,3), uv(N,2))}. Returns one
    [CalibrationResult|None] list per play, aligned to that play's frames."""
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose

    # Union frame_data with per-play offsets.
    frame_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    lengths = []
    for p, corrs in enumerate(corrs_by_play):
        maxf = -1
        for fr, (w, uv) in corrs.items():
            if len(w) >= 4:
                frame_data[p * _PLAY_STRIDE + int(fr)] = (np.asarray(w, np.float64),
                                                          np.asarray(uv, np.float64))
            maxf = max(maxf, int(fr))
        lengths.append(maxf + 1)
    if len(frame_data) < min_frames:
        raise CalibrationError(
            f"endzone identity solve: only {len(frame_data)} usable frames across "
            f"{len(corrs_by_play)} plays (need >= {min_frames}) — thin jersey OCR or "
            "too few plays; re-run identity precompute or add plays.")

    T = (max(frame_data) + 1) if frame_data else 0
    # Anchor solve_fixed_center's multi-start at the prior box center.
    anchor = CalibrationResult(
        intrinsics=CameraIntrinsics(sum(prior.focal_range) / 2, sum(prior.focal_range) / 2,
                                    0.0, 0.0, 1, 1),
        pose=CameraPose(R=np.eye(3), t=-prior.center0),
        rms_px=0.0, num_correspondences=0, refined_with_ba=False)
    init_results = [anchor if i < 3 else None for i in range(T)]

    results, _mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=image_size, init_results=init_results,
        _frame_data_override=frame_data, view_deg=0,
        center_bounds=prior.center_bounds, audit_drop_px=audit_drop_px)

    # Split the union results back per play by the offset.
    per_play = []
    for p, length in enumerate(lengths):
        base = p * _PLAY_STRIDE
        per_play.append([results[base + i] if base + i < len(results) else None
                         for i in range(length)])
    return per_play
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_endzone_multiplay.py -v`
Expected: PASS (center within 1 m, correct side, ≥30 frames).

- [ ] **Step 5: Add a fail-loud + prior-excludes-reflection test**

```python
def test_solve_endzone_identity_fails_loud_too_few():
    import pytest
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior, solve_endzone_identity
    from nfl_gsplat.errors import CalibrationError
    prior = EndzonePrior((-150, -60), (-15, 15), (10, 60), (1500, 3500))
    with pytest.raises(CalibrationError, match="usable frames"):
        solve_endzone_identity([{0: (np.zeros((4, 3)), np.zeros((4, 2)))}],
                               (1920, 1080), prior)


def test_prior_center_bounds_and_center0():
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    p = EndzonePrior((-150, -60), (-15, 15), (10, 60), (1500, 3500))
    assert p.center_bounds == ((-150, -60), (-15, 15), (10, 60))
    assert np.allclose(p.center0, [-105.0, 0.0, 35.0])
```

Run: `python -m pytest tests/test_endzone_multiplay.py -v` → PASS.

- [ ] **Step 6: ruff + commit**

Run: `python -m ruff check nfl_gsplat/calibration/endzone_multiplay.py`
```bash
git add nfl_gsplat/calibration/endzone_multiplay.py tests/test_endzone_multiplay.py
git commit -m "feat(calibration): multi-play shared-center endzone identity solve"
```

---

## Task 4: `identity_precompute.py` — team split + `player_uid` (geometry-free)

**Files:**
- Create: `nfl_gsplat/calibration/identity_precompute.py`
- Test: `tests/test_identity_precompute.py`

**Interfaces:**
- Consumes: `identity.registry.resolve_tracks(df, candidates, cfg, id_col="track_id")`, `identity.registry.IdentityMatchConfig`, `identity.roster.OcrOnlySource`, `identity.team_color.dominant_jersey_color`, `identity.team_color.split_two_teams`; columns `frame, cam, track_id, jersey_number_ocr`.
- Produces: `assign_identity_columns(tracks_df, crop_provider, *, season) -> DataFrame` with added `team` + `player_uid` columns. `crop_provider(cam, frame, track_id) -> np.ndarray | None` returns a torso BGR crop (injected so tests pass synthetic crops and the script wires it to the videos). Team labels are a GLOBAL 2-color split across BOTH cameras so the same team gets the same label (hence the same `player_uid`) in both.

- [ ] **Step 1: Write the failing test** (mocks OCR + crops — no PaddleOCR)

Create `tests/test_identity_precompute.py`:

```python
import numpy as np
import pandas as pd


def _rows(recs):
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    df = pd.DataFrame(recs)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c not in ("cam",) else ""
    return df


def test_assign_identity_columns_joins_across_cameras():
    from nfl_gsplat.calibration.identity_precompute import assign_identity_columns
    # two teams by color; same jersey/team must yield the SAME uid in both cams
    red = np.full((20, 20, 3), (0, 0, 200), np.uint8)      # BGR red
    blue = np.full((20, 20, 3), (200, 0, 0), np.uint8)     # BGR blue
    color = {("sideline", 58): red, ("endzone", 58): red,
             ("sideline", 20): blue, ("endzone", 20): blue}
    recs = []
    for cam in ("sideline", "endzone"):
        for tid, jersey in ((58, 58), (20, 20)):
            recs.append({"frame": 0, "cam": cam, "track_id": tid,
                         "jersey_number_ocr": jersey})
    df = _rows(recs)

    def crop_provider(cam, frame, track_id):
        return color[(cam, int(track_id))]

    out = assign_identity_columns(df, crop_provider, season=2025)
    uid = {(r.cam, r.track_id): r.player_uid for r in out.itertuples()}
    # #58 same uid across cameras; #20 same across cameras; the two differ
    assert uid[("sideline", 58)] == uid[("endzone", 58)]
    assert uid[("sideline", 20)] == uid[("endzone", 20)]
    assert uid[("sideline", 58)] != uid[("sideline", 20)]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_identity_precompute.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `identity_precompute.py`**

```python
"""Geometry-free identity columns for calibration tracks.

Adds `team` + `player_uid` per track using ONLY jersey OCR + jersey color --
no camera, no geometry -- so joining sideline and endzone on `player_uid` gives
cross-camera identity without either endzone estimate. Team labels come from a
GLOBAL two-color split across BOTH cameras, so the same team gets the same label
(and uid) in both views."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.identity.registry import IdentityMatchConfig, resolve_tracks
from nfl_gsplat.identity.roster import OcrOnlySource
from nfl_gsplat.identity.team_color import dominant_jersey_color, split_two_teams


def _one_crop_per_track(tracks_df, crop_provider):
    """{(cam, track_id): crop} using the first frame each track appears."""
    crops = {}
    for (cam, tid), grp in tracks_df.groupby(["cam", "track_id"]):
        fr = int(grp["frame"].min())
        c = crop_provider(cam, fr, int(tid))
        if c is not None and getattr(c, "size", 0):
            crops[(cam, int(tid))] = c
    return crops


def assign_identity_columns(tracks_df, crop_provider, *, season):
    """Return a copy of tracks_df with `team` + `player_uid` columns."""
    crops = _one_crop_per_track(tracks_df, crop_provider)
    keys = list(crops)
    team_by_key: dict[tuple[str, int], str] = {}
    if keys:
        colors = np.stack([dominant_jersey_color(crops[k]) for k in keys])
        labels = split_two_teams(colors)          # global split, both cams
        team_by_key = {k: f"T{int(lbl)}" for k, lbl in zip(keys, labels)}

    out = tracks_df.copy()
    out["team"] = [team_by_key.get((c, int(t))) for c, t in
                   zip(out["cam"], out["track_id"])]

    cfg = IdentityMatchConfig(season=season)
    src = OcrOnlySource()
    parts = []
    for cam, grp in out.groupby("cam"):
        cand = src.candidates_for_play(str(cam), "0")      # [] -> synthesize
        parts.append(resolve_tracks(grp, cand, cfg, id_col="track_id"))
    return pd.concat(parts, ignore_index=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_identity_precompute.py -v`
Expected: PASS.

- [ ] **Step 5: ruff + commit**

Run: `python -m ruff check nfl_gsplat/calibration/identity_precompute.py`
```bash
git add nfl_gsplat/calibration/identity_precompute.py tests/test_identity_precompute.py
git commit -m "feat(calibration): geometry-free team+player_uid identity columns"
```

---

## Task 5: precompute script + multi-play CLI driver

**Files:**
- Create: `scripts/03c_identity_tracks.py`
- Modify: `nfl_gsplat/calibration/run_autocalib.py` (add `build_endzone_identity_from_plays`)
- Modify: `scripts/02_autocalibrate.py` (add `--mode identity-endzone`)
- Test: `tests/test_run_autocalib.py` (add one wiring test for `build_endzone_identity_from_plays`)

**Interfaces:**
- Consumes: `endzone_identity.identity_correspondences`, `endzone_multiplay.EndzonePrior`/`solve_endzone_identity`, `run_autocalib.assemble_track_from_results`, `cameras_io.load_camera_track`/`write_camera_track`, `utils.video.ffprobe_meta`.
- Produces: `build_endzone_identity_from_plays(*, play_dirs: list[Path], prior: EndzonePrior, sideline_cam="sideline", endzone_cam="endzone", fps, audit_drop_px=15.0) -> list[Path]` — writes each play's `endzone_*` into its `cameras.npz` (sideline preserved) and returns the written paths.

- [ ] **Step 1: Write the failing wiring test** (synthetic tracks + cameras, no video)

Add to `tests/test_run_autocalib.py` a test that builds two tiny play dirs in `tmp_path`, each with a `cameras.npz` (sideline only) and a `tracks.parquet` carrying `player_uid` for a few shared players whose endzone pixels are the projection of a known endzone camera; assert `build_endzone_identity_from_plays` writes `endzone` into each `cameras.npz` with a center in the prior box. (Reuse `_sideline_track` + `_look_at` helpers from `tests/test_endzone_identity.py`/`test_endzone_multiplay.py` — copy them locally; the plan forbids "similar to" references, so duplicate the ~6 lines.)

```python
def test_build_endzone_identity_from_plays(tmp_path):
    import numpy as np, pandas as pd
    from nfl_gsplat.calibration.cameras_io import write_camera_track, CameraTrack
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_identity_from_plays
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points
    # ... build 2 play dirs each with cameras.npz (sideline) + tracks.parquet
    #     (sideline feet projecting to known field pts; endzone feet = projection
    #     of a ground-truth endzone camera at C=(-112,0,24) for shared uids) ...
    # (full body written during implementation; asserts endzone center in box)
```

Note to implementer: write the FULL test body here during implementation — the world→sideline-pixel and world→endzone-pixel projections mirror Task 2/3 helpers; assert the written `endzone` track's mean center is within the prior box and `|X|>60`.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_run_autocalib.py::test_build_endzone_identity_from_plays -v`
Expected: FAIL — `build_endzone_identity_from_plays` not defined.

- [ ] **Step 3: Implement `build_endzone_identity_from_plays`** in `run_autocalib.py`

```python
def build_endzone_identity_from_plays(*, play_dirs, prior, sideline_cam="sideline",
                                      endzone_cam="endzone", fps, audit_drop_px=15.0):
    """Calibrate the endzone camera from jersey-identity correspondences across
    several first-half plays sharing one center; write endzone_* into each play's
    cameras.npz (sideline preserved)."""
    from pathlib import Path
    import pandas as pd
    from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
    from nfl_gsplat.calibration.endzone_identity import identity_correspondences
    from nfl_gsplat.calibration.endzone_multiplay import solve_endzone_identity
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.video import ffprobe_meta

    play_dirs = [Path(p) for p in play_dirs]
    corrs_by_play, metas, cams_by_play, image_size = [], [], [], None
    for pdir in play_dirs:
        cams = load_camera_track(pdir / "cameras.npz")
        sl = cams.get(sideline_cam)
        if sl is None:
            raise SetupError(f"no {sideline_cam!r} camera in {pdir/'cameras.npz'} — "
                             "run the sideline calibration first.")
        df = pd.read_parquet(pdir / "tracks.parquet")
        if "player_uid" not in df.columns:
            raise SetupError(f"{pdir/'tracks.parquet'} has no player_uid — run "
                             "scripts/03c_identity_tracks.py first.")
        meta = ffprobe_meta(str(pdir / f"{endzone_cam}.mp4"))
        image_size = (meta.width, meta.height)
        corrs_by_play.append(identity_correspondences(
            df, sl, sideline_cam=sideline_cam, endzone_cam=endzone_cam))
        metas.append(meta); cams_by_play.append(cams)

    per_play = solve_endzone_identity(corrs_by_play, image_size, prior,
                                      audit_drop_px=audit_drop_px)
    written = []
    for pdir, cams, meta, results in zip(play_dirs, cams_by_play, metas, per_play):
        results = (results + [None] * meta.num_frames)[:meta.num_frames]
        cams[endzone_cam] = assemble_track_from_results(
            results, width=meta.width, height=meta.height, max_gap=30)
        written.append(write_camera_track(pdir / "cameras.npz", cams, fps=fps))
    return written
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_run_autocalib.py::test_build_endzone_identity_from_plays -v`
Expected: PASS.

- [ ] **Step 5: Wire the CLI** — add `identity-endzone` to `scripts/02_autocalibrate.py`

Add `identity_endzone = "identity-endzone"` to `CalibMode`; add options `--play-dirs` (comma-separated, defaults to the single `--play-dir`) and `--endzone-prior` (read from `meta.yaml`'s `endzone_prior:` block: `x_range/y_range/z_range/focal_range`). In `main`, when `mode is CalibMode.identity_endzone`:

```python
        from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
        from nfl_gsplat.calibration.run_autocalib import build_endzone_identity_from_plays
        ep = meta.endzone_prior   # loaded from meta.yaml; SetupError if absent
        prior = EndzonePrior(tuple(ep["x_range"]), tuple(ep["y_range"]),
                             tuple(ep["z_range"]), tuple(ep["focal_range"]))
        dirs = [PlayDir.from_dir(Path(p)).dir for p in (play_dirs_opt or [play_dir])]
        out = build_endzone_identity_from_plays(play_dirs=dirs, prior=prior, fps=meta.fps)
        _LOG.info(f"wrote endzone identity calibration → {out}")
```

Add an `endzone_prior` field to the meta loader (`load_meta`) reading the optional `endzone_prior:` mapping; raise `SetupError` naming the missing block if `--mode identity-endzone` is used without it.

- [ ] **Step 6: Implement `scripts/03c_identity_tracks.py`** (orchestration; manual/PACE)

```python
"""Detect+track both cameras, OCR jerseys, assign geometry-free player_uid ->
tracks.parquet (with track_id, jersey_number_ocr, team, player_uid).

    python scripts/03c_identity_tracks.py data/2025/week_04/SEA_at_AZ/play_001 \
        --weights yolov8n.pt --season 2025      # env: nfl_smplx (PaddleOCR, GPU)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import pandas as pd
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.tracking.detect_track import TrackingConfig, detect_and_track
from nfl_gsplat.tracking.jersey_ocr import vote_jersey_numbers, JerseyOCRConfig
from nfl_gsplat.calibration.identity_precompute import assign_identity_columns


def _crop_provider(pd_):
    caps = {cam: cv2.VideoCapture(str(pd_.video(cam))) for cam in pd_.cameras}
    def provider(cam, frame, track_id, _df=None):
        cap = caps[cam]; cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
        return img if ok else None
    return provider


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir"); ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--cameras", default="sideline,endzone")
    args = ap.parse_args()
    cams = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pd_ = PlayDir.from_dir(args.play_dir, cameras=cams)
    cfg = TrackingConfig(yolo_weights=args.weights, device=args.device)
    dfs = [detect_and_track(pd_.video(cam), cam, cfg) for cam in pd_.cameras
           if Path(pd_.video(cam)).exists()]
    df = pd.concat(dfs, ignore_index=True)
    df = vote_jersey_numbers(df, pd_, JerseyOCRConfig())      # fills jersey_number_ocr
    df = assign_identity_columns(df, _crop_provider(pd_), season=args.season)
    df.to_parquet(pd_.tracks, index=False)
    print(f"wrote {len(df)} rows with player_uid -> {pd_.tracks}")


if __name__ == "__main__":
    main()
```

Note: confirm `vote_jersey_numbers`' exact signature during implementation (Task read shows `vote_jersey_numbers(...)` at `jersey_ocr.py:75`); adapt the call to its real parameters. This script is exercised manually on PACE, not in CI.

- [ ] **Step 7: Run wiring test + ruff, commit**

Run: `python -m pytest tests/test_run_autocalib.py -v` and `python -m ruff check nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py scripts/03c_identity_tracks.py`
```bash
git add nfl_gsplat/calibration/run_autocalib.py scripts/02_autocalibrate.py scripts/03c_identity_tracks.py tests/test_run_autocalib.py
git commit -m "feat(calibration): identity-endzone CLI + multi-play driver + precompute script"
```

---

## Task 6: full regression + real acceptance (manual)

**Files:** none (verification only).

- [ ] **Step 1: Full fast suite** — `python -m pytest -m "not gpu and not slow" -q` → all green.
- [ ] **Step 2: ruff** — `python -m ruff check nfl_gsplat tests scripts` → clean.
- [ ] **Step 3: Precompute (PACE embers, nfl_smplx env)** — run `scripts/03c_identity_tracks.py` on each first-half SEA@AZ play (both cameras). Confirm each `tracks.parquet` has non-`-1` `jersey_number_ocr` for a healthy fraction of tracks and a `player_uid` per track; count shared uids across cameras per play (expect ≥ ~5).
- [ ] **Step 4: Set the prior** — add an `endzone_prior:` block to each play's `meta.yaml` (`x_range` behind Seattle's end i.e. X<0, `y_range` small around 0, `z_range` 10–60, `focal_range` 1500–3500). Verify the X<0 sign against the sideline territory/goal-line orientation once and record it.
- [ ] **Step 5: Solve** — `python scripts/02_autocalibrate.py --play-dir <play_001> --play-dirs <p1,p2,...> --mode identity-endzone`. Expect `endzone_*` written to each `cameras.npz` with center in the prior box (`|X|` 60–150, `|Y|` ≤ 15, Z 10–60). Back up each `cameras.npz` first (scratchpad), since the run overwrites it.
- [ ] **Step 6: Same-field-point proof** — pick a shared `player_uid` + frame; project its sideline foot to field, project that field point through the new endzone camera, confirm it lands within a few meters / tens of px of the endzone foot detection. Save any diagnostic image to `C:\Users\sumedh\diag\` (outside the repo).
- [ ] **Step 7: Report** — center, per-play solved-frame counts, focal range, same-field-point residual. No commit (verification task).

---

## Self-Review

**Spec coverage:**
- Native pixels / view_deg=0 / no roll → Global Constraints + Tasks 3/5 (`view_deg=0`). ✓
- Geometry-free cross-camera identity via `resolve_tracks(id_col=track_id)` + join on `player_uid` → Tasks 2, 4. ✓
- Multi-play shared-C via `solve_fixed_center` over union of frames → Task 3. ✓
- `center_bounds` (reflection unreachable) + `audit_drop_px` (player-foot floor), default-preserving → Task 1. ✓
- Precompute wiring (detect+track+OCR+identity) → Tasks 4, 5 (script). ✓
- `EndzonePrior` from meta.yaml → Task 5. ✓
- CLI `--mode identity-endzone`, writes endzone_* preserving sideline → Task 5. ✓
- Error handling (no player_uid, prior missing, too-few frames) → Tasks 3, 5 (`SetupError`/`CalibrationError`). ✓
- Testing: synthetic multi-play recover + reflection-excluded (Task 3), identity join + exclusion (Task 2), center_bounds + default-preserving (Task 1), OCR-only uid path (Task 4), real acceptance + same-field-point (Task 6). ✓
- Vanishing-point cross-check → explicitly OUT of v1 scope in the spec; no task. ✓ (intentional)

**Placeholder scan:** Task 5 Step 1 leaves the wiring-test *body* to be written during implementation with an explicit instruction (the projection helpers are fully specified in Tasks 2/3) — flagged, not a silent TODO. Task 5 Step 6 notes confirming `vote_jersey_numbers`' exact signature (script is manual/PACE, not CI-gated). No other placeholders.

**Type consistency:** `frame_data = {frame: (world(N,3) float64, uv(N,2) float64)}` consistent across Tasks 2/3/5. `EndzonePrior(x_range,y_range,z_range,focal_range)` + `.center_bounds`/`.center0` consistent Tasks 3/5. `solve_fixed_center(..., center_bounds, audit_drop_px)` consistent Tasks 1/3. `player_uid`/`OTHER_UID`/`REFEREE_UID` consistent Tasks 2/4. `assign_identity_columns(df, crop_provider, *, season)` consistent Tasks 4/5.

## Risks (carried from spec)

- OCR yield per play thin → multi-play + per-track voting + PaddleOCR; fail-loud floor (`min_frames`) surfaces it.
- OCR misreads → `registry`/`team_color` gating + robust loss + ICP tighten.
- Foot-point semantics differ between views → bounded systematic nuisance; a per-camera foot offset is a future follow-on if it dominates.
- `audit_drop_px` too tight/loose for real feet → tune in Task 6 with the same-field-point residual as the gate; start 15px.
- Shared-C assumption (fixed tripod across first-half plays) → first-half only; per-play residuals expose drift.
