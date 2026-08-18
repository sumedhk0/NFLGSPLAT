# Cross-Camera Bootstrap Roll Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--mode cross-endzone` recover a plausible endzone camera on real footage by teaching the cross-camera bootstrap that the endzone works in a 90°-rotated frame (the camera carries a 90° roll the current look-at never applies).

**Architecture:** Two contained changes to `nfl_gsplat/calibration/cross_cam_calib.py` only. (1) `_hyp_cam` gains a `roll_deg` keyword and composes `view_rotation._rz(roll_deg)` into its look-at rotation. (2) `solve_endzone_cross_camera`'s bootstrap sweeps roll ∈ {0,90,180,270} and scores candidates by tight-inlier count (≤15 px) instead of loose 200 px match count. The ICP rounds, `solve_fixed_center`, `joint_solve.py`, and the CLI are untouched.

**Tech Stack:** Python, NumPy, SciPy (`linear_sum_assignment`), pytest. No new deps.

## Global Constraints

- Change ONLY `nfl_gsplat/calibration/cross_cam_calib.py` (source) and `tests/test_cross_cam_calib.py` (tests). No change to `joint_solve.py`, `run_autocalib.py`, `scripts/02_autocalibrate.py`, `view_rotation.py`.
- Reuse `view_rotation._rz(deg)` for the roll matrix — do NOT re-derive rotation math.
- `roll_deg=0` (and the existing 2-arg-plus-focal call sites) must behave EXACTLY as before — `_rz` is only composed when a nonzero roll is requested by the bootstrap sweep.
- Fail loud with `CalibrationError` + actionable pointer; no silent fallback that changes numerical results. The existing bootstrap/no-usable-frames guards and messages stay verbatim.
- `pytest -m "not gpu and not slow"` green; ruff clean. Synthetic tests stay CPU/fast.
- Camera convention: `CameraPose.R/t` world→camera, `x_cam = R X + t`, center `C = −Rᵀt`. `_rz` composes as `R_orig = _rz(deg) @ R_rotated` (from `view_rotation` docstring), so the rotated-frame camera is `_rz(deg) @ look_at`.
- Module const to add: `_BOOTSTRAP_TIGHT_PX = 15.0`.
- New commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `nfl_gsplat/calibration/cross_cam_calib.py` — MODIFY
  - `_hyp_cam` (lines 87-95): add `roll_deg=0` keyword; compose `_rz(roll_deg)`.
  - `solve_endzone_cross_camera` (lines 164-262): bootstrap sweeps roll + scores by tight inliers; add `_BOOTSTRAP_TIGHT_PX` module const; thread winning roll into round-0 `_hyp_cam` calls only.
- `tests/test_cross_cam_calib.py` — MODIFY
  - Add `_hyp_cam` roll unit test.
  - Rewrite `test_solve_endzone_cross_camera_recovers_synthetic` so ground-truth carries a real 90° roll.
  - Add a roll-search-is-load-bearing regression guard.

---

## Task 1: Roll-aware `_hyp_cam`

**Files:**
- Modify: `nfl_gsplat/calibration/cross_cam_calib.py:87-95`
- Test: `tests/test_cross_cam_calib.py`

**Interfaces:**
- Consumes: `view_rotation._rz(deg) -> (3,3)` (existing), `_look_at_R(C, target) -> (3,3)|None` (existing, `cross_cam_calib.py:72`).
- Produces: `_hyp_cam(C, world_pts, image_size, f, *, roll_deg=0) -> (K, R, t) | None`, where for the returned rotation, `roll_deg=90` gives exactly `_rz(90) @ (the roll_deg=0 rotation)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cross_cam_calib.py`:

```python
def test_hyp_cam_applies_roll():
    from nfl_gsplat.calibration.cross_cam_calib import _hyp_cam
    from nfl_gsplat.calibration.view_rotation import _rz
    world = np.array([[-40.0, 5.0, 0.0], [-30.0, -8.0, 0.0], [-20.0, 2.0, 0.0]])
    C = np.array([-110.0, 0.0, 40.0])
    wh = (1080, 1920)
    _, R0, t0 = _hyp_cam(C, world, wh, 2000.0)                 # roll_deg=0 default
    K90, R90, t90 = _hyp_cam(C, world, wh, 2000.0, roll_deg=90)
    assert np.allclose(R90, _rz(90) @ R0, atol=1e-9)
    assert np.allclose(t90, _rz(90) @ t0, atol=1e-9)           # t = -R @ C stays consistent
    # roll_deg=0 unchanged vs the pre-existing 4-positional call
    _, R0b, _ = _hyp_cam(C, world, wh, 2000.0, roll_deg=0)
    assert np.allclose(R0b, R0, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cross_cam_calib.py::test_hyp_cam_applies_roll -v`
Expected: FAIL — `_hyp_cam() got an unexpected keyword argument 'roll_deg'`.

- [ ] **Step 3: Write minimal implementation**

Replace `cross_cam_calib.py:87-95` (`_hyp_cam`) with:

```python
def _hyp_cam(C, world_pts, image_size, f, *, roll_deg=0):
    """Per-frame hypothesized endzone (K,R,t): look at the frame's player
    centroid with focal f. roll_deg composes the known view-rotation roll
    (via view_rotation._rz) so the hypothesized camera matches a rotated
    working frame -- the endzone runs at view_deg=90, so its true camera
    carries a 90 deg roll a plain look-at would miss."""
    R = _look_at_R(C, np.asarray(world_pts, float).mean(axis=0))
    if R is None:
        return None
    if roll_deg:
        from nfl_gsplat.calibration.view_rotation import _rz
        R = _rz(roll_deg) @ R
    t = -R @ np.asarray(C, np.float64)
    K = np.array([[f, 0, image_size[0] / 2.0], [0, f, image_size[1] / 2.0], [0, 0, 1.0]])
    return K, R, t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cross_cam_calib.py::test_hyp_cam_applies_roll -v`
Expected: PASS.

- [ ] **Step 5: Run the existing cross-cam suite to prove roll_deg=0 is a no-op**

Run: `python -m pytest tests/test_cross_cam_calib.py -v`
Expected: the four `match_frame`/`sideline`/`endzone`/`fails_loud` tests still PASS. `test_solve_endzone_cross_camera_recovers_synthetic` still PASS (it uses no roll yet — Task 2/3 rewrite it).

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py
git commit -m "feat(calibration): _hyp_cam composes view-rotation roll_deg"
```

---

## Task 2: Roll-searched, tight-inlier bootstrap

**Files:**
- Modify: `nfl_gsplat/calibration/cross_cam_calib.py:164-262` (`solve_endzone_cross_camera`) + add module const near top.
- Test: covered by Task 3's rewritten synthetic test (this task's deliverable is verified there; the existing fail-loud test must keep passing here).

**Interfaces:**
- Consumes: `_hyp_cam(..., roll_deg=roll)` from Task 1; `_build_corrs(field_by, feet_by, cam_for_frame, image_size, max_px) -> (corrs, total)` (existing, unchanged); `match_frame` (existing).
- Produces: `solve_endzone_cross_camera(...)` signature UNCHANGED; internal behavior now selects `(C0, focal, roll)` by tight-inlier count and threads the winning roll into round-0 `_hyp_cam` calls.

- [ ] **Step 1: Add the module constant**

After the imports block near the top of `cross_cam_calib.py` (after line 17, `from nfl_gsplat.utils.geometry import project_points`), add:

```python
# Bootstrap scores candidate (center, focal, roll) by how many players
# reproject within this tight gate -- a quality score. The old loose 200px
# match COUNT rewarded spurious clustering (picked a wrong C with thousands of
# junk matches over the correct region).
_BOOTSTRAP_TIGHT_PX = 15.0
```

- [ ] **Step 2: Add a tight-inlier scorer helper (module-level, above `solve_endzone_cross_camera`)**

Insert immediately before `def solve_endzone_cross_camera` (line 164):

```python
def _tight_inliers(field_by, feet_by, C0, f, roll, image_size, tight_px):
    """Count players whose reprojection under _hyp_cam(C0, f, roll) lands within
    tight_px of an optimally-assigned endzone foot. A quality score for the
    bootstrap: unlike a loose match count it cannot be inflated by a wrong
    center accidentally gate-matching scattered detections."""
    total = 0
    for fidx, world in field_by.items():
        feet = feet_by.get(fidx)
        if feet is None or not len(world):
            continue
        krt = _hyp_cam(C0, world, image_size, f, roll_deg=roll)
        if krt is None:
            continue
        mw, _mu = match_frame(world, feet, *krt, max_px=tight_px)
        total += len(mw)
    return total
```

- [ ] **Step 3: Rewrite the bootstrap block to sweep roll + score by tight inliers**

Replace the bootstrap section of `solve_endzone_cross_camera` — from the `def _score(C0, f):` definition (line 196) through the `init_results = [anchor ...]` line (line 240) — with the version below. The docstring above `_score` (lines 180-195) describing the loose gate is being superseded; delete it along with `_score`. Everything from `results = None` (line 242) onward is UNCHANGED.

```python
    # Bootstrap: a single init_C can be tens of meters off the truth, so we
    # multi-start over init_C + the behind-endzone grid (the same grid the
    # joint solver trusts for this geometry). The endzone runs in a 90deg-
    # rotated working frame, so its true camera carries an in-plane roll that a
    # plain look-at omits -- every projected player would land ~90deg rotated
    # from its detection. We therefore also sweep roll in {0,90,180,270} and
    # score each (center, focal, roll) by TIGHT-inlier count (players landing
    # within _BOOTSTRAP_TIGHT_PX of an assigned foot). Tight-inlier count is a
    # quality score: a wrong center/roll cannot inflate it by accidentally
    # gate-matching scattered detections the way the old loose 200px COUNT
    # could. The winning roll is threaded into the round-0 _hyp_cam calls; the
    # seed corrs for the ICP are then built under the looser match_px[0] gate
    # (with that roll) so solve_fixed_center has enough points to lock.
    _ROLLS = (0, 90, 180, 270)
    total_possible = sum(min(len(w), len(feet_by_frame.get(fidx, ())))
                         for fidx, w in field_by_frame.items())

    candidate_centers = [np.asarray(init_C, np.float64)] + [
        np.array([X, Y, Z]) for X in _GRID_EZ_X
        for Y in _GRID_EZ_Y for Z in _GRID_EZ_Z]
    best = None   # (tight_count, C0, f, roll)
    for C0 in candidate_centers:
        for f in focal_guesses:
            for roll in _ROLLS:
                n = _tight_inliers(field_by_frame, feet_by_frame, C0, f, roll,
                                   image_size, _BOOTSTRAP_TIGHT_PX)
                if best is None or n > best[0]:
                    best = (n, C0, f, roll)
        # An init_C (roll included) that already tight-matches nearly every
        # player short-circuits the rest of the grid scan.
        if best is not None and total_possible > 0 and best[0] >= 0.9 * total_possible \
                and np.array_equal(best[1], np.asarray(init_C, np.float64)):
            break

    _n, C_boot, f_boot, roll_boot = best

    # Build the seed correspondences at the winner under the looser round-0
    # gate (match_px[0]) so the ICP has enough points; identity stays correct
    # because Hungarian assignment is driven by RELATIVE pixel distance.
    def _boot_cam(fidx):
        w = field_by_frame.get(fidx)
        return _hyp_cam(C_boot, w, image_size, f_boot, roll_deg=roll_boot) \
            if w is not None and len(w) else None
    corrs, _total = _build_corrs(field_by_frame, feet_by_frame, _boot_cam,
                                 image_size, match_px[0])

    if sum(len(w) for (w, _u) in corrs.values()) < 20 or len(corrs) < 5:
        raise CalibrationError(
            "endzone: cross-camera matching did not converge (too few player "
            "matches at bootstrap) — check frame sync and player detections.")

    # Anchor solve_fixed_center's OWN multi-start at the bootstrap winner.
    # solve_fixed_center's coarse plausibility grid (30-60m spacing) gets a
    # real ~50-iteration solve only at a "plausible anchor" candidate (see
    # init_from_results); everywhere else it only spends a cheap 12-iteration
    # scoring pass, which -- given the bootstrap corrs carry a few percent of
    # identity mismatches from the approximate gaze -- is not enough to
    # relocate the camera precisely. Feeding the bootstrap winner in as an
    # anchor (>=3 identical entries, the threshold init_from_results requires)
    # gives it the deep solve it needs.
    anchor = CalibrationResult(
        intrinsics=CameraIntrinsics(2000.0, 2000.0, 0.0, 0.0, 1, 1),
        pose=CameraPose(R=np.eye(3), t=-np.asarray(C_boot, np.float64)),
        rms_px=0.0, num_correspondences=0, refined_with_ba=False)
    init_results = [anchor if i < 3 else None for i in range(T)]
```

Note the `CalibrationError` message and the `>= 20 / < 5` guard are preserved verbatim; only the candidate scoring changed. The final-guard block `if results is None or all(r is None ...)` at the end of the function is unchanged.

- [ ] **Step 4: Run the fail-loud test to confirm the guard still triggers**

Run: `python -m pytest tests/test_cross_cam_calib.py::test_solve_endzone_cross_camera_fails_loud_no_matches -v`
Expected: PASS — feet at (9999,9999) yield zero tight inliers at every (center,focal,roll), so the seed corrs stay empty and `CalibrationError` raises (message matches `"cross-camera"`).

- [ ] **Step 5: Run ruff**

Run: `python -m ruff check nfl_gsplat/calibration/cross_cam_calib.py`
Expected: clean (no unused `_score`/`_total`, no unused imports left behind).

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/cross_cam_calib.py
git commit -m "feat(calibration): bootstrap sweeps roll + scores by tight inliers"
```

---

## Task 3: Synthetic recovery test carries a real 90° roll

**Files:**
- Modify: `tests/test_cross_cam_calib.py` — rewrite `test_solve_endzone_cross_camera_recovers_synthetic` (lines 99-139); add a load-bearing guard test.

**Interfaces:**
- Consumes: `solve_endzone_cross_camera(field_by, feet_by, ez_wh, *, init_C, view_deg=90)` (existing signature); `view_rotation._rz`, `rotated_wh`; `_look_at` test helper (existing, line 8).
- Produces: a synthetic test whose ground-truth endzone camera is `R = _rz(90) @ look_at(...)`, so the bootstrap roll sweep is exercised; plus a monkeypatched guard proving roll-search is load-bearing.

- [ ] **Step 1: Rewrite the synthetic recovery test to bake in a 90° roll**

Replace `test_solve_endzone_cross_camera_recovers_synthetic` (lines 99-139) with:

```python
def test_solve_endzone_cross_camera_recovers_synthetic():
    # Ground truth: a fixed endzone camera + ~20 players over 40 frames. The
    # camera is built in the 90deg-ROTATED working frame (R = _rz(90) @ look_at),
    # matching the real endzone pipeline -- so the bootstrap MUST find roll=90 to
    # seed. Sideline gives the players' true field points; endzone sees their feet.
    from nfl_gsplat.calibration.cross_cam_calib import solve_endzone_cross_camera
    from nfl_gsplat.calibration.view_rotation import rotated_wh, _rz
    rng = np.random.default_rng(0)
    ow = (1920, 1080)
    ez_wh = rotated_wh(90, ow)                      # endzone works in rotated frame
    C_true = np.array([-110.0, -8.0, 45.0])
    n_frames, n_players = 40, 20
    field_by, feet_by, f_by = {}, {}, {}
    for i in range(n_frames):
        tx = -20.0 + 30.0 * i / (n_frames - 1)                     # camera pans down-field
        # Representative endzone geometry: players cluster near the line of
        # scrimmage (the camera's per-frame gaze in down-field X) and spread
        # across the field width (Y) -- so _hyp_cam's centroid look-at is a
        # good approximation, exactly as on real footage (where the spec
        # measured 405 tight inliers at roll=90). A scene that scattered
        # players independently of the pan would decouple centroid from gaze
        # and defeat the tight-inlier gate even at the true center.
        xs = tx + rng.uniform(-8.0, 8.0, n_players)
        ys = rng.uniform(-22.0, 22.0, n_players)
        world = np.column_stack([xs, ys, np.zeros(n_players)])
        field_by[i] = world
        R = _rz(90) @ _look_at(C_true, np.array([tx, 0.0, 0.0]))   # real 90deg roll
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
    assert np.linalg.norm(C_rec - C_true) < 0.5
    for r in solved:
        assert np.allclose(r.pose.center_world(), C_rec)     # one fixed center
    for i, r in enumerate(results):
        if r is None:
            continue
        assert abs(r.intrinsics.fx - f_by[i]) / f_by[i] < 0.02
```

- [ ] **Step 2: Run it to verify the roll-aware bootstrap recovers the rolled scene**

Run: `python -m pytest tests/test_cross_cam_calib.py::test_solve_endzone_cross_camera_recovers_synthetic -v`
Expected: PASS — center within 0.5 m, focals within 2 %. (This only passes because Task 1+2 made the bootstrap sweep roll; with roll fixed to 0 the projected players land 90° off and nothing seeds.)

- [ ] **Step 3: Add the load-bearing guard test**

The guard proves the roll sweep is not a tautology: force `_ROLLS` to `(0,)` via monkeypatching the module and assert the SAME rolled scene fails to recover. Because `_ROLLS` is a local tuple inside `solve_endzone_cross_camera`, expose it as a module-level default so it can be patched. In `cross_cam_calib.py`, hoist the tuple to a module constant next to `_BOOTSTRAP_TIGHT_PX`:

```python
_BOOTSTRAP_ROLLS = (0, 90, 180, 270)
```

and in `solve_endzone_cross_camera` change `_ROLLS = (0, 90, 180, 270)` to `_ROLLS = _BOOTSTRAP_ROLLS`.

Then add this test to `tests/test_cross_cam_calib.py`:

```python
def test_bootstrap_roll_sweep_is_load_bearing(monkeypatch):
    # Same rolled scene as the recovery test, but with the roll sweep disabled
    # (only roll=0 tried): the solve must NOT recover, proving roll search is
    # what makes the recovery work -- not luck.
    import nfl_gsplat.calibration.cross_cam_calib as cc
    from nfl_gsplat.errors import CalibrationError
    from nfl_gsplat.calibration.view_rotation import rotated_wh, _rz
    monkeypatch.setattr(cc, "_BOOTSTRAP_ROLLS", (0,))
    rng = np.random.default_rng(0)
    ow = (1920, 1080)
    ez_wh = rotated_wh(90, ow)
    C_true = np.array([-110.0, -8.0, 45.0])
    n_frames, n_players = 40, 20
    field_by, feet_by = {}, {}
    for i in range(n_frames):
        tx = -20.0 + 30.0 * i / (n_frames - 1)
        xs = tx + rng.uniform(-8.0, 8.0, n_players)          # same representative
        ys = rng.uniform(-22.0, 22.0, n_players)             # scene as recovery test
        world = np.column_stack([xs, ys, np.zeros(n_players)])
        field_by[i] = world
        R = _rz(90) @ _look_at(C_true, np.array([tx, 0.0, 0.0]))
        t = -R @ C_true
        f = 2400.0 + 400.0 * i / (n_frames - 1)
        K = CameraIntrinsics(f, f, ez_wh[0] / 2, ez_wh[1] / 2, ez_wh[0], ez_wh[1]).K()
        uv = project_points(world, K, R, t)
        ok = np.isfinite(uv).all(axis=1)
        feet_by[i] = uv[ok] + rng.normal(0, 0.5, uv[ok].shape)
    # Without the roll sweep the bootstrap can't seed: either it fails loud at
    # the match guard, or it "solves" to a center far from truth. Accept either
    # as proof roll=0 alone is insufficient.
    try:
        results = solve_endzone_cross_camera(
            field_by, feet_by, ez_wh, init_C=np.array([-90.0, 0.0, 35.0]), view_deg=90)
    except CalibrationError:
        return
    solved = [r for r in results if r is not None]
    if solved:
        C_rec = solved[0].pose.center_world()
        assert np.linalg.norm(C_rec - C_true) > 5.0
```

(`solve_endzone_cross_camera` is already imported at the top of the recovery test's scope; add a module-level `from nfl_gsplat.calibration.cross_cam_calib import solve_endzone_cross_camera` import at the top of the test file if not already present, or import it inside this test as the other tests do.)

- [ ] **Step 4: Run the guard test**

Run: `python -m pytest tests/test_cross_cam_calib.py::test_bootstrap_roll_sweep_is_load_bearing -v`
Expected: PASS — disabling the roll sweep prevents recovery (fails loud or lands >5 m off).

- [ ] **Step 5: Run the whole cross-cam file**

Run: `python -m pytest tests/test_cross_cam_calib.py -v`
Expected: all PASS (match_frame ×3, sideline, endzone, hyp_cam roll, recovery, fail-loud, roll-load-bearing).

- [ ] **Step 6: Commit**

```bash
git add nfl_gsplat/calibration/cross_cam_calib.py tests/test_cross_cam_calib.py
git commit -m "test(calibration): synthetic recovery carries real 90deg roll + roll-sweep guard"
```

---

## Task 4: Full regression + real acceptance

**Files:** none (verification only).

- [ ] **Step 1: Full fast suite**

Run: `python -m pytest -m "not gpu and not slow" -q`
Expected: all green (337+ tests). Confirms `joint_solve` sideline path and everything else untouched.

- [ ] **Step 2: ruff on the repo**

Run: `python -m ruff check nfl_gsplat tests`
Expected: clean.

- [ ] **Step 3: Real acceptance (manual, needs local data + calibrated sideline)**

Run: `python scripts/02_autocalibrate.py --play-dir data\2025\week_04\SEA_at_AZ\play_001 --mode cross-endzone`
Expected: writes `endzone_*` into `cameras.npz`; endzone center in the plausible box (|X| 50–150 m near −115, |Y| ≤ 40, Z 10–80, near measured C≈(−115,0,25)); sideline entries unchanged. If it fails loud, capture the message and treat as a Risk (below) — do NOT loosen guards silently.

- [ ] **Step 4: Same-field-point proof (manual)**

Pick a frame + a player detected in both cameras. Project the sideline foot to field `(X,Y,0)`; project that field point through the new endzone camera; confirm it lands within a few meters / tens of px of the endzone foot detection. Save any diagnostic image to `C:\Users\sumedh\diag\` (OUTSIDE the repo — never commit real frames).

- [ ] **Step 5: Report results** — center, solved-frame count, focal range, same-field-point residual. No commit (verification task).

---

## Self-Review

**Spec coverage:**
- Roll-aware `_hyp_cam` (spec §1) → Task 1. ✓
- Quality-scored, roll-searched bootstrap, `_BOOTSTRAP_TIGHT_PX=15.0` (spec §2) → Task 2. ✓
- No change to `joint_solve`/`run_autocalib`/CLI (spec Components / Out of scope) → enforced by Global Constraints; only `cross_cam_calib.py` + its test touched. ✓
- Unit test `_rz(90) @ R` exact ≤1e-9 (spec Testing) → Task 1 Step 1. ✓
- Synthetic recovery rebuilt with real 90° roll, <0.5 m, focals 2 % (spec Testing) → Task 3 Step 1. ✓
- Roll-search-disabled guard proving load-bearing (spec Testing) → Task 3 Step 3 (monkeypatch `_BOOTSTRAP_ROLLS`). ✓
- Real acceptance on SEA_at_AZ + same-field-point proof (spec Testing) → Task 4 Steps 3-4. ✓
- Fail-loud thresholds/messages unchanged (spec Error handling) → Task 2 Step 3 preserves the guard + `CalibrationError` string verbatim; Task 2 Step 4 tests it. ✓

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step shows full code. ✓

**Type consistency:** `_hyp_cam(..., roll_deg=0) -> (K,R,t)|None` used identically in Task 1, `_tight_inliers`, `_boot_cam`. `_BOOTSTRAP_TIGHT_PX` (const) and `_BOOTSTRAP_ROLLS` (const, patchable) named consistently across Tasks 2-3. `best` tuple `(tight_count, C0, f, roll)` unpacked as `(_n, C_boot, f_boot, roll_boot)`. ✓

## Risks

- **20 px bootstrap seed too coarse for `solve_fixed_center` on real data:** if the first solve doesn't lock despite tight inliers, tighten `match_px[0]` or add one ICP round — do NOT loosen the fail-loud guard. Real acceptance (Task 4) is the confirmation.
- **`monkeypatch` on `_BOOTSTRAP_ROLLS`:** requires the tuple be a module constant referenced by name inside `solve_endzone_cross_camera` (Task 3 Step 3 hoists it). If the function captured a local literal, the patch would be a no-op and the guard tautological — Step 3 explicitly changes the local to reference the module const.
