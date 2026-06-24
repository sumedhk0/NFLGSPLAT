# Hash-Row-Intersection Correspondences Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-tick hash matching with diagonal-safe yard-line ordering + RANSAC-fitted hash-row lines + line×row intersections, so per-frame field registration produces correct, low-RMS correspondences on real broadcast frames.

**Architecture:** All changes are pure geometry in `nfl_gsplat/calibration/field_identify.py`. Order/merge yard lines by their x at image mid-height (not mean-x, which is wrong for diagonal lines). Fit the two hash rows with RANSAC (averaging out the dense 1-yard ticks). Emit correspondences as yard-line × hash-row-line intersections (exact) instead of nearest-tick matches. The detector, hint format, sweep, PnP, and consumers are unchanged.

**Tech Stack:** Python 3.10, numpy, pytest. `YardLineSeg` from `field_features`. No cv2 needed (RANSAC is hand-rolled, deterministic).

**Reference spec:** `docs/superpowers/specs/2026-06-23-hash-row-intersection-correspondences-design.md`

## Current state (what exists)
`nfl_gsplat/calibration/field_identify.py` has: `IdentityState(line_yardage)`, `_line_x(seg)` (mean-x), `_seg_intersection(a, b)`, `_yard_step`, `seed_state_from_hint(feats, hint)`, `identify_correspondences(feats, prior)`. `YardLineSeg(p0, p1)` and `landmark_name(side, yd, lr, row)` come from `field_features`. Tests in `tests/test_field_identify.py`.

## Conventions
- Repo root: `C:/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/NFLGSPLAT`. `python -m pytest …` locally.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Task 1: `line_x_at` — diagonal-safe x at a given height

**Files:**
- Modify: `nfl_gsplat/calibration/field_identify.py`
- Test: `tests/test_field_identify.py` (extend)

- [ ] **Step 1: Add tests** to `tests/test_field_identify.py`:

```python
def test_line_x_at_vertical_is_constant():
    from nfl_gsplat.calibration.field_identify import line_x_at
    from nfl_gsplat.calibration.field_features import YardLineSeg
    seg = YardLineSeg((500.0, 0.0), (500.0, 1080.0))
    assert abs(line_x_at(seg, 0) - 500.0) < 1e-6
    assert abs(line_x_at(seg, 540) - 500.0) < 1e-6


def test_line_x_at_diagonal_interpolates():
    from nfl_gsplat.calibration.field_identify import line_x_at
    from nfl_gsplat.calibration.field_features import YardLineSeg
    # from (400, 0) to (600, 1000): x grows 0.2 per y. At y=500 → 500.
    seg = YardLineSeg((400.0, 0.0), (600.0, 1000.0))
    assert abs(line_x_at(seg, 0) - 400.0) < 1e-6
    assert abs(line_x_at(seg, 500) - 500.0) < 1e-6
    assert abs(line_x_at(seg, 1000) - 600.0) < 1e-6
```

- [ ] **Step 2: Run** `python -m pytest tests/test_field_identify.py -q` → the 2 new tests FAIL (no `line_x_at`).

- [ ] **Step 3: Add `line_x_at` to `field_identify.py`** (after `_line_x`):

```python
def line_x_at(seg, y: float) -> float:
    """Image-x of a (near-vertical) line segment at height ``y``.

    Yard lines are near-vertical but slanted in broadcast views, so x varies
    with y; their mean-x is unreliable for ordering/matching. Interpolates along
    the segment's direction. Degenerates to the mean-x for a perfectly
    horizontal segment (|dy| ~ 0), which shouldn't occur for yard lines."""
    (x0, y0), (x1, y1) = seg.p0, seg.p1
    dy = y1 - y0
    if abs(dy) < 1e-6:
        return 0.5 * (x0 + x1)
    t = (float(y) - y0) / dy
    return x0 + t * (x1 - x0)
```

- [ ] **Step 4: Run** → PASS (2 new + existing).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git add nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git commit -m "field_identify: add line_x_at (diagonal-safe x at a height)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `_merge_lines` — dedupe by mid-height x; reorder seed/identify

**Files:**
- Modify: `nfl_gsplat/calibration/field_identify.py`
- Test: `tests/test_field_identify.py` (extend)

- [ ] **Step 1: Add test:**

```python
def test_merge_lines_dedupes_same_diagonal_line():
    from nfl_gsplat.calibration.field_identify import _merge_lines, line_x_at
    from nfl_gsplat.calibration.field_features import YardLineSeg
    a = YardLineSeg((500.0, 0.0), (520.0, 1080.0))      # x@540 ≈ 510
    b = YardLineSeg((505.0, 200.0), (515.0, 760.0))     # x@540 ≈ 510
    far = YardLineSeg((800.0, 0.0), (820.0, 1080.0))    # x@540 ≈ 810
    merged = _merge_lines([a, b, far], tol=25.0, ref_y=540.0)
    xs = sorted(round(line_x_at(s, 540.0)) for s in merged)
    assert len(merged) == 2                      # a+b collapsed to one, far separate
    assert any(abs(x - 510) < 6 for x in xs)
    assert any(abs(x - 810) < 6 for x in xs)
```

- [ ] **Step 2: Run** → FAIL (no `_merge_lines`).

- [ ] **Step 3: Add `_merge_lines`** to `field_identify.py`:

```python
def _merge_lines(lines, tol: float, ref_y: float):
    """Merge yard-line segments whose x at ``ref_y`` are within ``tol`` (the same
    physical line detected as multiple segments). Returns lines sorted by x@ref_y,
    one representative per cluster (the one spanning the largest y-range)."""
    items = sorted(lines, key=lambda s: line_x_at(s, ref_y))
    merged = []
    for seg in items:
        x = line_x_at(seg, ref_y)
        if merged and abs(line_x_at(merged[-1], ref_y) - x) <= tol:
            prev = merged[-1]
            # keep whichever segment spans more vertical extent (more reliable slope)
            if abs(seg.p1[1] - seg.p0[1]) > abs(prev.p1[1] - prev.p0[1]):
                merged[-1] = seg
        else:
            merged.append(seg)
    return merged
```

- [ ] **Step 4: Update `seed_state_from_hint`** to order + snap by `line_x_at(·, mid)` and merge first. Replace its body:

```python
def seed_state_from_hint(feats, hint) -> IdentityState:
    """Initial IdentityState for hint.ref_frame: merge duplicate line detections,
    snap ref_x (read at image mid-height) to the nearest yard line, and label the
    rest by 5-yd index spacing. ``increasing`` = image direction yards grow."""
    mid = feats.image_size[1] / 2.0
    lines = _merge_lines(feats.yard_lines, tol=25.0, ref_y=mid)
    if not lines:
        return IdentityState()
    xs = [line_x_at(s, mid) for s in lines]
    seed_idx = min(range(len(xs)), key=lambda i: abs(xs[i] - hint.ref_x))
    step_per_index = 1 if hint.increasing == "right" else -1
    out: dict[float, tuple[str, int]] = {}
    for i, s in enumerate(lines):
        side, yard = _yard_step(hint.side, hint.yard, step_per_index * (i - seed_idx))
        if side:
            out[line_x_at(s, mid)] = (side, yard)
    return IdentityState(line_yardage=out)
```

(Note: `line_yardage` is now keyed by x@mid — that key is just a stable id used by
the propagation nearest-match in `identify_correspondences`, which we update in
Task 4 to also key by x@mid. Consistent.)

- [ ] **Step 5: Run** `python -m pytest tests/test_field_identify.py -q` → the merge test passes; the existing seed tests (`test_seed_from_hint_labels_by_spacing_and_direction`, `test_seed_crosses_50_to_home_when_increasing`) still pass (their lines are vertical, so x@mid == mean-x → unchanged behavior).

- [ ] **Step 6: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git add nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git commit -m "field_identify: merge duplicate lines + order/snap by x@mid-height

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `fit_hash_rows` — RANSAC two hash-row lines

**Files:**
- Modify: `nfl_gsplat/calibration/field_identify.py`
- Test: `tests/test_field_identify.py` (extend)

- [ ] **Step 1: Add tests:**

```python
def test_fit_hash_rows_finds_two_rows():
    import numpy as np
    from nfl_gsplat.calibration.field_identify import fit_hash_rows, line_x_at
    rng = np.random.default_rng(0)
    pts = []
    for x in range(200, 1400, 20):                       # dense 1-yard ticks, two rows
        pts.append((float(x), 360.0 + rng.normal(0, 1.0)))   # upper row ~y=360
        pts.append((float(x), 620.0 + rng.normal(0, 1.0)))   # lower row ~y=620
    pts += [(700.0, 150.0), (300.0, 900.0)]              # 2 noise points
    rows = fit_hash_rows(pts, image_width=1920)
    assert len(rows) == 2
    ys = sorted(0.5 * (r.p0[1] + r.p1[1]) for r in rows)
    assert abs(ys[0] - 360) < 10 and abs(ys[1] - 620) < 10
    # each row spans the image width
    assert min(r.p0[0] for r in rows) <= 1 and max(r.p1[0] for r in rows) >= 1919


def test_fit_hash_rows_too_few_returns_empty():
    from nfl_gsplat.calibration.field_identify import fit_hash_rows
    assert fit_hash_rows([(10.0, 20.0), (30.0, 40.0)], image_width=1920) == []
```

- [ ] **Step 2: Run** → FAIL (no `fit_hash_rows`).

- [ ] **Step 3: Add `fit_hash_rows`** to `field_identify.py`:

```python
def _ransac_line(pts, *, inlier_px: float, iters: int, rng):
    """Best-fit line over 2D points by RANSAC. Returns (inlier_mask, (a, b)) for
    y = a*x + b, or (None, None) if degenerate. Hash rows are near-horizontal so
    y = a*x + b is well-conditioned."""
    import numpy as np
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    best_mask, best_count = None, -1
    for _ in range(iters):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        x0, y0 = pts[i]; x1, y1 = pts[j]
        if abs(x1 - x0) < 1e-6:
            continue
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        resid = np.abs(pts[:, 1] - (a * pts[:, 0] + b))
        mask = resid <= inlier_px
        if mask.sum() > best_count:
            best_count, best_mask = int(mask.sum()), mask
    if best_mask is None:
        return None, None
    # least-squares refit on inliers
    xin, yin = pts[best_mask, 0], pts[best_mask, 1]
    A = np.vstack([xin, np.ones_like(xin)]).T
    a, b = np.linalg.lstsq(A, yin, rcond=None)[0]
    return best_mask, (float(a), float(b))


def fit_hash_rows(hashes, *, image_width: int, inlier_px: float = 6.0,
                  min_inliers: int = 6, iters: int = 200):
    """Fit up to two hash-ROW lines from raw tick points via RANSAC, returning
    each as a width-spanning ``YardLineSeg``. Averages out the dense 1-yard ticks
    and noise. Returns [] / [one] / [two] sorted by row height (upper first)."""
    import numpy as np

    from nfl_gsplat.calibration.field_features import YardLineSeg
    pts = list(hashes)
    if len(pts) < min_inliers:
        return []
    rng = np.random.default_rng(12345)
    rows = []
    remaining = np.asarray(pts, dtype=np.float64)
    for _ in range(2):
        if len(remaining) < min_inliers:
            break
        mask, line = _ransac_line(remaining, inlier_px=inlier_px, iters=iters, rng=rng)
        if mask is None or int(mask.sum()) < min_inliers:
            break
        a, b = line
        rows.append(YardLineSeg((0.0, b), (float(image_width), a * image_width + b)))
        remaining = remaining[~mask]
    rows.sort(key=lambda r: 0.5 * (r.p0[1] + r.p1[1]))   # upper (smaller y) first
    return rows
```

- [ ] **Step 4: Run** → PASS. If the synthetic two-row test fails (e.g. one RANSAC pass grabs both rows because `inlier_px` too loose), the defaults are fine for the 260px row gap here; keep `inlier_px=6`. Don't loosen the test.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git add nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git commit -m "field_identify: RANSAC-fit two hash-row lines from tick points

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `identify_correspondences` via line×row intersections

**Files:**
- Modify: `nfl_gsplat/calibration/field_identify.py`
- Test: `tests/test_field_identify.py` (replace the propagation test + add an end-to-end PnP check)

- [ ] **Step 1: Replace `test_identify_propagates_from_prior_with_hashes`** and add a PnP round-trip. New tests (keep `test_identify_without_prior_returns_empty`, the seed tests, line_x_at/merge/fit tests):

```python
def test_identify_emits_two_hash_correspondences_per_line():
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.utils.meta import CalibHint
    # 3 vertical yard lines at x=400/800/1200; two dense tick rows at y=360/620.
    xs = [400, 800, 1200]
    hashes = []
    for x in range(200, 1400, 20):
        hashes += [(float(x), 360.0), (float(x), 620.0)]
    feats = DetectedFeatures(
        yard_lines=[YardLineSeg((float(x), 0.0), (float(x), 1080.0)) for x in xs],
        sidelines=[], hashes=hashes, numbers=[], image_size=(1920, 1080))
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    corrs, _ = identify_correspondences(feats, state)
    names = [c[0] for c in corrs]
    # away_30 line (x=800) → both hash rows, at the true intersections (800, 360/620)
    pt_by_name = dict(corrs)
    assert "away_30_left_hash" in names and "away_30_right_hash" in names
    lu = pt_by_name["away_30_left_hash"]; ld = pt_by_name["away_30_right_hash"]
    assert abs(lu[0] - 800) < 2 and abs(ld[0] - 800) < 2
    assert {round(lu[1]), round(ld[1])} == {360, 620}
    # exactly 2 hash corrs per line, no off-line spurious points
    assert len([n for n in names if n.endswith("_hash")]) == 6


def test_identify_pnp_roundtrip_under_2px():
    import numpy as np
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_correspondences
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points
    from nfl_gsplat.utils.meta import CalibHint

    # A known camera viewing the field; project the away 20/25/30 hash + sideline
    # world points to build synthetic detected lines + hash ticks, then verify the
    # recovered camera matches.
    intr = CameraIntrinsics(1100.0, 1100.0, 960, 540, 1920, 1080)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    pose = CameraPose(R=R, t=np.array([0.0, 5.0, 30.0]))

    def proj(name):
        return project_points(NFL_LANDMARKS[name][None], intr.K(), pose.R, pose.t)[0]

    # Build a yard line per yard as a 2-pt segment through its two hash projections.
    yards = [20, 25, 30, 35, 40]
    lines, hashes = [], []
    for y in yards:
        lh = proj(f"away_{y}_left_hash"); rh = proj(f"away_{y}_right_hash")
        lines.append(YardLineSeg((float(lh[0]), float(lh[1])), (float(rh[0]), float(rh[1]))))
        hashes += [(float(lh[0]), float(lh[1])), (float(rh[0]), float(rh[1]))]
    feats = DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                             numbers=[], image_size=(1920, 1080))
    # seed at the away_30 line's mid-height x
    from nfl_gsplat.calibration.field_identify import line_x_at
    ref_x = line_x_at(lines[2], 540.0)
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    corrs, _ = identify_correspondences(feats, state)
    assert len(corrs) >= 6
    res = solve_pnp_from_correspondences(corrs, image_size=(1920, 1080), max_reproj_px=1e9)
    assert res.rms_px < 2.0
```

(Note on the round-trip: each yard-line segment is built through its two hash
points, so its slope matches the real line; `fit_hash_rows` recovers the two rows
from all 10 hash points; intersections reproduce the exact projected points →
PnP recovers the camera. If `increasing`/side direction is off for this synthetic
camera, flip them in the test until it solves < 2 px — the geometry, not the
solver, is under test.)

- [ ] **Step 2: Run** → FAIL (old per-tick code still there; new intersection emission missing).

- [ ] **Step 3: Rewrite `identify_correspondences`** in `field_identify.py`:

```python
def identify_correspondences(feats, prior):
    """Propagate yard-line identity from ``prior`` (nearest x@mid-height) and emit
    [(landmark_name, (u,v))] at yard-line × hash-row and yard-line × sideline
    intersections. With no prior, returns ([], empty)."""
    import numpy as np

    mid = feats.image_size[1] / 2.0
    lines = _merge_lines(feats.yard_lines, tol=25.0, ref_y=mid)
    if not lines or prior is None or not prior.line_yardage:
        return [], IdentityState()
    prior_xs = np.array(list(prior.line_yardage.keys()))
    prior_vals = list(prior.line_yardage.values())

    rows = fit_hash_rows(feats.hashes, image_width=feats.image_size[0])
    # rows[0] = upper (world +Y = 'left'), rows[1] = lower ('right')
    row_lr = ["left", "right"]

    corrs: list[tuple[str, tuple[float, float]]] = []
    state_map: dict[float, tuple[str, int]] = {}
    W, H = feats.image_size
    for seg in lines:
        x = line_x_at(seg, mid)
        j = int(np.argmin(np.abs(prior_xs - x)))
        if abs(prior_xs[j] - x) > 60.0:
            continue
        side, yd = prior_vals[j]
        state_map[x] = (side, yd)
        # hash-row intersections
        for ri, row in enumerate(rows[:2]):
            pt = _seg_intersection(seg, row)
            if pt is None or not (0 <= pt[0] <= W and 0 <= pt[1] <= H):
                continue
            corrs.append((landmark_name(side, yd, row_lr[ri], "hash"), pt))
        # sideline intersections (when present)
        for sl in feats.sidelines:
            pt = _seg_intersection(seg, sl)
            if pt is None or not (0 <= pt[0] <= W and 0 <= pt[1] <= H):
                continue
            lr = "left" if pt[1] < mid else "right"
            corrs.append((landmark_name(side, yd, lr, "sideline"), pt))
    seen: set[str] = set()
    deduped = []
    for name, uv in corrs:
        if name not in seen:
            seen.add(name)
            deduped.append((name, uv))
    return deduped, IdentityState(line_yardage=state_map)
```

Update the module docstring's step 4 to say "intersect with the two fitted
hash-row lines (and sidelines when present)".

- [ ] **Step 4: Run** `python -m pytest tests/test_field_identify.py -q` → all pass (line_x_at, merge, fit, the two new identify tests, seed tests, no-prior test). Debug the PnP round-trip / direction if needed (flip `increasing`/`side` in the test to match the synthetic camera; the implementation must produce exact intersections).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git add nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git commit -m "field_identify: correspondences via line x hash-row intersections (drop per-tick)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Full suite + diagnostic note

**Files:**
- Modify: `scripts/diag_calib.py` (one print line)

- [ ] **Step 1: Run the full suite** — `python -m pytest -m "not gpu and not slow and not real_video" -q`. Expect all pass (the `field_identify` changes are internal; `run_autocalib`/`register_frame` consume the same `identify_correspondences` signature, unchanged). A torch `c10.dll` OSError on `test_pipeline_smoke` is a known local-only issue — confirm it's the only failure if any.

- [ ] **Step 2: Add a hint to the diagnostic's PnP output** — in `scripts/diag_calib.py`, after the `print(f"hashes detected: ...")` line, add a fitted-rows count so bring-up sees the rows:

```python
        from nfl_gsplat.calibration.field_identify import fit_hash_rows
        _rows = fit_hash_rows(feats.hashes, image_width=img.shape[1])
        print(f"  hash rows fitted: {len(_rows)}")
```

Confirm it parses: `python -c "import ast; ast.parse(open('scripts/diag_calib.py').read())"`.

- [ ] **Step 3: Lint + commit**

```bash
python -m ruff check nfl_gsplat scripts tests
git add -A
git commit -m "diag_calib: report fitted hash-row count for bring-up

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** A (order/merge by x@mid) → Tasks 1-2; B (RANSAC two rows) → Task 3; C (line×row intersection correspondences) → Task 4; failure handling (<2 rows → sidelines only; out-of-image skipped) → Task 4 guards; testing (line_x_at, merge, fit, end-to-end PnP < 2px) → Tasks 1-4; real-frame validation → Task 5 diagnostic. `seed_state_from_hint` reorder → Task 2. Downstream unchanged (no tasks needed) — verified by the full-suite step in Task 5.
- **Type consistency:** `line_x_at(seg, y) -> float`, `_merge_lines(lines, tol, ref_y) -> list`, `fit_hash_rows(hashes, *, image_width, inlier_px, min_inliers, iters) -> list[YardLineSeg]`, `identify_correspondences(feats, prior) -> (corrs, IdentityState)` consistent across Tasks 1-5. `IdentityState.line_yardage` keyed by `line_x_at(·, mid)` consistently in seed (Task 2) and identify (Task 4). `YardLineSeg(p0, p1)` from field_features used throughout.
- **Placeholder scan:** the Task 2 Step 1 test has a first draft with a placeholder lambda that is immediately replaced by the corrected version below it — the implementer uses the corrected `test_merge_lines_dedupes_same_diagonal_line`. No other placeholders.

## Known follow-ups (bring-up)
- Re-run `diag_calib --mask` + hint on real frame 0; expect 2 clean hash points per line and RMS → single digits. Tune `inlier_px`/`merge_tol`/`min_inliers` against the real ticks if needed.
- If upper/lower → left/right is mirrored for this camera, flip in the hint (`side`/`increasing`) — RMS validates.
