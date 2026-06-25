# Homography-Consensus Line Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace index-based line→yardage labeling (which mislabels lines when spurious detections corrupt the 5-yard spacing) with a deterministic homography-consensus labeler that rejects noise lines and produces geometrically consistent correspondences + a field homography.

**Architecture:** New module `nfl_gsplat/calibration/field_homography.py` holds `fit_plane_homography` + `label_lines_by_consensus` (enumerated hypotheses anchored by the hint, scored by inlier consensus). `field_identify.py`'s `IdentityState` carries the homography/anchor/direction; `identify_correspondences` calls the labeler. Output stays `(corrs, IdentityState)` so `register_frame`/`run_autocalib`/`solve_pnp` are untouched.

**Tech Stack:** Python 3.10, numpy, OpenCV (`findHomography`/`perspectiveTransform`), pytest. Pure/deterministic — no randomness.

**Reference spec:** `docs/superpowers/specs/2026-06-23-homography-consensus-labeling-design.md`

## Global Constraints
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Repo root: `C:/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/NFLGSPLAT`. Local `python -m pytest`.
- Field constants (from `field_landmarks`): `YARD_LINE_SPACING_M = 4.572`, `HASH_OFFSET_M = 2.8194`. World +X toward home endzone; `_yardline_x_m("away_30") = -(50-30)*0.9144 = -18.288`. `_yard_step(side, yard, +1)` moves +1 yard-line toward home (= +`YARD_LINE_SPACING_M` in world X), consistent with `_yardline_x_m`.
- `IdentityState` stays a `frozen` dataclass; all fields set at construction.

## Current state
`field_identify.py` has: `IdentityState(line_yardage)`, `line_x_at(seg, y)`, `_merge_lines(lines, tol, ref_y)`, `_ransac_line`, `fit_hash_rows(hashes, *, image_width, ...)`, `_seg_intersection(a, b)`, `_yard_step(side, yard, step) -> (side, yard)` (returns `("", 0)` off-field), `seed_state_from_hint(feats, hint)`, `identify_correspondences(feats, prior)`. `YardLineSeg(p0, p1)` + `landmark_name(side, yd, lr, row)` in `field_features`. `_yardline_x_m(name)` in `field_landmarks`.

---

## Task 1: `field_homography.py` — `fit_plane_homography` + `LabelResult`

**Files:**
- Create: `nfl_gsplat/calibration/field_homography.py`
- Test: `tests/test_field_homography.py` (create)

**Interfaces:**
- Produces: `LabelResult(correspondences: list[tuple[str, tuple[float,float]]], homography: np.ndarray | None, inlier_count: int)` (frozen dataclass); `fit_plane_homography(world_xy, image_uv) -> np.ndarray | None`.

- [ ] **Step 1: Write the failing test** `tests/test_field_homography.py`:

```python
import numpy as np

from nfl_gsplat.calibration.field_homography import LabelResult, fit_plane_homography


def test_fit_plane_homography_recovers_known_H():
    import cv2
    # Build a real perspective H from 4 field corners → 4 image points.
    world = np.array([[-20.0, 3.0], [20.0, 3.0], [20.0, -3.0], [-20.0, -3.0]])
    image = np.array([[300.0, 200.0], [1600.0, 220.0], [1500.0, 850.0], [400.0, 870.0]])
    H = fit_plane_homography(world, image)
    assert H is not None
    proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.allclose(proj, image, atol=1e-6)


def test_fit_plane_homography_too_few_points_returns_none():
    assert fit_plane_homography(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_label_result_is_frozen_dataclass():
    r = LabelResult(correspondences=[], homography=None, inlier_count=0)
    assert r.inlier_count == 0 and r.correspondences == [] and r.homography is None
```

- [ ] **Step 2: Run** `python -m pytest tests/test_field_homography.py -q` → FAIL (module missing).

- [ ] **Step 3: Create `nfl_gsplat/calibration/field_homography.py`:**

```python
"""Robust field-line labeling via planar-homography consensus.

The per-camera hint pins one detected yard line's absolute yardage (the anchor).
Real yard lines are monotonic in world-X; the two hash rows are world Y = ±HASH.
A correct labeling makes every (yard-line × hash-row) point consistent with one
field→image homography; spurious lines (painted numbers, jersey scraps) fit no
consensus and are rejected. Hypotheses are enumerated deterministically — the
space is tiny, so no randomness is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_gsplat.calibration.field_features import landmark_name
from nfl_gsplat.calibration.field_identify import _seg_intersection, _yard_step, line_x_at
from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, YARD_LINE_SPACING_M


@dataclass(frozen=True)
class LabelResult:
    correspondences: list[tuple[str, tuple[float, float]]]
    homography: np.ndarray | None
    inlier_count: int


def fit_plane_homography(world_xy, image_uv) -> np.ndarray | None:
    """Least-squares field→image homography (3×3) from ≥4 point pairs, else None."""
    import cv2
    world = np.asarray(world_xy, dtype=np.float64)
    image = np.asarray(image_uv, dtype=np.float64)
    if len(world) < 4 or len(image) != len(world):
        return None
    H, _ = cv2.findHomography(world, image, 0)
    return H
```

- [ ] **Step 4: Run** `python -m pytest tests/test_field_homography.py -q` → PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_homography.py tests/test_field_homography.py
git add nfl_gsplat/calibration/field_homography.py tests/test_field_homography.py
git commit -m "field_homography: fit_plane_homography + LabelResult

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `label_lines_by_consensus`

**Files:**
- Modify: `nfl_gsplat/calibration/field_homography.py`
- Test: `tests/test_field_homography.py` (extend)

**Interfaces:**
- Consumes: `fit_plane_homography`, `LabelResult` (Task 1); `_seg_intersection`, `line_x_at`, `_yard_step` (field_identify); `landmark_name` (field_features); `YardLineSeg` (field_features); `HASH_OFFSET_M`, `YARD_LINE_SPACING_M`.
- Produces: `label_lines_by_consensus(lines, hash_rows, *, anchor_idx, anchor_world_x, anchor_side, anchor_yard, direction, image_size, inlier_px=5.0, max_offset=12) -> LabelResult`. `direction` ∈ {+1, −1}: +1 means image-right ⇒ +1 yard-line step (`_yard_step` toward home); −1 mirrors it. Resolves the X-mirror ambiguity (the hint's `increasing`).

- [ ] **Step 1: Write the failing test** (append to `tests/test_field_homography.py`):

```python
def _known_H():
    import cv2
    world = np.array([[-25.0, 3.0], [25.0, 3.0], [25.0, -3.0], [-25.0, -3.0]])
    image = np.array([[260.0, 180.0], [1660.0, 210.0], [1520.0, 900.0], [380.0, 930.0]])
    return cv2.getPerspectiveTransform(world.astype(np.float32), image.astype(np.float32))


def _project(H, X, Y):
    import cv2
    p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
    return (float(p[0]), float(p[1]))


def test_label_consensus_recovers_real_lines_rejects_noise():
    from nfl_gsplat.calibration.field_features import YardLineSeg
    from nfl_gsplat.calibration.field_homography import label_lines_by_consensus
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, _yardline_x_m

    H = _known_H()
    yards = [15, 20, 25, 30, 35, 40, 45]                 # 7 true away yard lines
    lines = []
    for y in yards:
        X = _yardline_x_m(f"away_{y}")
        top = _project(H, X, +HASH_OFFSET_M); bot = _project(H, X, -HASH_OFFSET_M)
        lines.append(YardLineSeg(top, bot))
    # 4 spurious lines: vertical segments at arbitrary x spanning the hash band.
    for sx in (520.0, 905.0, 1210.0, 1602.0):
        lines.append(YardLineSeg((sx, 150.0), (sx + 3.0, 950.0)))
    # Hash rows: width-spanning segments through the projected +H / -H bands.
    xL, xR = _yardline_x_m("away_15"), _yardline_x_m("away_45")
    row_top = YardLineSeg(_project(H, xL, +HASH_OFFSET_M), _project(H, xR, +HASH_OFFSET_M))
    row_bot = YardLineSeg(_project(H, xL, -HASH_OFFSET_M), _project(H, xR, -HASH_OFFSET_M))

    anchor_idx = yards.index(30)                          # away_30 line
    res = label_lines_by_consensus(
        lines, [row_top, row_bot], anchor_idx=anchor_idx,
        anchor_world_x=_yardline_x_m("away_30"), anchor_side="away", anchor_yard=30,
        direction=+1, image_size=(1920, 1080))
    assert res.inlier_count == 7                          # all 7 real, 0 spurious
    names = {n for n, _ in res.correspondences}
    for y in yards:
        assert f"away_{y}_left_hash" in names and f"away_{y}_right_hash" in names
    # refit homography reprojects an inlier point cleanly
    import cv2
    pj = cv2.perspectiveTransform(
        np.array([[[_yardline_x_m("away_30"), +HASH_OFFSET_M]]], np.float64),
        res.homography).reshape(2)
    true_top = _project(H, _yardline_x_m("away_30"), +HASH_OFFSET_M)
    assert np.linalg.norm(pj - np.array(true_top)) < 2.0


def test_label_consensus_too_few_lines_empty():
    from nfl_gsplat.calibration.field_features import YardLineSeg
    from nfl_gsplat.calibration.field_homography import label_lines_by_consensus
    one = [YardLineSeg((800.0, 0.0), (800.0, 1080.0))]
    rows = [YardLineSeg((0, 300), (1920, 300)), YardLineSeg((0, 600), (1920, 600))]
    res = label_lines_by_consensus(one, rows, anchor_idx=0, anchor_world_x=0.0,
                                   anchor_side="mid", anchor_yard=50, direction=+1,
                                   image_size=(1920, 1080))
    assert res.inlier_count == 0 and res.correspondences == []
```

- [ ] **Step 2: Run** → FAIL (`label_lines_by_consensus` missing).

- [ ] **Step 3: Append the implementation** to `field_homography.py`:

```python
def _in_image(p, W, H) -> bool:
    return p is not None and 0 <= p[0] <= W and 0 <= p[1] <= H


def label_lines_by_consensus(
    lines, hash_rows, *, anchor_idx, anchor_world_x, anchor_side, anchor_yard,
    direction, image_size, inlier_px: float = 5.0, max_offset: int = 12,
) -> LabelResult:
    """Label detected yard lines by homography consensus; reject noise lines.

    Each kept line contributes 2 field↔image points (its intersections with the
    two hash rows at world Y = ±HASH_OFFSET_M). Enumerate hypotheses (anchor +
    one other line at a signed yard-line offset), fit a homography, score by how
    many kept lines map onto valid yard-line positions, keep the best, refit, and
    emit (landmark_name, uv) correspondences. ``direction`` (+1/−1) pins the
    image-x→yard orientation (the hint's ``increasing``)."""
    import cv2
    W, H = image_size
    if len(hash_rows) < 2:
        return LabelResult([], None, 0)
    row_top, row_bot = hash_rows[0], hash_rows[1]
    mid = H / 2.0

    # Pre-filter: keep lines crossing BOTH rows in-image; record x@mid + the 2 pts.
    kept = []  # (x_mid, p_top, p_bot)
    for seg in lines:
        pt = _seg_intersection(seg, row_top)
        pb = _seg_intersection(seg, row_bot)
        if _in_image(pt, W, H) and _in_image(pb, W, H):
            kept.append((line_x_at(seg, mid), pt, pb))
    if len(kept) < 2:
        return LabelResult([], None, 0)
    kept.sort(key=lambda r: r[0])

    # Re-find the anchor among kept lines (nearest x@mid to the original anchor line).
    anchor_x0 = line_x_at(lines[anchor_idx], mid)
    a = min(range(len(kept)), key=lambda i: abs(kept[i][0] - anchor_x0))
    ax, a_top, a_bot = kept[a]
    Hp = HASH_OFFSET_M

    def homog_for(world_x_j, j):
        world = np.array([[anchor_world_x, Hp], [anchor_world_x, -Hp],
                          [world_x_j, Hp], [world_x_j, -Hp]], np.float64)
        image = np.array([a_top, a_bot, kept[j][1], kept[j][2]], np.float64)
        return fit_plane_homography(world, image)

    def score(Hm):
        """Return list of (kept_index, k_offset, p_top, p_bot, resid) inliers."""
        try:
            Hinv = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            return []
        inliers = []
        for i, (xm, pt, pb) in enumerate(kept):
            img = np.array([[pt], [pb]], np.float64)
            fld = cv2.perspectiveTransform(img, Hinv).reshape(2, 2)
            avg_x = 0.5 * (fld[0, 0] + fld[1, 0])
            k = int(round((avg_x - anchor_world_x) / YARD_LINE_SPACING_M))
            if abs(k) > max_offset:
                continue
            x_snap = anchor_world_x + k * YARD_LINE_SPACING_M
            snap = np.array([[[x_snap, Hp]], [[x_snap, -Hp]]], np.float64)
            rep = cv2.perspectiveTransform(snap, Hm).reshape(2, 2)
            resid = max(np.linalg.norm(rep[0] - pt), np.linalg.norm(rep[1] - pb))
            if resid <= inlier_px:
                inliers.append((i, k, pt, pb, resid))
        return inliers

    best = []
    best_key = (-1, float("inf"))
    for j in range(len(kept)):
        if j == a:
            continue
        # direction pins the sign: image-right of anchor ⇒ +direction yard steps.
        side = 1 if kept[j][0] > ax else -1
        for d in range(1, max_offset + 1):
            k = direction * side * d
            world_x_j = anchor_world_x + k * YARD_LINE_SPACING_M
            Hm = homog_for(world_x_j, j)
            if Hm is None:
                continue
            inl = score(Hm)
            total = sum(r[4] for r in inl)
            key = (len(inl), -total)
            if key > (best_key[0], -best_key[1]):
                best, best_key = inl, (len(inl), total)
    if len(best) < 2:
        return LabelResult([], None, 0)

    # Refit on all inliers, emit correspondences.
    wpts, ipts, labels = [], [], []
    for (_i, k, pt, pb, _r) in best:
        side, yard = _yard_step(anchor_side, anchor_yard, k)
        if side == "":
            continue
        x_snap = anchor_world_x + k * YARD_LINE_SPACING_M
        wpts += [[x_snap, Hp], [x_snap, -Hp]]
        ipts += [pt, pb]
        labels.append((side, yard, pt, pb))
    H_refit = fit_plane_homography(np.array(wpts), np.array(ipts))
    corrs, seen = [], set()
    for (side, yard, pt, pb) in labels:
        for lr, p in (("left", pt), ("right", pb)):
            name = landmark_name(side, yard, lr, "hash")
            if name not in seen:
                seen.add(name)
                corrs.append((name, (float(p[0]), float(p[1]))))
    return LabelResult(corrs, H_refit, len(labels))
```

- [ ] **Step 4: Run** `python -m pytest tests/test_field_homography.py -q` → PASS (5 tests). Debug notes: the 7 real lines must all become inliers (residual ~0 under the true H) and the 4 vertical spurious lines must map to non-integer yard offsets or land off a valid position → rejected. If a spurious line happens to be an inlier, it means a coincidental fit — verify `inlier_px=5.0` and the spurious x's aren't accidentally on a yard line. Don't loosen the `== 7` assertion.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_homography.py tests/test_field_homography.py
git add nfl_gsplat/calibration/field_homography.py tests/test_field_homography.py
git commit -m "field_homography: label_lines_by_consensus (enumerated hypotheses + inlier consensus)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Integrate into `field_identify` (`IdentityState` + seed + identify)

**Files:**
- Modify: `nfl_gsplat/calibration/field_identify.py`
- Test: `tests/test_field_identify.py` (update)

**Interfaces:**
- Consumes: `label_lines_by_consensus`, `LabelResult` (Task 2); `_yardline_x_m` (field_landmarks); `yardline_label` (field_features).
- Produces: unchanged public signatures `seed_state_from_hint(feats, hint) -> IdentityState`, `identify_correspondences(feats, prior) -> (list[(name,(u,v))], IdentityState)`. `IdentityState` gains `homography`, `anchor_label`, `anchor_x`, `direction` (all default None/0).

- [ ] **Step 1: Update tests in `tests/test_field_identify.py`.** Replace `test_identify_emits_two_hash_correspondences_per_line`, `test_identify_propagates_identity_across_shifted_frame`, and `test_identify_pnp_roundtrip_under_2px` bodies to assert via the new path (the seed tests `test_seed_*`, `test_line_x_at_*`, `test_merge_lines_*`, `test_fit_hash_rows_*`, `test_identify_without_prior_returns_empty`, `test_identify_skips_hashes_with_single_row` stay). New/updated:

```python
def test_identify_emits_hash_correspondences_via_consensus():
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import _yardline_x_m
    from nfl_gsplat.utils.meta import CalibHint
    import cv2
    import numpy as np
    # Build a perspective H, project away_20..away_40 to make consistent lines+rows.
    world = np.array([[-25.0, 3.0], [25.0, 3.0], [25.0, -3.0], [-25.0, -3.0]], np.float32)
    image = np.array([[260, 180], [1660, 210], [1520, 900], [380, 930]], np.float32)
    H = cv2.getPerspectiveTransform(world, image)

    def proj(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
        return (float(p[0]), float(p[1]))
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M
    yards = [20, 25, 30, 35, 40]
    lines, hashes = [], []
    for y in yards:
        X = _yardline_x_m(f"away_{y}")
        t = proj(X, +HASH_OFFSET_M); b = proj(X, -HASH_OFFSET_M)
        lines.append(YardLineSeg(t, b)); hashes += [t, b]
    feats = DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                             numbers=[], image_size=(1920, 1080))
    ref_x = 0.5 * (lines[2].p0[0] + lines[2].p1[0])     # away_30 line ~ x
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    state0 = seed_state_from_hint(feats, hint)
    corrs, state = identify_correspondences(feats, state0)
    names = {n for n, _ in corrs}
    assert "away_30_left_hash" in names and "away_30_right_hash" in names
    assert state.homography is not None


def test_identify_pnp_roundtrip_under_2px():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, _yardline_x_m
    from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_correspondences
    from nfl_gsplat.utils.meta import CalibHint
    world = np.array([[-25.0, 3.0], [25.0, 3.0], [25.0, -3.0], [-25.0, -3.0]], np.float32)
    image = np.array([[260, 180], [1660, 210], [1520, 900], [380, 930]], np.float32)
    H = cv2.getPerspectiveTransform(world, image)

    def proj(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
        return (float(p[0]), float(p[1]))
    yards = [15, 20, 25, 30, 35, 40, 45]
    lines, hashes = [], []
    for y in yards:
        X = _yardline_x_m(f"away_{y}")
        t = proj(X, +HASH_OFFSET_M); b = proj(X, -HASH_OFFSET_M)
        lines.append(YardLineSeg(t, b)); hashes += [t, b]
    # inject 3 spurious lines
    for sx in (505.0, 1005.0, 1505.0):
        lines.append(YardLineSeg((sx, 150.0), (sx + 2.0, 950.0)))
    feats = DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                             numbers=[], image_size=(1920, 1080))
    ref_x = 0.5 * (lines[3].p0[0] + lines[3].p1[0])
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    state0 = seed_state_from_hint(feats, hint)
    corrs, _ = identify_correspondences(feats, state0)
    assert len(corrs) >= 6
    res = solve_pnp_from_correspondences(corrs, image_size=(1920, 1080), max_reproj_px=1e9)
    assert res.rms_px < 2.0


def test_identify_propagates_via_prior_homography():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, _yardline_x_m
    from nfl_gsplat.utils.meta import CalibHint
    world = np.array([[-25.0, 3.0], [25.0, 3.0], [25.0, -3.0], [-25.0, -3.0]], np.float32)

    def make(image):
        H = cv2.getPerspectiveTransform(world, image.astype(np.float32))
        lines, hashes = [], []
        for y in [20, 25, 30, 35, 40]:
            X = _yardline_x_m(f"away_{y}")
            p = cv2.perspectiveTransform(np.array([[[X, +HASH_OFFSET_M]]], np.float64), H).reshape(2)
            q = cv2.perspectiveTransform(np.array([[[X, -HASH_OFFSET_M]]], np.float64), H).reshape(2)
            t = (float(p[0]), float(p[1])); b = (float(q[0]), float(q[1]))
            lines.append(YardLineSeg(t, b)); hashes += [t, b]
        return DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                                numbers=[], image_size=(1920, 1080))
    f0 = make(np.array([[260, 180], [1660, 210], [1520, 900], [380, 930]]))
    f1 = make(np.array([[280, 180], [1680, 210], [1540, 900], [400, 930]]))   # panned
    ref_x = 0.5 * (f0.yard_lines[2].p0[0] + f0.yard_lines[2].p1[0])
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    _, prior = identify_correspondences(f0, seed_state_from_hint(f0, hint))
    assert prior.homography is not None
    corrs, _ = identify_correspondences(f1, prior)            # anchored via prior H, no hint
    names = {n for n, _ in corrs}
    assert "away_30_left_hash" in names
```

- [ ] **Step 2: Run** `python -m pytest tests/test_field_identify.py -q` → the updated tests FAIL.

- [ ] **Step 3: Edit `field_identify.py`.** Extend `IdentityState`; add a `_yardline_name` helper; rewrite `seed_state_from_hint` + `identify_correspondences`.

Replace the `IdentityState` class:
```python
@dataclass(frozen=True)
class IdentityState:
    line_yardage: dict[float, tuple[str, int]] = field(default_factory=dict)
    homography: "np.ndarray | None" = None
    anchor_label: "tuple[str, int] | None" = None
    anchor_x: "float | None" = None
    direction: int = 0
```
Add a helper (near `_yard_step`):
```python
def _yardline_name(side: str, yard: int) -> str:
    return "mid_50" if side == "mid" else f"{side}_{yard}"
```
Rewrite `seed_state_from_hint`:
```python
def seed_state_from_hint(feats, hint) -> IdentityState:
    """Ref-frame seed: record the hint anchor (side/yard + image-x) and direction.
    The actual labeling happens in identify_correspondences (consensus)."""
    from nfl_gsplat.calibration.field_features import yardline_label
    side, yard = yardline_label(hint.side, hint.yard)
    direction = 1 if hint.increasing == "right" else -1
    return IdentityState(anchor_label=(side, yard), anchor_x=float(hint.ref_x),
                         direction=direction)
```
Rewrite `identify_correspondences`:
```python
def identify_correspondences(feats, prior):
    """Label this frame's yard lines by homography consensus (anchored by the hint
    at the ref frame, by prior.homography afterwards) and emit hash correspondences.
    Returns (correspondences, IdentityState carrying the homography)."""
    import cv2
    import numpy as np

    from nfl_gsplat.calibration.field_homography import label_lines_by_consensus
    from nfl_gsplat.calibration.field_landmarks import _yardline_x_m

    mid = feats.image_size[1] / 2.0
    lines = _merge_lines(feats.yard_lines, tol=25.0, ref_y=mid)
    rows = fit_hash_rows(feats.hashes, image_width=feats.image_size[0])
    if (len(lines) < 2 or len(rows) < 2 or prior is None
            or prior.anchor_label is None):
        return [], IdentityState()

    side, yard = prior.anchor_label
    anchor_world_x = _yardline_x_m(_yardline_name(side, yard))
    # Predict the anchor's image-x: via prior homography if we have one, else the
    # carried anchor_x (ref frame = hint.ref_x).
    if prior.homography is not None:
        p = cv2.perspectiveTransform(
            np.array([[[anchor_world_x, 0.0]]], np.float64), prior.homography).reshape(2)
        pred_x = float(p[0])
    else:
        pred_x = float(prior.anchor_x)
    anchor_idx = min(range(len(lines)), key=lambda i: abs(line_x_at(lines[i], mid) - pred_x))

    res = label_lines_by_consensus(
        lines, rows, anchor_idx=anchor_idx, anchor_world_x=anchor_world_x,
        anchor_side=side, anchor_yard=yard, direction=prior.direction or 1,
        image_size=feats.image_size)
    if res.homography is None or res.inlier_count < 2:
        return [], IdentityState()
    new_anchor_x = line_x_at(lines[anchor_idx], mid)
    state = IdentityState(homography=res.homography, anchor_label=(side, yard),
                          anchor_x=new_anchor_x, direction=prior.direction or 1)
    return res.correspondences, state
```
Remove any now-unused imports/helpers flagged by ruff (the old per-row emission loop is replaced; `_seg_intersection` is still used by `field_homography`, keep it).

- [ ] **Step 4: Run** `python -m pytest tests/test_field_identify.py -q` → all pass. Debug the consensus direction if the round-trip mislabels (the synthetic camera is built with away_X increasing toward +X; `increasing="right"` ⇒ direction=+1; flip the hint to `"left"` in the test only if the synthetic image-x ordering is reversed — but with the given trapezoid, away_15(−X) projects left and away_45 projects right, so right ⇒ toward midfield ⇒ +1, matching `increasing="right"`).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git add nfl_gsplat/calibration/field_identify.py tests/test_field_identify.py
git commit -m "field_identify: label via homography consensus; IdentityState carries H + anchor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Full suite + diagnostic homography readout

**Files:**
- Modify: `scripts/diag_calib.py`
- Test: full suite

- [ ] **Step 1: Full suite** — `python -m pytest -m "not gpu and not slow and not real_video" -q`. Expect all pass (the `run_autocalib` sweep + `register_frame` consume the unchanged `identify_correspondences` signature; the carried `homography`/anchor fields are additive). A torch `c10.dll` OSError on `test_pipeline_smoke` is known local-only — confirm it is the only failure if any.

- [ ] **Step 2: Diagnostic** — the existing `scripts/diag_calib.py` already fits hash rows and prints a `HOMOGRAPHY:` residual from the emitted correspondences. Confirm the emitted `corrs` now come from consensus (no code change needed — it calls `identify_correspondences`). Verify it still parses: `python -c "import ast; ast.parse(open('scripts/diag_calib.py').read())"`.

- [ ] **Step 3: Lint + commit (if any change)**

```bash
python -m ruff check nfl_gsplat scripts tests
git add -A
git commit -m "homography-consensus labeling: full-suite green

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(If Step 2 needs no diagnostic edit and the suite is green with everything already committed in Tasks 1-3, skip the empty commit.)

---

## Self-Review (completed during planning)

- **Spec coverage:** new `field_homography.py` (`fit_plane_homography` → Task 1; `label_lines_by_consensus` with pre-filter, enumerated hypotheses, consensus scoring, refit, emit → Task 2); `IdentityState` gains homography/anchor_label/anchor_x (+direction, an implementation necessity for the X-mirror) → Task 3; `seed_state_from_hint` records the anchor, `identify_correspondences` anchors via hint (ref) / prior-H (later) and calls the labeler → Task 3; failure handling (<2 rows / <2 inliers → empty → gap) → Tasks 2-3; testing (known-H recovery, injected-noise rejection == 7, pre-filter, PnP <2px, propagation via prior H) → Tasks 1-3; real-frame validation via existing diagnostic → Task 4. Sweep/register/solve_pnp unchanged — verified by Task 4 full suite.
- **Type consistency:** `LabelResult(correspondences, homography, inlier_count)`; `fit_plane_homography(world_xy, image_uv) -> H|None`; `label_lines_by_consensus(lines, hash_rows, *, anchor_idx, anchor_world_x, anchor_side, anchor_yard, direction, image_size, inlier_px, max_offset) -> LabelResult`; `IdentityState(line_yardage, homography, anchor_label, anchor_x, direction)`; `identify_correspondences(feats, prior) -> (corrs, IdentityState)` consistent across tasks. `direction` ∈ {+1,−1}; `_yard_step` step sign matches world-X sign (Global Constraints).
- **Placeholder scan:** none — every code/test step is complete.

## Known follow-ups (bring-up + next cycle)
- Re-run `diag_calib --mask` + hint on real frame 0; expect `HOMOGRAPHY` residual 61px → single digits, inliers ≈ count of real yard lines. Tune `inlier_px`/`max_offset` on the real lines.
- Performance: ~`len(kept)·max_offset` homography fits per frame; fine for bring-up, optimize later if the full sweep is slow.
- **Next cycle (see project memory):** extract per-frame focal / K,R,t from the homography under the near-affine telephoto degeneracy.
