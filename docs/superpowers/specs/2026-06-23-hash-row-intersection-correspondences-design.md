# Robust Hash-Row-Intersection Correspondences

**Date:** 2026-06-23
**Status:** Approved (design); implementation plan pending

## Problem

On real broadcast frames, automatic field registration produces wrong camera
solves (reprojection RMS ~154 px on the bring-up frame). Confirmed cause, from
the diagnostic's correspondence visualization on `data/2025/week_04/SEA_at_AZ/
play_001` frame 0:

1. **Yard lines are diagonal** (broadcast camera angle). The current hash matching
   uses each line's **mean image-x** (`_line_x`), but a diagonal line's mean-x
   differs from its x at the hash-row height — so hashes match to the wrong line.
2. **NFL hash marks exist at every yard**, not just the painted 5-yard lines. The
   detector returns ~79 hash points (the dense 1-yard tick rows). "Nearest tick
   within 25 px of a yard line" grabs an off-line tick (e.g. the 27-yard tick
   matched to the 25-yard line) → wrong world coordinate.
3. **Duplicate yard-line detections** (e.g. `967/993/1024/1044`) are not merged
   (current `_merge_collinear` tolerance is 18 px on mean-x), so the
   index→yardage spacing in `seed_state_from_hint` is corrupted — an extra line
   shifts every label past it by 5 yd.

The result is scattered, mislabeled correspondences feeding PnP.

## Fix

Replace per-tick hash matching with **hash-row line fitting + yard-line ×
hash-row intersection**, and order/merge yard lines by x-at-a-reference-height
(diagonal-safe). All pure geometry in `field_identify.py`; the detector keeps
returning raw hash points. Three parts:

### A. Order + merge yard lines by x-at-mid-height
Two segments of the *same* diagonal line share the same x at a given y but can
have different mean-x. Add `line_x_at(seg, y)` returning the segment's x at image
height `y` (from its slope; vertical-safe). Order lines by `line_x_at(seg, H/2)`
and **merge** lines whose mid-height x fall within `merge_tol_px` (default ~25)
into one (kills duplicate-detection clusters). `seed_state_from_hint` switches
from `_line_x` to `line_x_at(seg, H/2)` for ordering + snapping `ref_x`
(`ref_x` is read at mid-height from the diagnostic, consistent).

### B. Fit the two hash rows (RANSAC)
`fit_hash_rows(hashes, *, cfg) -> list[Seg]` (0, 1, or 2 row segments):
- RANSAC-fit a line to all hash points (sample 2, count inliers within
  `row_inlier_px`, keep best), require ≥ `min_row_inliers`.
- Remove that line's inliers; RANSAC-fit a second line to the remainder.
- Return each row as a segment spanning the image width (two endpoints at x=0 and
  x=W on the fitted line), so it intersects yard lines cleanly. RANSAC averages
  out the 1-yard ticks and noise (≈79 messy points → 2 clean lines).
- Pure numpy; deterministic seed for testability.

### C. Correspondences = intersections
In `identify_correspondences`, after labeling lines (propagation unchanged):
- Fit the two hash rows once per frame (B).
- The two rows are sorted by mean y: **upper row → `left`** (world +Y), **lower
  row → `right`** — consistent with the sideline left/right convention. (Camera-
  side assumption; a wrong choice raises RMS and is flipped at bring-up, same as
  today.)
- For each labeled yard line: intersect with each hash-row segment (reuse
  `_seg_intersection`) → `({yard}_left_hash, uv)` and `({yard}_right_hash, uv)`.
  Skip an intersection that falls outside the image bounds.
- Keep the existing yard-line × sideline intersections (when sidelines detected).
- Drop the old per-tick `for hx, hy in feats.hashes` loop entirely.

## Data flow (unchanged except inside identify)

```
detect_field_features → lines + raw hash points + sidelines
seed_state_from_hint (now orders by line_x_at(·, H/2))     → IdentityState
identify_correspondences:
    merge/order lines by mid-height x
    fit_hash_rows(hashes) → up to 2 row segments
    per labeled line: line × {row_top, row_bottom, sidelines} → [(name, uv)]
→ solve_pnp_from_correspondences  (unchanged)
```

The hint format, meta parsing, the bidirectional sweep in `run_autocalib`,
`register_frame`, `solve_pnp`, and all consumers are unchanged.

## Components

- `nfl_gsplat/calibration/field_identify.py`:
  - new `line_x_at(seg, y) -> float` (diagonal-safe x at height y; vertical-safe).
  - new `_merge_lines(lines, tol, ref_y) -> list[seg]` (dedupe by mid-height x).
  - new `fit_hash_rows(hashes, *, cfg) -> list[Seg]` (RANSAC two rows). Returns
    lightweight segments (reuse `YardLineSeg` from `field_features`).
  - `seed_state_from_hint`: order/snap by `line_x_at(·, H/2)`; merge first.
  - `identify_correspondences`: merge lines, fit rows, emit intersection
    correspondences (replaces per-tick loop).
  - A small `HashRowConfig` (or params on the functions) for `row_inlier_px`,
    `min_row_inliers`, `merge_tol_px` — defaults tuned at bring-up.
- No change to `field_detect.py` (still returns raw `hashes`), `field_features.py`,
  `run_autocalib.py`, `register_frame.py`, `solve_pnp.py`, `meta.py`.

## Failure handling

- < 2 hash rows fittable (too few hashes) → emit only sideline correspondences
  (may yield < 6 → frame is a gap, handled by `register_frame`/`assemble`).
- A yard line parallel to a row (no intersection) → skipped.
- Wrong upper/lower → left/right or wrong hint direction → high RMS → existing
  fail-loud. Unchanged self-validation.

## Testing (CPU, pure)

- `line_x_at`: vertical line → constant x; diagonal line → correct interpolated x
  at a given y; horizontal-ish guarded.
- `_merge_lines`: two segments of the same diagonal line (same x at mid-height,
  different mean-x) merge to one; genuinely distinct lines stay separate.
- `fit_hash_rows`: synthetic two rows of ticks (with extra 1-yard ticks + a few
  noise points) → two row segments at the right heights; a single sparse row →
  one segment; <min points → empty.
- `identify_correspondences` end-to-end (synthetic): 3 diagonal labeled yard
  lines + two hash rows of dense ticks → exactly 2 hash correspondences per line
  at the true intersections (within tolerance); off-line ticks do NOT create
  spurious points; a known camera's projected field reproject-checks < 2 px via
  `solve_pnp_from_correspondences`.
- Keep `pytest -m "not gpu and not slow"` green; ruff clean.

## Validation on real footage

After implementation, re-run the diagnostic on frame 0 with `--mask` + the hint
(`--ref-x 562 --yard 30 --side away --increasing left`). Expect the `_corr.png`
to show 2 clean hash points per yard line on the actual intersections and the
RMS to drop from 154 px toward single digits. Threshold tuning (`row_inlier_px`,
`merge_tol_px`) happens here.

## Out of scope

- Player-mask wiring into the pipeline (separate bring-up TODO; the diagnostic
  already masks).
- Sideline detection improvements (sidelines absent in zoomed frames; hashes are
  the primary source).
- Number OCR (abandoned).
- Any change to the hint format or the per-frame sweep.
