# Robust Line Labeling via Homography Consensus

**Date:** 2026-06-23
**Status:** Approved (design); implementation plan pending

## Problem

Hash-row-intersection correspondences are geometrically exact, but on real frame 0
only **6 of 18** correspondences fit any planar homography (median residual 61 px,
max 559 px). Cause confirmed by the diagnostic's homography sanity check: the
**line→yardage labeling is wrong**, not the focal solve. `seed_state_from_hint`
labels detected lines by index, assuming each consecutive detected line is the next
5-yard line. Real frames mix ~7 true yard lines with ~7 spurious ones (painted
number strokes like the "30"/"20" digits, jersey scraps surviving the player mask,
yard-line dashes). Each spurious line shifts every label past it, so most
correspondences get the wrong world coordinate and no camera can fit them.

The downstream 6.3e9-px PnP focal is a *symptom* of feeding inconsistent points to
PnP, not a separate bug. (Per-frame focal recovery under the near-affine telephoto
geometry is a real but **separate** problem, deferred to the next cycle.)

## Goal

Turn noisy line + hash-row detections into a **correctly-labeled, geometrically
consistent** correspondence set, robust to spurious lines, anchored in absolute
yardage by the existing per-camera hint. Success metric: the field→image
homography residual on real frame 0 drops from ~61 px to a few px, with the inlier
count ≈ the number of true yard lines. Output stays `(corrs, IdentityState)` so all
downstream code is unchanged.

**Out of scope (next cycle):** extracting per-frame focal / K,R,t from the
homography under near-affine degeneracy. This work stops at "consistent
correspondences + a verified field homography" (stored on `IdentityState`).

## Approach

A real yard line at world-X `Xᵢ` contributes two field↔image point correspondences:
`(Xᵢ, +H)` and `(Xᵢ, −H)` (H = `HASH_OFFSET_M` = 2.8194 m), whose image points are
that line's intersections with the two fitted hash rows. A correct labeling makes
all such points consistent with one field→image homography; spurious lines fit no
consensus. We find the labeling by **deterministic enumerated hypotheses + inlier
consensus** (the hypothesis space is tiny, so no randomness needed).

## New module: `nfl_gsplat/calibration/field_homography.py`

### `fit_plane_homography(world_xy, image_uv) -> np.ndarray | None`
Thin wrapper over `cv2.findHomography(world_xy, image_uv, 0)` (plain least-squares
DLT; the consensus loop handles robustness, so no inner RANSAC). Returns the 3×3 `H`
mapping field `(X, Y)` → image `(u, v)`, or `None` if degenerate (<4 points or
`cv2` returns `None`).

### `label_lines_by_consensus(...) -> LabelResult`
```
LabelResult = namedtuple/dataclass(correspondences: list[(name, (u,v))],
                                    homography: np.ndarray | None,
                                    inlier_count: int)
```
Inputs: ordered detected `lines` (already merged by `_merge_lines`), the two
`hash_rows` (from `fit_hash_rows`, upper-first), `anchor_idx` (index into `lines`)
and `anchor_world_x` (the anchor line's world-X), `image_size`, and tolerances
(`inlier_px`, `max_offset` yard-lines to search).

Algorithm:
1. **Pre-filter:** keep only lines whose intersection with *both* hash rows is
   inside the image. (`_seg_intersection` + bounds; reuses existing helper.) Build,
   for each kept line, its two image intersection points `(p_top, p_bot)` and its
   ordering x at mid-height. Re-find the anchor among the kept lines (nearest x to
   the original anchor line's x). If <2 kept lines → return empty `LabelResult`.
2. **Enumerate hypotheses:** the anchor's world-X is fixed. For every *other* kept
   line `j` and every offset `d ∈ {±1, …, ±max_offset}` (yard-LINE steps of
   `YARD_LINE_SPACING_M`), hypothesize `world_x[j] = anchor_world_x + d *
   YARD_LINE_SPACING_M`. Anchor + line `j` give 4 field↔image points
   `{(Xa,±H),(Xj,±H)} ↔ {their hash-row intersections}` → `H = fit_plane_homography`.
   Skip if `H is None`.
3. **Score by consensus:** invert `H`; map every kept line's `(p_top, p_bot)` to the
   field. A line is an inlier if both map near `Y ≈ +H` / `Y ≈ −H` and a common
   `X ≈ k * YARD_LINE_SPACING_M + anchor_world_x` (some integer `k`) within
   `inlier_px` (measured in image space by reprojecting the snapped field point
   through `H`). Count inlier lines.
4. **Winner:** keep the hypothesis with the most inliers (tie-break: lowest median
   residual). Refit `H` on all inlier correspondences (DLT over all inlier points).
5. **Emit:** for each inlier line, its snapped integer `k` → `(side, yard)` via the
   existing fold logic (`field_identify._yard_step(anchor_side, anchor_yard, k)`),
   then two correspondences `landmark_name(side, yard, "left"/"right", "hash")` at
   `p_top` / `p_bot` (upper row = left, lower = right — same convention as today).
   Return `LabelResult(correspondences, H_refit, inlier_count)`.

Pure numpy + cv2 (cv2 only for `findHomography`/`perspectiveTransform`); the
enumeration is deterministic and fully unit-testable.

## Changes to `nfl_gsplat/calibration/field_identify.py`

`IdentityState` gains three fields (all default `None`, so existing construction
still works):
- `homography: np.ndarray | None` — the field→image `H` from this frame's labeling.
- `anchor_label: tuple[str, int] | None` — the anchor line's `(side, yard)`; its
  world-X is `field_landmarks._yardline_x_m(yardline_name(side, yard))`.
- `anchor_x: float | None` — the anchor line's expected image-x (used to snap to the
  nearest detected line this frame).

`line_yardage` stays for backward compatibility but is no longer the labeling
mechanism. (`IdentityState` is currently `frozen`; keep it frozen — all fields are
set at construction.)

- `identify_correspondences(feats, prior)`:
  - Merge lines (`_merge_lines`), fit hash rows (`fit_hash_rows`). If <2 rows or
    `prior is None` or no anchor on `prior` → `([], IdentityState())` (we can't form
    the ±H point pairs without two rows + an anchor).
  - **Anchor selection:** `anchor_label` carries `(side, yard)` from `prior`;
    `anchor_world_x = _yardline_x_m(yardline_name(side, yard))`. If
    `prior.homography` is set, map `(anchor_world_x, 0)` through it to predict the
    anchor's image-x; else use `prior.anchor_x`. Snap that x to the nearest merged
    line → `anchor_idx`.
  - Call `label_lines_by_consensus(lines, rows, anchor_idx, anchor_world_x, …)`;
    return its `correspondences` and a new `IdentityState(homography=H_refit,
    anchor_label=(side, yard), anchor_x=<anchor line's x this frame>,
    line_yardage=…)`.
- **Seeding the ref frame:** `seed_state_from_hint(feats, hint)` returns
  `IdentityState(anchor_label=(hint.side, hint.yard), anchor_x=hint.ref_x,
  homography=None, line_yardage=…)`. With `homography=None`, the first
  `identify_correspondences` snaps `anchor_x` (= `hint.ref_x`) to the nearest merged
  line — exactly today's hint-snap behavior — then runs the consensus labeling.

The sweep in `run_autocalib` (seed at ref, propagate fwd/back) is unchanged: it
already threads `prior` through `register_frame`; the `prior` now additionally
carries `homography`, used for anchoring. `register_frame`/`solve_pnp` unchanged.

## Reused constants/helpers

`field_landmarks`: `YARD_LINE_SPACING_M` (4.572), `HASH_OFFSET_M` (2.8194),
`_yardline_x_m(name)`. `field_features`: `landmark_name`, `yardline_label`.
`field_identify`: `_merge_lines`, `fit_hash_rows`, `line_x_at`, `_seg_intersection`,
`_yard_step`.

## Failure handling

- <2 kept lines after pre-filter, or best hypothesis <2 inlier lines (→ <4 points)
  → empty correspondences → frame is a gap (existing `register_frame`/`assemble`
  fail-loud). A wrong hint produces no consensus → gaps → `CalibrationError`.
- The homography residual on the inliers is the built-in self-check; a future
  guard could reject a winning hypothesis whose residual exceeds a threshold.

## Testing (CPU, pure)

- `fit_plane_homography`: 4+ exact points → recovers a known `H` (reproject <1e-6);
  <4 points → `None`.
- `label_lines_by_consensus` core: project NFL landmarks through a known homography
  for 7 true yard lines (away_15..away_45) to build lines + two hash rows; **inject
  4 spurious lines** (arbitrary x crossing both rows). Assert: exactly the 7 true
  lines are inliers, each labeled with correct `(side, yard)`; all 4 spurious lines
  rejected; refit `H` reprojects inliers <2 px; `inlier_count == 7`.
- Pre-filter: a "line" that crosses only one row (a number stroke) is dropped.
- `identify_correspondences` end-to-end with injected noise → only consistent
  correspondences emitted; PnP/homography residual <2 px. Carries `homography` on
  the returned state.
- Propagation: a shifted frame anchored via `prior.homography` (no hint) → same
  labels recovered.
- Keep `pytest -m "not gpu and not slow and not real_video"` green; ruff clean.

## Validation on real footage

Re-run `diag_calib --mask` + the hint on frame 0. Expect `HOMOGRAPHY:
inliers≈7/…, median_resid` a few px (down from 61). Tune `inlier_px` / `max_offset`
on the real lines if needed.

## Out of scope

- Per-frame focal / K,R,t extraction from the homography (next cycle; see project
  memory).
- Sideline-based correspondences (sidelines absent in zoomed frames).
- Changing the hint format, the sweep, `solve_pnp`, or `register_frame`.
- Multi-frame joint focal estimation.
