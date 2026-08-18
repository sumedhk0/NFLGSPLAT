# Endzone Calibration via Static Mosaic

**Date:** 2026-08-17
**Status:** Approved (design); implementation plan pending
**Supersedes (approach):** the player-correspondence approaches —
[2026-07-15 cross-camera](2026-07-15-cross-camera-endzone-calibration-design.md),
[2026-07-17 roll fix](2026-07-17-cross-camera-bootstrap-roll-fix-design.md),
[2026-07-22 jersey identity](2026-07-22-jersey-identity-endzone-calibration-design.md).
Those stay merged (the identity work is reused for labeling), but they are no
longer the precision path.

## Problem

The endzone camera still does not calibrate. Three approaches have failed, each
for a *measured* reason:

1. **Field markings (~9 prototypes)** — yard-line IDENTITY is unrecoverable from
   a single endzone frame (the painted numbers face the sideline, so they are
   edge-on). Labels came out 67-210 px wrong; no single camera fit them.
2. **Geometric cross-camera** — per-frame Hungarian matching of player feet
   gives ~85 % mispairs. Even at the TRUE camera the correspondences reproject
   at **87 px median**, and the Y-reflection outscores the truth.
3. **Jersey identity** — correct pairs, but far too sparse: on play_002, 5 uids
   shared across cameras and **0 of 327 frames** reached the >= 4
   correspondences per frame `solve_fixed_center` requires (median 1).
   Multi-play does not help: more plays add frames that each still carry ~1.

Root cause common to 2 and 3: **players are a bad measurement**. The sideline's
weak axis (grazing Y, ~1-2 m error) is exactly the axis the endzone measures
most sensitively, and a correspondence needs the same player usable in both
views at the same instant.

## Key measurement that motivates this design

The endzone camera is a **fixed tripod**: it pans/tilts/zooms but never
translates. With no translational parallax, frames are related **exactly** by a
homography, independent of scene depth. Measured on play_002 (SIFT on
player-masked frames):

| pair | inliers | median reprojection |
|---|---|---|
| 100 -> 115 | 259 | **0.32 px** |
| 200 -> 215 | 222 | 0.52 px |
| 0 -> 420 (direct) | 47 | ~0.6 px |

**0.55 px median across pairs — ~160x better than the 87 px from players.**
Two further measured facts shape the design:

- **Naive sequential chaining drifts** (6 px at k=60 -> 282 px at k=360), so the
  registration must be a globally bundled graph, not a chain.
- **Direct frame->reference links keep working across the whole play** (36-47
  inliers even at a 420-frame gap), so long-range anchors are available to
  bundle against.
- Absolute focal is **not** recoverable from the homographies alone (the
  orthonormality cost is flat in f; motion is zoom + <1.5 deg pan). Metric scale
  must come from the field, not from self-calibration.

## Goal

Per-frame endzone cameras accurate enough for Gaussian-Splat training:
**few-px reprojection (~1-3 px) against the field model, consistently across the
play**, with an overlay that visibly tracks the real yard lines and hashes, and
the same-field-point proof holding. Sideline calibration unchanged.

## Approach

Three stages, each using the strongest available signal. Precision comes from
**paint** (static, sub-pixel, metrically known); players are demoted to
resolving a discrete labeling choice, so their metre-scale noise never enters
the precision path.

```
1. MOSAIC      masked frames --SIFT--> pairwise links (adjacent + long-range)
               --global bundle--> H_t : frame t -> reference frame

2. REFERENCE   warp every frame's masked white-mask into the reference and
   CAMERA      ACCUMULATE -> paint reinforces, transients wash out. Fit the
               metric field model to the accumulated lines; the discrete
               labeling is pinned by an absolute anchor (goal line / field
               logo) plus the calibrated sideline's yard-range prior.
               Solve ONE camera (C, R_ref, f_ref).

3. PROPAGATE   per-frame camera from H_t + the reference camera. C is shared
               (tripod), so K_t R_t follows from the homography.
```

Why this clears the bar the others could not: stage 2 fits accumulated paint
across a span far wider than any single frame, which is also what makes the
labeling tractable — the failure mode of approach 1 was too few, too ambiguous
lines in one frame.

**Calibrate once per game, not per play.** C is shared across the whole first
half, so the expensive anchor/labeling resolution happens once and every play
reuses the result. Each play still needs its own mosaic (its own pan/zoom).

## Components

New module `nfl_gsplat/calibration/endzone_mosaic.py`:

- `frame_homographies(video, boxes_provider, *, ref_frame=0, step, anchor_every)
  -> dict[int, np.ndarray]`
  SIFT features on player-masked frames; pairwise homographies for adjacent
  links **and** periodic long-range links to the reference; then a global
  bundle so the graph is consistent (naive chaining drifts — measured).
  Fails loud when a frame cannot be linked into the graph.

- `accumulate_field_paint(video, H_by_frame, boxes_provider) -> np.ndarray`
  Warp each frame's player-masked white-mask into the reference frame and
  accumulate. Paint reinforces; moving players and transient junk wash out.
  Returns the accumulated mask (a votes-per-pixel image).

  Cables: SkyCam support cables cross these frames, but they are DARK and the
  accumulated mask is a WHITE-paint mask, so they are excluded by construction
  rather than by a separate rejection stage. Their real effect is OCCLUSION —
  a cable crossing a yard line breaks it in that frame. Accumulation is the
  mitigation: the line is unbroken in the frames where the cable sits
  elsewhere, so paint still wins on votes. Line fitting must therefore tolerate
  gaps and must not treat a per-frame break as a line ending.

- `fit_field_model(accumulated, *, yard_prior, anchor) -> list[(world_xyz, uv)]`
  Detect lines/hashes in the accumulated mask (reusing `field_detect`), fit the
  metric field (`YARD_LINE_SPACING_M=4.572`, field width `2*HALF_WIDTH_M`), and
  resolve the discrete label offset using the absolute anchor plus the
  sideline-derived yard range. **Fails loud on ambiguity** rather than guessing —
  a wrong label offset is exactly how approach 1 failed silently.

- `solve_reference_camera(corrs, image_size, prior) -> CalibrationResult`
  Reuses `joint_solve.solve_fixed_center` (via `_frame_data_override`, since
  these are arbitrary field points, not named landmarks) with the existing
  `center_bounds` to keep C on the correct side.

- `propagate(H_by_frame, ref_camera) -> list[CalibrationResult | None]`
  Per-frame camera from the homography and the reference camera, in TRUE
  endzone pixels (native — no view rotation, no roll).

Driver: `run_autocalib.build_endzone_mosaic(...)`, exposed as
`scripts/02_autocalibrate.py --mode mosaic-endzone`, writing `endzone_*` into
each play's `cameras.npz` with `sideline` preserved.

**Reuse:** `field_detect` (incl. the OpenCV-5 `_hough_rows` fix),
`player_masks.boxes_provider_from_tracks`, `solve_fixed_center`,
`EndzonePrior`, `assemble_track_from_results`, and the existing
identity/`player_uid` work for the yard-range prior.

## Error handling

Fail loud (`CalibrationError` / `SetupError`) with an actionable pointer at every
gate; no silent fallback that changes numbers:

- a frame cannot be linked into the homography graph (names the frame);
- too few accumulated paint pixels / detected lines to fit the field;
- **labeling ambiguous** — more than one label offset fits within tolerance;
- reference-solve residual above the acceptance threshold;
- missing sideline camera or missing `endzone_prior` in `meta.yaml`.

## Testing

- **Synthetic rotating-camera fixture (the core test):** a known field, a fixed
  centre, per-frame rotation + zoom, rendered lines. Assert homographies recover
  the true inter-frame transforms, labeling resolves correctly, and per-frame
  reprojection is **within a few px** — the actual bar.
- **Drift test:** assert the bundled graph beats naive chaining on a synthetic
  sequence long enough for chaining to drift.
- **Unit:** `accumulate_field_paint` reinforces static lines, suppresses a
  moving blob, and recovers a line that a simulated cable occludes in a subset
  of frames;
  `fit_field_model` raises on a deliberately ambiguous input.
- **Real acceptance (the gate):** run on play_002 (and the batch) and report
  **per-frame reprojection against the accumulated field lines**; require
  few-px. Plus overlay tracking and the same-field-point proof. Backup
  `cameras.npz` before the run.
- `pytest -m "not gpu and not slow"` green; ruff clean; synthetic tests CPU/fast.

## Scope

In scope: the three stages above and the CLI mode, for the endzone camera.

Out of scope (explicitly): improving jersey-OCR yield further; changing the
sideline path; `solve_fixed_center` internals beyond the params it already has;
re-attempting single-frame field-marking labeling; splat training itself.

## Risks

- **Labeling remains the crux.** It killed approach 1. Mitigated by the far
  wider accumulated span, an absolute anchor, and the sideline yard prior — and
  by failing loud instead of guessing. If it proves ambiguous in practice, a
  one-time human-confirmed anchor per game is an acceptable fallback (C is
  shared, so it is genuinely once).
- **Global bundle is the most intricate new code**; naive chaining is proven
  inadequate, so this must be right. Covered by the drift test.
- **Zoom range:** if the camera zooms far enough that the reference frame shares
  little overlap with late frames, one reference may not suffice; the fallback
  is multiple reference keyframes tied together.
- **Accumulation assumes the mask is dominated by static paint** once players
  are masked; heavy graphics overlays could violate this (a static-graphics
  filter already exists in the codebase if needed).
