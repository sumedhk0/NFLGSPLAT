# Hinted Per-Frame Field Registration (one yardage hint per play)

**Date:** 2026-06-22
**Status:** Approved (design); implementation plan pending

## Problem

Automatic per-frame field registration (classical CV + OCR) is merged, but on the
real footage the **painted-number OCR does not work** — PaddleOCR can't localize
the small, perspective-skewed, rotated field numbers (verified: full-frame and
band-cropped, both rotations, all empty). Number OCR was the only mechanism for
assigning **absolute yardage** ("which detected line is the 30"), so today nothing
registers.

Everything else is sound or clearly feasible (confirmed on real frames):
- Yard-**line** detection works (over-fires, mostly on white player jerseys).
- Hash-**tick** rows are crisp and easily detectable in the white mask.
- Per-frame PnP, the `CameraTrack`/`cameras.npz` data layer, temporal smoothing,
  and all downstream consumers are built and tested.

Replace the failed OCR with a **single human yardage hint per play per camera**:
fully automatic line/hash detection + per-frame PnP + label propagation, anchored
once. This scales to a season (one number + side + direction per camera, vs. the
keyframe approach's many clicked landmarks per play) and is reliable.

## Decisions (locked)

1. **Hint in `meta.yaml`** — per camera: a reference frame, the approx image-x of a
   yard line the user can identify, that line's absolute yard + side, and the
   image direction in which yards increase. Headless; recorded with the play.
2. **Player-masked line detection + real hash detection** are the real-frame pieces.
   Hash×line intersections are first-class correspondences (so zoomed frames with
   no sideline in view still register).
3. **Seed-at-ref + bidirectional label propagation** carries the hint's labeling
   across panning/zoom via line-continuity (the `prior` logic already built).
4. **Self-validating:** a wrong hint → high reprojection RMS → fail-loud (correct
   the hint). No silent wrong calibration.
5. **OCR path removed/disabled.** The keyframe + homography-tracking fallback
   (`02_calibrate_cameras.py`, `02b_track_calibration.py`, `track_homography.py`)
   stays on disk.

## Hint format (`meta.yaml`)

```yaml
calib_hints:
  sideline: {ref_frame: 0, ref_x: 866, yard: 30, side: away, increasing: right}
  endzone:  {ref_frame: 0, ref_x: 540, yard: 35, side: home, increasing: left}
```
- `ref_frame`: a frame where the detector cleanly finds the yard lines (a wider /
  clean moment).
- `ref_x`: approx image-x of an identifiable yard line — read from the diagnostic's
  printed line x-positions + the saved frame.
- `yard` + `side`: that line's absolute yard (`side` ∈ {home, away, mid}; mid ⇒ 50).
- `increasing`: `left` or `right` — which image direction yards increase (resolves
  the 50-mirror / direction ambiguity).

A `CalibHint` dataclass + loader (`load_meta` gains an optional `calib_hints`
field: `dict[cam -> CalibHint]`). Missing hints → `SetupError` naming the play
(the auto stage needs them). Validation: `side`/`increasing` enums, `yard` a valid
5-yard multiple, all fail-loud.

## Components

### `nfl_gsplat/utils/meta.py` (extend)
Add `CalibHint(ref_frame, ref_x, yard, side, increasing)` and parse a top-level
`calib_hints` mapping into `PlayMeta.calib_hints: dict[str, CalibHint]` (default
empty). Reuse the existing fail-loud validation style.

### `nfl_gsplat/calibration/field_detect.py` (real implementations)
- **`detect_lines`**: accept an optional list of player boxes; zero them out of the
  white mask before HoughLinesP (kills jersey over-detection); keep the merge +
  orientation filter. Tune min-length so short blobs drop.
- **`_detect_hashes` → real**: from the white mask, find small tick marks (small
  connected components within a height/area band), cluster their centroids into
  **two rows** by y, return the tick image points tagged with row (the two rows map
  to left/right hash relative to the field). The OCR seam (`_ocr_numbers`) is
  deleted; `DetectedFeatures.numbers` is dropped (or kept empty/unused).
- Real-frame thresholds (`white_thresh`, hash size band, min line length, mask
  dilation) are tuned at bring-up via the diagnostic; the structure + the
  geometry are unit-tested on synthetic masks.

### `nfl_gsplat/calibration/field_identify.py` (swap seed source)
- Replace `_assign_from_numbers` (OCR) with `assign_from_hint(lines_sorted, hint)`:
  snap `hint.ref_x` to the nearest detected line; label it `(side, yard)`; fill the
  rest by the constant 5-yard index spacing and `increasing` (handles the
  home/away fold consistently — reuse the existing fold logic, now driven by the
  hint's direction instead of OCR `inc`).
- Keep `identify_correspondences(feats, prior)`'s continuity-propagation for frames
  without a fresh seed. Add hash×line and sideline×line correspondence emission
  (hashes now real). De-dup by name.

### `nfl_gsplat/calibration/run_autocalib.py` (seed-at-ref + sweep)
- `build_autocalib_npz` loads the per-cam `CalibHint`. For each camera: detect
  features for the `ref_frame`, `assign_from_hint` to seed identity, then sweep
  **forward** (ref→end) and **backward** (ref→0) calling
  `identify_correspondences(feats, prior)` per frame (prior = previous frame's
  `IdentityState`), `register_frame` → per-frame result. Assemble + smooth + gate
  exactly as today (`assemble_track_from_results`).
- Player boxes per frame come from the existing detector / `tracks.parquet` (the
  `04` order already runs `detect_track` before calibration). A `masks_provider`
  threads the boxes into `detect_field_features`.

### `scripts/02_autocalibrate.py` (load hints)
Pass `meta.calib_hints` into `build_autocalib_npz`; fail loud if a camera's hint is
missing.

### `scripts/diag_calib.py` (keep as the hint-authoring aid)
Already prints detected line x-positions + saves the frame — that's how the user
picks `ref_x`. Keep it; drop the now-irrelevant OCR section (or leave it behind a
flag).

## Data flow

```
meta.calib_hints[cam] ─┐
                       ▼
detect_field_features(frame, player_boxes)  →  lines + hashes
   at ref_frame: assign_from_hint  ─────────►  seed IdentityState
   sweep fwd/back: identify_correspondences(feats, prior)  →  [(name,(u,v))], prior'
   register_frame → solve_pnp_from_correspondences → (K,R,t)|None per frame
assemble_track_from_results (smooth + gap-fill + fail-loud) → CameraTrack → cameras.npz
```

## Cross-camera consistency

One hint per camera; both produce labels in the shared `NFL_LANDMARKS` world frame,
so the two `CameraTrack`s are mutually consistent and two-view triangulation is
valid. A per-camera wrong hint surfaces independently via that camera's RMS gate.

## Failure handling

- Missing `calib_hints[cam]` → `SetupError` (author the hint).
- Wrong hint (bad side/direction/yard) → systematically high per-frame RMS → those
  frames are gaps → long-gap `CalibrationError` (fix the hint).
- Continuity break across a long stretch (e.g., everything pans out of frame) →
  long-gap `CalibrationError` (add a second hint frame). Not expected on clean,
  cut-free footage.

## Testing (CPU)

- `meta.py`: `calib_hints` parse + validation (enums, yard multiple, missing →
  SetupError).
- `assign_from_hint`: snaps to nearest line, labels by spacing + direction, both
  `increasing` values; resolves home/away/mid correctly.
- Propagation sweep: a synthetic sequence of shifting labeled lines (simulated pan)
  → identity carried; a new line entering gets neighbor ±5; PnP recovers a known
  panning camera end-to-end (extend the existing register/autocalib tests).
- `_detect_hashes`: synthetic white mask with two tick rows → correct row grouping
  + points; spurious blobs filtered.
- `detect_lines` masking: synthetic mask with a white "jersey" blob over a line →
  player box removes it; the yard line still detected.
- Keep `pytest -m "not gpu and not slow"` green; ruff clean.

## Phasing (for the plan)

1. `meta.py` `CalibHint` + parse/validate.
2. `field_identify` `assign_from_hint` (+ remove OCR seed) + tests.
3. `field_detect`: real `_detect_hashes` + player-masked `detect_lines` (+ drop
   `_ocr_numbers`) + synthetic tests.
4. `run_autocalib` seed-at-ref + bidirectional sweep + masks threading + tests.
5. `02_autocalibrate` hint wiring + `diag_calib` trim + docs.

## Out of scope

- Resurrecting number OCR (and any trained-model detector).
- Removing the keyframe/tracking fallback.
- Auto-deriving the second camera's hint from the first.
- Lens distortion / rolling shutter.
