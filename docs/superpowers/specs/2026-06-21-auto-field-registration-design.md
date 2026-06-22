# Automatic Per-Frame Field Registration (Classical CV + OCR)

**Date:** 2026-06-21
**Status:** Approved (design); implementation plan pending

## Problem

Per-frame camera calibration is needed for PTZ All-22 footage. The first design
(keyframe anchors + homography tracking) requires manual landmark annotation per
play, which **does not scale to a 17-game season** (hundreds of plays). We replace
the manual front end with **automatic per-frame field registration**: detect and
identify field markings in each frame, solve the camera per frame, with no manual
annotation.

The footage is favorable for a classical approach: clean coaches/tactical angle,
no graphics overlays, no hard cuts, mostly field, painted yard numbers usually
visible. Frames are often zoomed to ~20 horizontal yards (not the whole field).

## Decisions (locked)

1. **Method: classical CV + OCR** (no training data). Detect yard lines + hash
   marks geometrically; OCR the painted yard numbers (reuse PaddleOCR) to pin
   absolute yardage; build correspondences; solve per frame.
2. **Drop-in for the manual GUI**: the detector emits the same
   `[{"name","uv"}]` correspondence format `annotate_gui` produced, so
   `solve_pnp_from_annotations` is reused unchanged.
3. **Per-frame solve + temporal**: independent PnP per frame, then one-euro
   smoothing + short-gap interpolation over the `(K,R,t)` sequence. **Fail loud**
   only on a long run of unregisterable frames.
4. **Temporal identity propagation**: yard-line *identity* (which yard line is the
   35) is established when a number is OCR'd and carried across frames by line
   continuity, so pose is still solved per frame but labeling survives frames with
   no visible number.
5. The keyframe + homography-tracking path (`02_calibrate_cameras.py`,
   `02b_track_calibration.py`, `track_homography.py`) is **kept as a dormant
   fallback**, not deleted. The new auto stage produces `cameras.npz` directly.

## Architecture

Per frame `t`:
```
frame_t ──detect──▶ {line segments, hash ticks, OCR number boxes}
        ──identify─▶ assign absolute yardage + map intersections to NFL_LANDMARKS
                     names  →  correspondences [{name, uv}]
        ──solve───▶ solve_pnp_from_annotations(...)  →  (K, R, t), rms
```
Across the clip:
```
[(K_t,R_t,t_t) or None per frame]
   ──temporal──▶ one-euro smooth + interpolate_short_gaps over the pose params
   ──assemble──▶ CameraTrack per cam  →  cameras.npz
   ──gate─────▶ fail loud if a long run had no registration
```

This reuses the entire per-frame data layer and every downstream consumer; only the
front end (detection + identification + per-frame orchestration) is new.

## Components (isolated, single-responsibility)

### `nfl_gsplat/calibration/field_detect.py` (cv2 + OCR seam)
`detect_field_features(frame_bgr, *, cfg) -> DetectedFeatures` where
`DetectedFeatures` holds:
- `lines`: list of line segments (endpoints) classified into `yard` vs `boundary`
  (sideline) families by orientation clustering.
- `hashes`: detected hash-tick image points (two rows).
- `numbers`: list of `(text, center_uv, orientation)` from OCR of candidate
  painted-number regions.
Implementation: field-white mask (bright pixels on green), `cv2.createLineSegmentDetector`
or `HoughLinesP`, orientation clustering; hash detection via small-blob/tick
filtering near the field interior; number regions rectified using the detected
line geometry, then PaddleOCR (reuse the wrapper in `tracking/jersey_ocr.py`). This
is the OpenCV/OCR seam — its internals are validated on synthetic rendered field
images and at real-footage bring-up.

### `nfl_gsplat/calibration/field_identify.py` (pure, the testable core)
`identify_correspondences(features, prior, *, cfg) -> (correspondences, identity_state)`:
- From OCR'd numbers, pin absolute yardage to the nearest yard line; propagate via
  the regular 5-yard spacing of detected lines; resolve the 50-yard mirror using
  number facing/arrow or the monotonic ordering toward the 50.
- When no number is OCR'd this frame, use `prior` (the previous frame's
  `identity_state`) + line continuity (nearest-line matching) to carry yardage.
- Map each identified yard-line × hash-row / × sideline intersection to an
  `NFL_LANDMARKS` name → `{name, uv}`. Returns the correspondence list + updated
  `identity_state` for the next frame.
- Pure numpy/geometry; fully unit-testable with synthetic detected-feature inputs.

### `nfl_gsplat/calibration/register_frame.py`
`register_frame(features, prior, image_size, *, cfg) -> (CalibrationResult | None, identity_state)`:
calls `identify_correspondences`, then `solve_pnp_from_annotations` on the
in-memory correspondences (refactor `solve_pnp` to accept an in-memory list, not
only a JSON path — small additive change). Returns `None` when fewer than the
minimum correspondences are found or RMS exceeds tolerance (that frame is a gap).

### `nfl_gsplat/calibration/run_autocalib.py` + `scripts/02_autocalibrate.py`
Per camera: iterate frames (reuse `utils.video.iter_frames`), `detect → register`,
collect per-frame `(K,R,t)` (NaN/None for gaps), then reuse
`temporal_smooth.smooth_param_sequence` + `interpolate_short_gaps` on the pose
parameter streams, assemble a `CameraTrack`, and gate with the existing
fail-loud-on-long-gap logic (reuse/share `track_homography.check_confidence`'s
contract, or a small `check_registration_gaps`). Write `cameras.npz` via
`write_camera_track`. This **replaces `02b`** in `04_process_play.sh`.

### Small reuse refactor
`solve_pnp_from_annotations` currently takes a JSON path. Add a sibling
`solve_pnp_from_correspondences(world_uv_pairs, image_size, ...)` (the path loader
becomes a thin wrapper) so per-frame registration avoids temp files.

## Field world model

`NFL_LANDMARKS` already maps landmark names → world `(x, y, 0)`. The identifier
produces those exact names, so world coordinates come for free and the
correspondence format matches the manual annotator's output.

## Failure handling

- Frame with `< min_correspondences` or high RMS → registration `None` (a gap);
  bridged by `interpolate_short_gaps` if short.
- A long run of consecutive gaps → `CalibrationError` naming the frame range
  (fail loud; consistent with the project philosophy).
- OCR yields nothing for a stretch → identity propagated from `prior`; if identity
  was never established for the clip (no number ever OCR'd), fail loud with a clear
  message (this clip needs a manual keyframe — the dormant fallback path).

## Acknowledged risks (bring-up watch items)

1. **Painted-number OCR** is the main real-footage risk: large, perspective-warped,
   split digits ("4 0") with directional arrows. Mitigation: rectify the number
   region via detected line geometry before OCR; the arrow aids 50-yard
   disambiguation. Tunable; may need per-footage thresholds.
2. **No-number frames** (very zoomed, between 10-yard-spaced numbers) rely on
   temporal identity propagation; a clip that opens already zoomed between numbers
   with no establishing number is the worst case → fail loud → manual fallback.
3. **Decompose/PnP conditioning** when very few well-spread points are visible
   (already noted in the per-frame-calibration spec); per-frame RMS gate catches it.

## Testing (CPU)

- `field_identify` (core): synthetic detected-feature + OCR inputs → assert correct
  yardage assignment, 50-yard mirror resolution, identity propagation across a
  no-number frame, and correct `NFL_LANDMARKS` correspondences.
- `solve_pnp_from_correspondences`: in-memory correspondences from a known camera →
  recovered `(K,R,t)` within tolerance (mirror existing solve_pnp tests).
- `register_frame`: synthetic features for a known camera → `CalibrationResult`
  matching truth; too-few-correspondences → `None`.
- `run_autocalib`: a synthetic per-frame `(K,R,t)` sequence with a short gap →
  smoothed/interpolated `CameraTrack`; a long gap → `CalibrationError`.
- `field_detect`: synthetic rendered field image (lines + numbers drawn on a green
  canvas under a known homography) → detects the lines/numbers; real cv2/OCR. Real
  broadcast robustness is bring-up.
- Keep `pytest -m "not gpu and not slow"` green; ruff clean.

## Phasing (for the plan)

1. `solve_pnp_from_correspondences` refactor (+ keep JSON wrapper).
2. `field_identify.py` pure core + tests (the disambiguation logic).
3. `field_detect.py` cv2/OCR seam + synthetic-image tests.
4. `register_frame.py` + tests.
5. `run_autocalib.py` + `scripts/02_autocalibrate.py`, temporal smooth + fail-loud,
   write `cameras.npz`.
6. Wire into `04_process_play.sh` (replace the `02b` step) + docs.

## Out of scope

- Trained keypoint models (synthetic or hand-labeled) — the hardening path if
  classical proves too brittle on real footage.
- Removing the keyframe/tracking fallback code.
- Lens-distortion / rolling-shutter modeling.
- Any change to pose/avatar/render stages beyond consuming `cameras.npz` (already
  done).
