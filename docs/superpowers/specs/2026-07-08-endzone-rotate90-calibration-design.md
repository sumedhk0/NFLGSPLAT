# Endzone Camera Calibration via Rotate-90 View Normalization

**Date:** 2026-07-08
**Status:** Approved (design); implementation plan pending
**Builds on:** [2026-07-06-fixed-center-joint-solve-design.md](2026-07-06-fixed-center-joint-solve-design.md)
(+ its real-footage addendum) and
[2026-07-04-pretrained-hybrid-field-registration-design.md](2026-07-04-pretrained-hybrid-field-registration-design.md).

## Problem

The pretrained-hybrid + fixed-center pipeline is sideline-shaped. On real
endzone footage (SEA_at_AZ play_001 `endzone.mp4`, 2026-07-07) it fails loud
at the joint solve's consistency gate (0 frames), for three reasons:

1. **Fusion geometry assumes the sideline look.** `detect_lines` keeps
   near-VERTICAL Hough segments; `fit_hash_rows` fits horizontal rows. The
   endzone view has yard lines near-horizontal and hash marks in two vertical
   tick-ladder columns (verified on extracted frames).
2. **The multi-start candidate grid only covers sideline positions**
   (X ∈ ±30, Y ∈ ±45–130). An endzone camera sits behind an endzone,
   roughly (±60–120, ≈0, 15–65).
3. **Sparse identity anchors:** the pretrained model yields keypoints on only
   134/1302 endzone frames (tight framing, yard numbers rarely visible) —
   vs 1086/1302 sideline.

## Goal

`--mode pretrained` on a two-camera play writes `cameras.npz` containing BOTH
cameras, with the endzone camera solved to a physically plausible
behind-endzone position and its grid overlay tracking the painted field on
anchored frames. Coverage is a measured outcome, not a gate: fewer solved
endzone frames than sideline's 816 is acceptable this cycle (the count is
recorded; a later iteration raises it if splat quality demands).

## Approach (chosen: rotate-90 view normalization)

Rotate endzone frames 90° before the pipeline: yard lines become vertical,
hash columns become rows — detection, fusion, static-graphics filter, sweep,
and joint solve all run UNCHANGED. The known in-plane rotation is composed
back into each recovered camera afterwards (exact: fx = fy, so an in-plane
90° rotation absorbs entirely into R; principal point maps center → center of
the swapped-dimension image).

Rejected alternatives:
- Orientation-parameterized detectors (horizontal-line Hough, vertical hash
  columns): duplicates every geometry stage and its tests.
- Fine-tuning the model on endzone views: rejected for the same reason as the
  original no-training decision; geometry reuse beats data collection.

## Components

### 1. Per-camera view rotation seam (`run_autocalib.py` + CLI)

- Rotation resolved per camera: name `endzone` → 90°, all others → 0°;
  CLI override `--rotate cam=deg` (deg ∈ {0, 90, 180, 270}, repeatable /
  comma-separated). 90 vs 270 is immaterial (they differ by a 180° proper
  rotation the camera solve absorbs); the default is always 90.
- Applied inside `build_autocalib_npz_pretrained` per camera:
  - frames rotated after decode (`cv2.rotate`; 90° = ROTATE_90_CLOCKWISE);
  - cached Roboflow keypoints' (u, v) mapped by the same pixel transform
    (no API re-run);
  - `image_size` becomes the rotated (H, W) for detection/fusion/solve;
  - `ffprobe_meta` frame-count validation unchanged (count is
    rotation-invariant).

### 2. Pixel-map + de-rotation utilities (`nfl_gsplat/calibration/view_rotation.py`, new)

- `rotate_image(bgr, deg)` — thin cv2.rotate wrapper (0 = passthrough).
- `rotate_uv(u, v, deg, orig_wh) -> (u', v')` — pixel map matching
  cv2.rotate's convention exactly (unit-tested against cv2.rotate on a
  marked synthetic image, not derived on faith).
- `derotate_result(result, deg, orig_wh) -> CalibrationResult` — converts a
  solution computed in rotated-image coordinates back to ORIGINAL pixel
  coordinates: R_out = Rz(θ) @ R_solved with the sign/axis matching the
  pixel map; intrinsics rebuilt with the original width/height and the same
  focal (fx = fy, principal point = image center in both frames). Exactness
  unit-tested: projecting a world point through the de-rotated camera onto
  the original frame equals mapping the rotated projection back through the
  inverse pixel map (≤ 1e-6 px).
- `cameras.npz` therefore stays in true original-pixel coordinates —
  downstream consumers never see the rotation.

### 3. Candidate grid extension (`joint_solve.py`)

Add behind-endzone candidates to `_candidate_centers`:
X ∈ {−120, −90, −60, +60, +90, +120}, Y ∈ {−20, 0, +20}, Z ∈ {15, 35, 65}
(54 new), keeping the existing sideline grid and plausible-anchor candidate.
The multi-start scoring already selects the winner; sideline plays simply
never pick the new candidates. The anchor plausibility gate widens to accept
|X| ≤ 300 with no |Y| floor (already the case).

### 4. Sparse-anchor reality (measured, not designed around)

134 anchored frames feed identity; temporal propagation and the joint solve
spread them. No new identity machinery this cycle. The acceptance run
records: frames with fused correspondences, frames kept by the rescue refit,
and the recovered camera. If coverage is too low for downstream use, the
NEXT cycle adds endzone-specific anchors (goal line / endzone paint), with
cross-camera consistency as a further option.

## Error handling

- Unknown rotation value in `--rotate` → typer.BadParameter.
- Everything downstream keeps its existing fail-loud behavior; a
  still-unsolvable endzone camera fails at the existing joint-solve gates
  with its existing messages (now naming physically reachable candidates).
- Per-camera failure currently aborts the whole run before writing anything;
  this cycle ALSO tags the failing camera name into the raised
  CalibrationError message in `build_autocalib_npz_pretrained` (small,
  discovered during the 2026-07-07 failure where the camera was ambiguous).

## Testing

- **Unit:** `rotate_uv` vs cv2.rotate ground truth on a marked image, all
  four rotations; `derotate_result` exactness (projection equivalence);
  keypoint-cache rotation round-trip (rotate 90 then 270 = identity).
- **Synthetic:** a 90°-rotated variant of the joint-solve ground-truth
  scene — rotate the synthetic uv observations and image size, solve, then
  `derotate_result`; assert the recovered camera matches the unrotated
  ground truth (C < 0.5 m, focal 2 %).
- **Integration:** `build_autocalib_npz_pretrained` wiring test asserting the
  endzone camera's frames and cached keypoints are rotated before fusion and
  results de-rotated before assembly (monkeypatch seam capture, following the
  existing wiring-test pattern).
- **Real acceptance (manual):** two-camera run on SEA_at_AZ play_001 writes
  `cameras.npz` with `sideline_*` AND `endzone_*` arrays; endzone camera
  center behind an endzone (|X| ∈ 50–150, |Y| ≤ 40, Z ∈ 10–80); grid overlay
  through the de-rotated endzone K,R,t tracks the painted field on anchored
  frames. Record the endzone kept-frame count.
- `pytest -m "not gpu and not slow"` green; ruff clean.

## Out of scope

- Raising endzone anchor coverage (goal-line/endzone-paint identity,
  cross-camera anchoring, model fine-tuning) — next cycle, informed by this
  cycle's measured coverage.
- Multi-camera JOINT optimization (shared points across cameras).
- Non-broadcast camera types (skycam, handheld).

## Risks

- **Yard lines too faint in the endzone view** for the white-threshold Hough
  even after rotation (hash ticks are the strong signal). Mitigation: hash
  ROWS (post-rotation) come from the tick ladders, which are dense and
  bright; lines may ride on fewer segments. If detection starves, the
  fail-loud gates say so and the next cycle tunes `FieldDetectConfig` for the
  endzone camera.
- **134 anchors may not reach** the 10-frame consistency gate after fusion.
  Then the run fails loud with the endzone camera named — that outcome still
  ships the seam + grid work and precisely scopes the follow-up.
