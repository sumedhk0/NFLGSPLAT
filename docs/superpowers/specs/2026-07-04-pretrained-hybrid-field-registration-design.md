# Pretrained-Hybrid Field Registration (no-training path)

**Date:** 2026-07-04
**Status:** Approved (design); implementation plan pending
**Supersedes:** the labeling/training phases of
[2026-06-26-field-landmark-detector-design.md](2026-06-26-field-landmark-detector-design.md)
(the `nfl_gsplat/landmarks/` package stays as a fallback; no further hand-labeling
or training is planned).

## Problem

The learned-detector plan requires ~100–150 hand-labeled frames per clip plus a
training run per footage domain. Evaluation of a pretrained Roboflow keypoint
model (`football-field-key-points-mvmjf/2`, American-football fields) on our
All-22 footage showed we can skip both:

- **Identity is solved.** The model reads the painted yard numbers: on 16/20
  sampled frames it emitted correctly-identified yard-line keypoints (`20`,
  `30`, `40`, hash variants), never swapping lines (RANSAC residuals 1–3 px;
  a swap would show ~76 px).
- **Geometry is coarse.** Model keypoints are ±3–30 px off the painted marks —
  good enough to pick the right line (next candidate 170+ px away), not good
  enough to be the geometric measurement.
- **Classical detection has precise geometry but no identity** — and one bug:
  `_merge_collinear` (field_detect.py) collapses merged Hough segments to
  vertical, destroying slant on real (non-synthetic) views. A scratchpad
  slope-preserving merge put lines dead on all painted yard lines across the 16
  field-visible frames.

Hybrid: **model names the lines, classical measures them.**

## Goal

Per-frame labeled, precise correspondences → homography/PnP → `cameras.npz`,
with zero hand-labeling and zero training. Acceptance: field-overlay grid
tracks the painted lines/hashes/numbers across the play; per-frame homography
from ≥8 well-spread inliers at ≤2 px median residual on field-visible frames.

## Architecture & data flow

```
[Windows, once per play]
  scripts/03_roboflow_precompute.py video.mp4
      → hosted Roboflow inference per frame → <play_dir>/roboflow_kps.json

[PACE or local, deterministic, offline]
  run_autocalib.build_autocalib_npz_pretrained
      per frame:
        classical: slope-preserving yard lines + hash-row fits   (repaired detect)
        model:     named keypoints from roboflow_kps.json        (identity)
        fuse:      name→nearest-line assignment → line×hash-row & number
                   intersections → labeled correspondences
        solve:     existing solve_pnp_from_correspondences path
      → assemble_track_from_results (unchanged) → cameras.npz
```

## Components

### 1. `scripts/03_roboflow_precompute.py` (new; runs on Windows)

- Reads the play video, sends frames to Roboflow hosted inference
  (`inference-sdk` HTTP client, as in the eval script), writes
  `<play_dir>/roboflow_kps.json`:
  `{frame_idx: [{"name": "30-top-hash", "u":…, "v":…, "conf":…}, …]}` plus a
  header (model id/version, video hash, frame count, kp-conf threshold).
- `--stride N` (default 1) to economize API credits; downstream tolerates
  missing frames (identity propagates via the per-frame homography of the
  nearest solved frame, mirroring `_register_sequence`'s prior-propagation).
- Sideline classes (`*-sl`) are dropped at write time (hallucinated off-frame
  in eval).
- This is the only step needing internet/API key; the cluster consumes the
  JSON. Frames/JSON stay outside git (contains no imagery, but lives in
  `<play_dir>` with the video, not the repo).

### 2. Classical repair (`nfl_gsplat/calibration/field_detect.py`)

- `_merge_collinear`: group near-vertical Hough segments by x-at-mid-height
  (tolerance ~25 px), then least-squares fit `x = a·y + b` through member
  endpoints; return slanted segments spanning the members' y-range. (Validated
  in scratchpad on all 20 eval frames.)
- Player masking stays as-is (`player_boxes` → masked white mask); the
  pretrained path passes YOLO boxes via the existing `masks_provider` seam to
  kill white-uniform false lines.
- Existing synthetic-image tests updated: merged lines must preserve slope
  (regression test with a slanted synthetic field).

### 3. Model-keypoint mapping (`nfl_gsplat/calibration/roboflow_kps.py`, new)

- Load/validate `roboflow_kps.json` (fail loud on model-id mismatch or empty).
- `to_nfl_name(model_class, territory) -> str | None`: promote the eval
  script's `_to_nfl_name` table (yard + top/bottom/hash variants →
  `NFL_LANDMARKS` names; `mid_50` special case; unmappable → None).
- `territory` ("home"/"away") comes from the play's `meta.yaml` calib hints —
  the model's classes carry yard numbers but not which side of the 50.

### 4. Fusion (`nfl_gsplat/calibration/fuse_pretrained.py`, new)

Per frame, given classical lines `x = a·y + b`, classical hash points, and
model keypoints:

- **Line identity:** each model yard-line keypoint (numbers and hash variants
  carry a yard id) votes for the classical line nearest in x at the keypoint's
  row. Gates: assignment distance ≤ 60 px AND ≥ 2× margin over the runner-up;
  conflicting votes for one line → majority, tie → drop line (fail toward
  fewer, correct correspondences). Lines with no vote get identity by yardage
  ordering between voted neighbors (5-yd spacing is monotone in x).
- **Hash rows:** existing `fit_hash_rows` on classical hash points → top/bottom
  row curves.
- **Correspondences:** identified line × hash-row intersections (precise,
  labeled) + model number keypoints (coarse but vertically far from hashes —
  the conditioning anchors, exactly the thin-band fix from the 06-26 design).
- Output `[(landmark_name, (u,v))]` → existing `_register_corrs` (PnP) path.

### 5. Orchestration (`run_autocalib.build_autocalib_npz_pretrained`, new)

- Mirrors `build_autocalib_npz_learned`: stream frames, per-frame fuse+solve,
  `assemble_track_from_results`, `write_camera_track`. Reuses `masks_provider`.
- Frames with no model keypoints within `--identity-reach` frames and no
  propagated identity → gap (None); existing gap interpolation/fail-loud
  applies (post-play sideline-crowd frames are the expected gaps).

## Error handling

- Missing/malformed `roboflow_kps.json` → `SetupError` pointing at
  `scripts/03_roboflow_precompute.py`.
- Video/JSON frame-count mismatch → `SetupError` (stale cache).
- Ambiguous identity votes → drop (never guess); too few correspondences →
  frame gap → existing `CalibrationError` on long gaps.
- No silent fallbacks: pretrained mode never quietly reverts to the
  hint/consensus path.

## Testing

- **Unit:** slope-preserving merge (synthetic slanted lines); `to_nfl_name`
  table (incl. mid_50, unmappable classes); vote gating (distance, margin,
  conflict, interpolated neighbors); intersection correspondence naming.
- **Integration (synthetic):** render slanted synthetic field, synthesize model
  keypoints with ±20 px noise + correct names → fused homography reprojects
  landmarks ≤1 px.
- **Acceptance (real, manual):** field-overlay diagnostic on the eval play —
  grid tracks painted field on all field-visible frames.
- `pytest -m "not gpu and not slow"` stays green; no GPU needed anywhere in
  this cycle.

## Out of scope

- Fixed-camera-center pan/zoom joint solve (next cycle; consumes this cycle's
  per-frame homographies — see [[next-cycle-focal-pose-from-homography]]).
- Training/fine-tuning any model (explicitly rejected: snap-to-classical beats
  fine-tuning and needs no data).
- Player detection/pose, multi-view.
- Retiring `nfl_gsplat/landmarks/` (kept as fallback, unused by default).

## Risks

- **Roboflow API dependency** (hosted, per-play precompute): mitigated by JSON
  caching (one call per play ever), `--stride`, and version pinning in the JSON
  header. If the hosted model vanishes, fallback is the 06-26 learned path.
- **Territory flag wrong** → all identities mirrored; caught by the
  field-overlay acceptance check (grid slants opposite to paint).
- **Uniform-colored false lines** on unmasked runs: masked in production via
  YOLO boxes; vote gating tolerates leftovers.
