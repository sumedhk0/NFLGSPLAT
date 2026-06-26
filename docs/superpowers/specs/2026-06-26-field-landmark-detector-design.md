# Learned Field-Landmark Detector

**Date:** 2026-06-26
**Status:** Approved (design); implementation plan pending

## Problem

Classical field registration on real All-22 footage hit two walls:
1. **Labeling** — distinguishing real yard lines from noise (painted numbers,
   jersey scraps) and assigning absolute yardage. The homography-consensus labeler
   (merged) solved this on clean frames but is fragile under heavy occlusion.
2. **Conditioning** — every correspondence we can extract (yard-line × hash-row)
   lives in a thin ±2.8194 m band (≈12% of the 48.77 m field width). A homography
   fit from such a thin strip is well-constrained along the band but wrong in slant;
   the field-overlay diagnostic showed the predicted grid drifting off the painted
   lines away from the hashes, despite a 1.31 px in-band residual.

Both are detection/correspondence problems. A learned **field-landmark detector**
that outputs semantically labeled, vertically-spread landmarks dissolves both.

## Goal

A keypoint model that, per frame, detects named NFL field landmarks (labeled
points) with enough vertical spread to condition the homography/PnP well. Success
metric: on real footage, the **field-overlay** (predicted field grid projected
through the recovered homography) tracks the painted lines/hashes/numbers across
the whole frame — the test classical just failed — and held-out reprojection error
is a few px. Output feeds the existing `solve_pnp_from_correspondences` / homography
path; the hint + consensus + classical-detect machinery is retired to fallback.

**Scope:** the field-landmark detector only. Player detection/tracking (exists),
player pose/joints (SMPL-X avatars, planned), and multi-view triangulation are
separate cycles. See [[next-cycle-focal-pose-from-homography]] — better-conditioned
correspondences from this detector also make that focal/pose problem less
degenerate.

## Key idea — number anchors are the conditioning fix

The painted yard **numbers** are the always-visible source of vertical spread.
Per the NFL rulebook the bottom of a field number is 12 yd (10.9728 m) from the
sideline and numbers are 6 ft (1.8288 m) tall, centered on their yard line. So
number-corner/anchor landmarks live at world **Y ≈ ±13.4112 m (bottom)** and
**±15.24 m (top)** — between the ±2.8194 m hashes and the ±24.384 m sidelines.
Labeling/detecting number anchors gives every frame points spanning Y from ±2.8 to
±15+ m even when sidelines are off-screen, which is exactly the conditioning the
hash-only correspondences lacked. This is the load-bearing schema choice.

## Architecture & data flow

```
hand-label (extended annotate_gui, pre-filled from classical homography)
    → dataset {frame, landmark_name, uv}
    → train heatmap keypoint net (PACE GPU)
    → infer(frame) → [(landmark_name, uv, conf)]
    → solve_pnp_from_correspondences / fit_plane_homography  (existing, unchanged)
    → field-overlay validation
```

A compact heatmap network predicts one Gaussian-heatmap channel per landmark
class; peaks (subpixel-refined, confidence-thresholded) are detections. Output is
*already labeled*, so no hint/consensus/merge.

## Components (new package `nfl_gsplat/landmarks/`)

- **`schema.py`** — the ordered list of landmark classes (the model's output
  channels) and their world coordinates. Reuses `field_landmarks.NFL_LANDMARKS`
  (line × hash, line × sideline) and **adds number-anchor landmarks**
  (`{side}_{yd}_left_number_top/bottom`, `…_right_number_top/bottom`) at the Y
  values above. Classes are **scoped to the footage's yard-range** (config) to keep
  K and per-class data sane. Provides `class_names() -> list[str]` and
  `world_xyz(name) -> (X,Y,Z)`.
- **`label_tool.py`** (extend `calibration/annotate_gui.py`) — step through many
  sampled frames of a clip; click visible landmarks; **pre-populate suggested
  positions** by projecting the field model through the classical homography on
  clean frames (nudge, don't click-from-scratch). Saves a per-clip dataset JSON
  (list of `{frame, name, uv}`). Keeps the existing keys/zoom UI.
- **`dataset.py`** — `LandmarkDataset` loads the label JSONs + extracts frames →
  `(image_tensor, target_heatmaps, visibility_mask)`. Gaussian target at each
  labeled `uv` in its class channel; invisible classes → zero channel. Augmentation
  (color/brightness/blur, small affine with consistent uv transform).
- **`model.py`** — `LandmarkNet`: compact UNet/HRNet-style backbone (PyTorch, no
  mmpose), `K` output channels at a fixed input size (e.g. 960×540 → heatmaps at
  ¼ res). `forward(image) -> heatmaps`.
- **`train.py`** — training loop on PACE GPU: heatmap MSE/focal loss over visible
  channels, augmentation, val split, checkpointing, fail-loud on missing
  data/CUDA. CLI with charge-account-friendly flags.
- **`infer.py`** — `detect_landmarks(frame, model, *, conf_thresh) ->
  [(name, (u,v), conf)]`: forward, per-channel peak + subpixel refine + threshold.
- **Integration** — `scripts/02_autocalibrate.py` gains a **learned mode**: per
  frame, `detect_landmarks` → correspondences → `solve_pnp_from_correspondences`
  (or `fit_plane_homography`) → `assemble_track_from_results` (unchanged). The
  hint/consensus path stays as a selectable fallback.
- **`field_landmarks.py`** — add the number-anchor world points (so they exist in
  the shared world frame for both labeling and reprojection).

## Labeling workflow

1. Sample ~100–150 frames per clip spanning the pan/zoom range (denser where the
   view changes fast).
2. `label_tool.py` shows each frame with classical-homography-projected suggestions;
   the user confirms/nudges visible landmarks (hashes, number anchors, sidelines
   when present), skips occluded ones.
3. Output: one dataset JSON per clip; a few clips → the training set.

## Model & training

- Input frame downscaled (e.g. 960×540); heatmaps at ¼ resolution; Gaussian σ≈2 px.
- Loss: MSE (or focal-MSE) on channels whose landmark is labeled-visible in that
  frame; unlabeled-but-present landmarks are a known limitation (treated as absent).
- Heavy augmentation (this is the overfit defense on limited data).
- Per-dataset model: trained and used on the same footage domain; retrained as
  footage grows. K scoped to covered yard-range.
- PACE: `module load anaconda3`, charge account `paceship-pso`, GPU partition.

## Integration & what it replaces

`detect_landmarks` output is labeled correspondences with vertical spread → straight
into the existing PnP/homography solve. The learned path **does not need**
`seed_state_from_hint`, `label_lines_by_consensus`, `identify_correspondences`, or
classical `detect_lines`/`fit_hash_rows` — those remain for the fallback/pre-labeler
only. `assemble_track_from_results` + smoothing + `cameras.npz` are unchanged.

## Validation & testing

- **Acceptance (real):** the field-overlay diagnostic (`scripts/diag_calib.py`,
  already built) on held-out frames — the projected grid must track the painted
  field; plus median reprojection error on held-out labeled landmarks (target: few
  px, well-conditioned across the frame, not just in-band).
- **Unit (CPU):** heatmap target generation (Gaussian at uv, zero when absent);
  peak extraction + subpixel refinement recovers a known uv; dataset loader shapes
  + augmentation uv-consistency; `schema.world_xyz`/`class_names` correctness incl.
  number-anchor Y values; `detect_landmarks` thresholding/format; landmark→
  correspondence conversion.
- **Integration (tiny):** overfit the model to a handful of frames and confirm the
  pipeline (train→infer→homography→overlay) runs end-to-end and the overlay aligns.
- Keep `pytest -m "not gpu and not slow"` green; GPU/train tests gated by markers.

## Failure handling

- Too few confident detections in a frame (< min for a solve) → that frame is a
  gap (existing `register_frame`/`assemble` fail-loud).
- Missing model weights / dataset / CUDA → `SetupError`/`CalibrationError` with a
  pointer (consistent with the project's fail-loud convention).
- The field-overlay is the guard against a confidently-wrong model: drift =
  reject, retrain/relabel.

## Risks

- **Overfit to one stadium/lighting** (limited footage): accepted — per-game model,
  retrained as footage grows; augmentation + number/hash landmarks (high-contrast,
  stable) mitigate.
- **Per-class data sparsity:** mitigated by per-dataset scoping (panning yields many
  examples of the covered yard lines).
- **Labeling effort:** mitigated by classical pre-labeling + the existing GUI.
- **Unlabeled-visible landmarks** train as "absent" (label-completeness noise):
  mitigated by pre-labeling completeness and confidence thresholding at inference.

## Out of scope

- Player detection/tracking, player pose/joints, multi-view triangulation.
- A universal (cross-stadium) model — this is per-dataset for now.
- Synthetic data generation (a future option if footage diversity proves limiting).
- Per-frame focal/K,R,t extraction (deferred cycle; this detector improves its
  conditioning but does not solve it).

## Phasing (for the plan)

1. `field_landmarks.py` number-anchor world points + `landmarks/schema.py`.
2. `landmarks/dataset.py` + heatmap target / peak utils (+ unit tests).
3. `landmarks/model.py` (`LandmarkNet`) + forward-shape tests.
4. `label_tool.py` (extend `annotate_gui`) multi-frame + classical pre-fill.
5. `landmarks/train.py` (PACE GPU) + `landmarks/infer.py` (`detect_landmarks`).
6. `02_autocalibrate` learned mode + field-overlay validation harness.
