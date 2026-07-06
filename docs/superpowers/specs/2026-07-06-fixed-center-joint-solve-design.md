# Fixed-Center Joint Solve (pretrained calibration, phase 3)

**Date:** 2026-07-06
**Status:** Approved (design); implementation plan pending
**Builds on:** [2026-07-04-pretrained-hybrid-field-registration-design.md](2026-07-04-pretrained-hybrid-field-registration-design.md)

## Problem

Per-frame metric PnP on planar telephoto correspondences is multimodal and
fragile — measured on real footage (SEA_at_AZ play_001 sideline, 2026-07-05/06):

- The SAME frame solves blind but fails with a neighbor's intrinsics prior,
  and vice versa; RMS jumps 0.5 → 20–140 px between adjacent frames.
- Successive per-frame mitigations (RANSAC gate, intrinsics-prior sweep,
  temporal identity propagation) each moved but never closed the
  registration gap; `--mode pretrained` still fails loud at assembly.
- A consistently mislabeled yard line (off by 5 yd) shifts the whole
  per-frame solution 4.57 m with a PERFECT residual — undetectable per frame.
- Meanwhile the per-frame correspondences/homographies are excellent
  (0.1–1.3 px median on all field-visible frames), and every frame that DOES
  solve agrees: camera center ≈ (−19, 1, 95) m, fx ∈ 7000–8500.

The camera is physically fixed (tripod); only pan/tilt/zoom vary. Per-frame
solving re-estimates the camera position every frame as if it could teleport —
wrong estimator for the physics.

## Goal

`cameras.npz` for a full play from `--mode pretrained`: one shared camera
center, smooth per-frame rotation and focal, fit jointly to all fused
correspondences. Acceptance: full-play run completes; grid-overlay diagnostic
tracks the painted field; recovered center stable and physically plausible;
synthetic ground-truth recovery (center < 0.5 m, focals within 2 %).

## Approach (chosen: single joint least-squares)

Alternatives considered and rejected:
- Two-stage (solve C alone, then per-frame pan/zoom given C): simpler math but
  stage 1 needs its own outlier handling and loses joint self-auditing.
- Per-frame solves with a soft center-clustering penalty: keeps the multimodal
  per-frame estimator that already failed.

## Formulation

Unknowns:
- `C ∈ R³` — shared camera center (world frame, meters).
- Per usable frame `t`: `r_t ∈ R³` (Rodrigues rotation), `f_t > 0` (focal, px).

Fixed camera model: `fx = fy = f_t`, principal point = image center, zero
skew/distortion. Per-frame translation is derived, not free:
`t_t = −R(r_t) @ C` (matches the existing `CameraPose` world-to-camera
convention `x_cam = R X + t`, so the camera center is `−Rᵀ t = C`).

Residuals:
- Reprojection of every fused correspondence `(landmark_name → world XY, uv)`
  from phase 1, all usable frames, robust loss `soft_l1` with
  `f_scale ≈ 3 px`.
- Smoothness penalties between consecutive usable frames:
  `λ_f (f_{t+1} − f_t)` and `λ_r (r_{t+1} − r_t)` with small weights
  (pinned in the plan; order such that they matter only when data is thin).

Solver: `scipy.optimize.least_squares(method="trf")` with an explicit
Jacobian sparsity pattern — each reprojection residual touches only `C` and
its frame's `(r_t, f_t)`. Scale: ~4 × T + 3 unknowns (T ≈ 1100 usable frames
→ ~4400), ~16 k residuals. CPU, minutes; no GPU (embers constraint moot).

## Initialization

The existing per-frame prior sweep (`_solve_sweep`, unchanged) runs first and
becomes the initializer:
- `C₀` = median of camera centers from its successful frames.
- `r_t, f_t` per frame from the per-frame successes; linear interpolation
  (and boundary clamp) across frames the sweep could not solve.
- Fail loud (`CalibrationError` with counts) if the sweep produced too few
  successes to initialize (threshold pinned in the plan, order ~20 frames).

## Self-audit loop

After convergence: compute per-frame median reprojection residual; frames
above a threshold (~6 px) are dropped (this is where 5-yd identity shifts
land — irreconcilable with the shared center) and the solve re-runs once
without them (max 2 rounds). Dropped and never-fused frames are gaps for the
existing assembly (interior interpolation ≤ max_gap, boundary
clamp-extrapolation with conf = 0, long interior gaps still fail loud).

## Integration

- New module `nfl_gsplat/calibration/joint_solve.py`; entry point
  `solve_fixed_center(corrs_by_frame, image_size, *, init_results, …) ->
  list[CalibrationResult | None]` (aligned to frame indices).
- Pretrained pipeline becomes: phase 1 detection + fusion (unchanged) →
  phase 2 per-frame sweep (unchanged; initializer) → **phase 3 joint solve** →
  existing `assemble_track_from_results` → `write_camera_track`.
- Per-frame sweep results are no longer written to the track directly; the
  joint solve's output replaces them everywhere in pretrained mode. Hint and
  learned modes untouched.
- Output `CalibrationResult`s carry the joint solution (`intrinsics` with
  `f_t`, `pose` with `R_t, t_t = −R_t C`, per-frame rms, correspondence
  count).

## Error handling

- Too few initializer successes → `CalibrationError` naming the count and
  pointing at the acceptance diagnostic.
- Optimizer diverges (final robust cost worse than initial) →
  `CalibrationError` (never silently return the init).
- Self-audit dropping so many frames that a long interior gap forms → the
  existing assembly failure fires (fail loud, names the range).
- No silent fallback to per-frame results.

## Testing

- **Unit:** parameter pack/unpack round-trip; sparsity pattern shape/rows;
  residual function against hand-projected points; `t_t = −R C` convention
  against `CameraPose`/`project_points`.
- **Synthetic ground truth:** fixed C, smooth pan/zoom over ~50 frames,
  landmarks projected with noise (~0.5 px); assert C recovered < 0.5 m and
  focals within 2 %. Add one frame with all identities shifted by 5 yd;
  assert the self-audit drops exactly that frame.
- **Real acceptance (manual):** full-play `--mode pretrained` run produces
  `cameras.npz`; grid-overlay diagnostic on sampled frames tracks the painted
  field; recovered C within ~2 m of (−19, 1, 95) on the eval play.
- `pytest -m "not gpu and not slow"` stays green; joint solve on the
  synthetic 50-frame problem must run in seconds (unit-test friendly).

## Out of scope

- Distortion, principal-point, or fx≠fy estimation.
- Moving-camera (handheld/skycam) support — this solver assumes a tripod.
- Multi-camera coupling (endzone + sideline joint solve) — later cycle.
- Replacing the hint/learned modes.

## Risks

- **Zoom smoothness weight too strong** could bias focal during fast zooms:
  weights chosen small and validated on the synthetic zoom ramp.
- **Initialization basin:** if the per-frame sweep's successes were all in one
  camera pose regime, C₀ could still be fine (center is pose-independent) but
  interpolated `r_t, f_t` for distant gaps may start far from truth; the
  robust loss + smoothness keep the solve anchored by neighboring frames.
- **Runtime** on 1302 frames: sparse trf should be minutes; if not, decimate
  smoothness terms, not data.
