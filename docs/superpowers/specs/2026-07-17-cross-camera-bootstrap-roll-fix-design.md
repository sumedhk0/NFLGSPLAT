# Cross-Camera Bootstrap Roll Fix

**Date:** 2026-07-17
**Status:** Approved (design); implementation plan pending
**Builds on / fixes:** [2026-07-15-cross-camera-endzone-calibration-design.md](2026-07-15-cross-camera-endzone-calibration-design.md)

## Problem

The merged cross-camera endzone calibration recovers the synthetic camera to
<0.06 m but FAILS on real footage: `solve_fixed_center` reports "0 frames
consistent with the multi-start camera." Diagnosis (2026-07-16/17, real data,
videos confirmed frame-synced):

- Sideline is calibrated (816 frames), 12 players/frame project to sensible
  field positions (X ∈ [−27, −9], matching the visible 40–20 yard action);
  17 endzone feet/frame; 1302 common frames. The data is there.
- But the bootstrap's best player-to-foot alignment is **59 px** across the
  whole grid + focal sweep — far above the 6 px rescue gate, so every frame is
  rejected.
- ROOT CAUSE: `cross_cam_calib._hyp_cam` builds its hypothesized endzone camera
  with a plain look-at rotation, but the endzone pipeline works in a
  **90°-rotated frame** (`view_deg=90`). The true camera therefore carries a
  90° roll that `_hyp_cam` never applies, so every projected player lands ~90°
  rotated from its detection → garbage matches → the ICP never seeds. The
  synthetic test passed only because its ground-truth camera AND `_hyp_cam`
  both used the same plain look-at (no roll), masking the bug.
- Confirmed fix: adding a 90° roll to the bootstrap camera yields **405 tight
  (≤15 px) inliers at 20 px median** at a plausible C=(−115, 0, 25) — vs 0
  before. `roll=90` wins over 0/180/270, exactly as the rotated frame predicts.

A secondary weakness: the bootstrap scores candidates by match COUNT within a
loose 200 px gate, which rewards spurious clustering (it picked a wrong C with
6125 junk matches over the correct region).

## Goal

`--mode cross-endzone` on the real two-camera play produces a plausible
`endzone_*` camera (near the measured C≈(−115, 0, 25); |X| 50–150, |Y| ≤ 40,
Z 10–80), overlay tracking the field, with the same-field-point proof holding.
The synthetic recovery test — updated to carry a real 90° roll — still recovers
<0.5 m and would fail if the bootstrap ignored roll.

## Approach

Two contained changes to `nfl_gsplat/calibration/cross_cam_calib.py`, nothing
else:

### 1. Roll-aware `_hyp_cam` (the fix)

`_hyp_cam(C, world_pts, image_size, f, *, roll_deg=0)` composes the known
view-rotation roll into the look-at rotation:
`R = view_rotation._rz(roll_deg) @ look_at_R(C, centroid)` (`roll_deg=0`
unchanged). This mirrors exactly how `joint_solve._init_frame` already handles
rotated views (it 4-roll-searches with the same `_rz`). Reuse `_rz` from
`view_rotation` — do not re-derive rotation math.

### 2. Quality-scored, roll-searched bootstrap

In `solve_endzone_cross_camera`'s bootstrap, replace the loose-count score with
a tight-inlier score:
- Candidate set unchanged: `[init_C]` + the behind-endzone grid.
- For each (candidate center, focal, roll ∈ {0, 90, 180, 270}), build
  correspondences with `_hyp_cam(..., roll_deg=roll)` and count matches whose
  reprojection is ≤ a TIGHT gate (`_BOOTSTRAP_TIGHT_PX = 15.0`). Keep the
  combination with the most tight inliers.
- Build the seed correspondences from that winner under a slightly looser gate
  (the existing `match_px[0]`, now with the winning roll) so the ICP has enough
  points, then proceed with the UNCHANGED ICP rounds (`solve_fixed_center` +
  re-match + gate tightening). `solve_fixed_center`'s own per-frame refinement
  and `_init_frame` roll search drive the 20 px seed under its 6 px gate.
- The winning roll is threaded into the round-0 `_hyp_cam` calls only; rounds
  1+ use the solved per-frame cameras (already roll-correct from the solve), so
  no roll bookkeeping is needed there.

The fail-loud thresholds (too-few bootstrap matches, no usable frames) and the
`CalibrationError` messages are unchanged.

## Components

- `nfl_gsplat/calibration/cross_cam_calib.py`
  - `_hyp_cam` — add `roll_deg` keyword; compose `_rz(roll_deg)`.
  - `solve_endzone_cross_camera` — bootstrap now sweeps roll ∈ {0,90,180,270}
    and scores by tight-inlier count; add module const `_BOOTSTRAP_TIGHT_PX`.
- No change to `joint_solve.py`, `run_autocalib.py`, `scripts/02_autocalibrate.py`.

## Error handling

Unchanged: `CalibrationError("endzone: cross-camera matching did not converge …")`
when even the best (center, focal, roll) yields too few tight inliers; existing
no-usable-frames guard. No silent fallback. A still-failing solve fails loud,
naming the endzone camera.

## Testing

- **Unit:** `_hyp_cam(C, pts, wh, f, roll_deg=90)` returns a rotation equal to
  `_rz(90) @ (its roll_deg=0 rotation)` (exact, ≤1e-9); `roll_deg=0` unchanged.
- **Synthetic recovery (updated to be representative):** the synthetic
  ground-truth endzone camera is rebuilt to carry a real 90° roll
  (`R = _rz(90) @ look_at`), so the scene matches the rotated real case. The
  solve must still recover center <0.5 m and focals within 2 % — and this test
  now genuinely exercises the roll search (it would fail if the bootstrap only
  tried roll=0). Add an assertion-light guard test: with roll search DISABLED
  (roll fixed to 0) the same scene does NOT recover, proving the roll sweep is
  load-bearing (implement via a private param or a monkeypatch, whichever is
  cleaner — keep it a real regression guard, not a tautology).
- **Real acceptance (manual):** `--mode cross-endzone` on SEA_at_AZ play_001 →
  `endzone_*` in cameras.npz, center in the plausible box (near (−115,0,25)),
  overlay tracks, same-field-point proof (a sideline player and its endzone
  match agree on a field point within a few meters). Sideline unchanged.
- `pytest -m "not gpu and not slow"` green; ruff clean. Synthetic tests stay
  CPU/fast.

## Out of scope

- Any change to `solve_fixed_center` / the ICP structure / orchestration / CLI.
- A general roll-agnostic matcher for arbitrary view rotations (only the
  90°-family the endzone uses is needed; the 4-roll sweep covers it).
- Player-detection quality improvements (foot-point noise is absorbed by the
  robust solve + hundreds of inliers).

## Risks

- **20 px bootstrap seed still too coarse for `solve_fixed_center`:** if the
  first solve doesn't lock despite the 405 tight inliers, tighten the seed gate
  or add one extra ICP round — but the measured 405 inliers strongly suggest it
  seeds. The real acceptance is the confirmation.
- **Roll convention mismatch** between `_hyp_cam`'s `_rz` and the frame's actual
  rotation: pinned by the unit test (exact `_rz(90) @ R`) and the fact that the
  diagnostic already measured `roll=90` as the winner on real data with the
  same `_rz`.
