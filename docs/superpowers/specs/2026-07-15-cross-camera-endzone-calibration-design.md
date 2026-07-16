# Cross-Camera Endzone Calibration (via shared players)

**Date:** 2026-07-15
**Status:** Approved (design); implementation plan pending
**Builds on:** the fixed-center joint solve
([2026-07-06](2026-07-06-fixed-center-joint-solve-design.md)), player masking /
detection ([2026-07-12](2026-07-12-player-masks-for-calibration-design.md)), and
the rotate-90 endzone infra ([2026-07-08](2026-07-08-endzone-rotate90-calibration-design.md)).

## Problem

Endzone-camera calibration from field markings is exhausted (measured to the
floor 2026-07-15: even with rotation + player masking + consensus labeling,
yard-line identity comes out wrong under the steep down-field perspective —
labels 67–210 px off, no rigid camera fits). But the endzone camera IS
solvable from a source the markings can't provide: the **players**. They appear
in both the already-calibrated sideline view and the endzone view, spread
across the whole image, and their identities are unambiguous per instant.

## Goal

`cameras.npz` gains a plausible `endzone_*` camera (|X| 50–150 m, |Y| ≤ 40 m,
Z 10–80 m) solved from cross-camera player correspondences, with its overlay
tracking the painted field AND — the real proof — a sideline player and its
matched endzone detection projecting to the same field point. Sideline
calibration is a prerequisite (must already be in `cameras.npz`) and is never
modified. Coverage is measured, not gated.

## Key insight

The endzone camera is also fixed (tripod: pan/tilt/zoom only). So this is NOT
a new solver — it is the existing `solve_fixed_center` fed correspondences from
cross-camera instead of field markings. The sideline camera turns each player's
foot pixel into a true field point `(X, Y, 0)`; paired with the same player's
endzone foot pixel, that is exactly the `(world_xyz, uv)` pair the joint solve
consumes. Player spread across the endzone image is the well-conditioning the
field markings lacked.

## Data flow

```
sideline cameras.npz (calibrated)         tracks.parquet (both cams, foot points)
        │                                          │
        ▼                                          │
project_foot_points_to_field  ──►  sideline player (X,Y,0) per frame
        │                                          │
        ▼        hypothesize endzone camera (grid + measured C)
   project (X,Y,0) → predicted endzone foot pixels
        │                                          │
        ▼   Hungarian match to endzone foot detections (dist-gated)
   (world (X,Y,0), endzone foot uv) correspondences per frame
        │
        ▼   solve_fixed_center (unchanged) → endzone camera
        │                                          │
        └────────── re-match with improved camera, re-solve (ICP, 2–3 rounds) ──┘
        ▼
assemble_track_from_results → cameras.npz  (endzone_*)
```

## Components (new module `nfl_gsplat/calibration/cross_cam_calib.py`)

- **Foot→world (reuse):** `cross_cam_reid.project_foot_points_to_field(tracks, cameras)`
  already adds `foot_x_m, foot_y_m` by projecting `(foot_u, foot_v)` to Z=0 via
  the calibrated camera. Used on the SIDELINE tracks only.
- **`match_endzone_to_field(world_pts, endzone_feet, endzone_cam_frame, *, max_px)`**
  — project the frame's sideline-derived `world_pts (N,3)` through a hypothesized
  endzone `(K,R,t)` to predicted pixels; `scipy.optimize.linear_sum_assignment`
  against the frame's `endzone_feet (M,2)`; keep matches within `max_px`.
  Returns `[(world_xyz, (u,v)), …]`.
- **`corrs_from_matches(sideline_world_by_frame, endzone_feet_by_frame, endzone_cam) -> corrs_by_frame`**
  — run the match per frame under the current endzone camera estimate.
- **`solve_endzone_cross_camera(sideline_world_by_frame, endzone_feet_by_frame, image_size, *, init_cams, max_rounds=3) -> list[CalibrationResult|None]`**
  — the ICP loop: from an initial endzone camera, build correspondences,
  `solve_fixed_center`, re-match with the new camera, re-solve; stop when the
  matched-reprojection stabilizes or `max_rounds`. Fails loud on divergence /
  too-few matches.
- **Orchestration** (`run_autocalib` gains `build_endzone_from_sideline` or a
  `--mode cross-endzone`): load sideline `cameras.npz` + `tracks.parquet`,
  project sideline feet, gather endzone feet (rotated into the endzone working
  frame like the pretrained path — foot pixels get `rotate_uv`), run the solve,
  de-rotate results, write `endzone_*` into `cameras.npz` (merge, don't clobber
  sideline).

## Initialization (the bootstrap's load-bearing risk)

Matching needs a starting endzone camera good enough that projected players
land near their detections. Reuse the multi-start pattern from `joint_solve`:
candidate centers = the behind-endzone grid (`_GRID_EZ_*`) **plus the measured
C=(−111,−21,64) seeded first**; per candidate, run ONE match+solve round on a
frame subsample, score by matched-reprojection, keep the winner, then refine on
all frames. Per-frame rotation/focal seeded by the same look-at/span heuristic
`_init_frame` already uses.

## Error handling

- Sideline camera absent/low-confidence in `cameras.npz` → `SetupError`
  (prerequisite: run sideline calibration first).
- Missing `tracks.parquet` → `SetupError` pointing at `03b_detect_players.py`.
- Fewer than a floor of matched players per frame across the play, or the ICP
  matched-reprojection not improving → `CalibrationError` naming the endzone
  camera ("cross-camera matching did not converge") — this is also the
  fail-loud signal for a frame-sync mismatch between the two videos.
- No silent fallback to field-markings fusion.

## Data assumption

The two videos are frame-synchronized (same index = same instant). Both are
1302 frames, strongly implying one synced clip; the acceptance verifies via the
same-field-point check (a sideline player and its endzone match must project to
the same `(X,Y)`). Bad sync → matches never converge → the fail-loud above.

## Testing

- **Unit:** `project_foot_points_to_field` round-trip (place a world point,
  project to a synthetic camera's foot pixel, recover the world point);
  `match_endzone_to_field` on a synthetic 2-camera scene (known cameras +
  players → correct Hungarian pairs, distance gate drops a planted outlier).
- **Synthetic integration:** fixed synthetic endzone camera + ~20 moving
  players over ~40 frames; project into both a known sideline camera and the
  endzone camera; feed the whole bootstrap (foot→world via sideline, match,
  solve) → recover the endzone camera center < 0.5 m, focal within 2 %, from a
  grid init that does NOT start at the truth.
- **Real acceptance (manual):** on SEA_at_AZ play_001 (sideline `cameras.npz`
  + `tracks.parquet` exist) → `endzone_*` written, center in the plausible box,
  overlay tracks the painted field; the same-field-point proof: sampled matched
  (sideline player, endzone detection) pairs project to within a few meters.
- `pytest -m "not gpu and not slow"` green; ruff clean. Synthetic tests are
  CPU/fast (no YOLO, no video — construct tracks DataFrames directly).

## Out of scope

- Jersey-OCR-based matching (rejected: heavy PaddleOCR dep, unreliable on small
  broadcast players; geometric bootstrap needs no new deps).
- Cross-camera re-ID / global player IDs (`cross_cam_reid` already exists for
  that; this reuses only its foot→world projection).
- Simultaneously refining the sideline camera (it is a fixed, trusted input).
- Ball-based correspondence, multi-play, or >2 cameras.
- Player 3D placement / pose (later; this calibration unblocks them).

## Risks

- **Bootstrap basin:** if every grid candidate projects players too far from
  their detections to match, the ICP never seeds. Mitigation: the measured
  C=(−111,−21,64) is seeded first; the distance gate starts generous and
  tightens across rounds.
- **Player-detection quality on the endzone view** (small players, occlusion):
  foot points are noisy; the robust loss + rescue-first in `solve_fixed_center`
  reject bad matches, and hundreds of player-instances across the play
  overdetermine the fixed center.
- **Frame-sync** (above): caught by the same-field-point acceptance check and
  the converge-or-fail-loud gate.
