# Jersey-Identity Endzone Calibration

**Date:** 2026-07-22
**Status:** Approved (design); implementation plan pending
**Supersedes (approach):** the geometric cross-camera bootstrap in
[2026-07-15-cross-camera-endzone-calibration-design.md](2026-07-15-cross-camera-endzone-calibration-design.md)
and its roll-fix follow-up
[2026-07-17-cross-camera-bootstrap-roll-fix-design.md](2026-07-17-cross-camera-bootstrap-roll-fix-design.md).

## Problem

The endzone camera cannot be calibrated from field markings (yard-line numbers
are edge-on and unreadable from that angle, so lines can't be labeled) nor from
the geometric player bootstrap that shipped: per-frame foot-point Hungarian
matching produces ~85% mispairs, a Y-reflection scores higher than the truth,
and even the true camera reprojects the matches at ~87px median. Measured on
SEA@AZ week_04 play_001: only 6 of 1302 frames yield a usable per-frame camera;
`solve_fixed_center` keeps 0 frames. Root cause: the sideline camera's weak
(grazing) axis is Y, which is exactly the axis the endzone measures most
sensitively, so foot correspondences are both noisy and, without identity,
mostly wrong.

The fix is to stop *guessing* correspondence geometrically and instead take
**identity** from jersey numbers. Both cameras see the same players; jersey
numbers are legible in both views (measured below); a full jersey-OCR +
team-color + roster identity subsystem already exists in `nfl_gsplat/identity/`
and `nfl_gsplat/tracking/jersey_ocr.py` — it was built for the pose/avatar
pipeline and never wired into calibration. Identity gives correct, reflection-
proof, noise-independent correspondences directly.

### Evidence

- Jersey OCR yield (easyocr proxy — weaker than the pipeline's PaddleOCR — on 12
  sampled frames, per-detection, an under-estimate of per-track voting):
  endzone 20%, sideline 9%. Distinct numbers read: endzone 17, sideline 12.
  **Cross-camera overlap: 6 players read in BOTH cameras** ({18,20,58,63,70,85}).
  Per-track voting over ~1300 frames and PaddleOCR will raise this substantially.
- `identity.registry.resolve_tracks(tracks_df, id_col="track_id")` assigns a
  stable `player_uid` from **jersey + team + roster only — no geometry, no
  camera**. Running it per camera and joining on `player_uid` yields cross-camera
  identity without any endzone camera estimate, breaking the chicken-and-egg in
  `tracking.cross_cam_reid.reid_pipeline` (which needs both cameras calibrated).
- The endzone camera is a fixed tripod: its center C is shared across all
  first-half plays. Aggregating several plays over-determines C and averages out
  per-foot noise.

## Goal

`--mode identity-endzone` on the first-half SEA@AZ plays produces a plausible
`endzone_*` camera in `cameras.npz`: center in the prior box (behind Seattle's
end, near center), overlay tracking the field, and a same-field-point proof (a
jersey-matched player's sideline field point and its endzone back-projection
agree within a few meters). Sideline unchanged. A held-out play's player pixels
are predicted by the recovered camera to within tens of pixels.

## Approach

Identity-first, multi-play, shared-center. **We work in native endzone pixel
coordinates — no 90° rotation and therefore no roll.** The rotation existed only
to feed the field-marking detector, which this approach bypasses; a tripod
endzone camera looking down a level field carries no roll, so the solve runs at
`view_deg=0` exactly like the sideline.

### Data flow

```
per play, per camera:
  video → detect_and_track → tracks (track_id) → jersey_ocr (voted #/track)
        → team_color → resolve_tracks(id_col=track_id) → player_uid per track
  (stored back into tracks.parquet)

calibration:
  sideline tracks ─project foot→field (calibrated), smooth per uid─┐
  endzone  tracks ─native foot pixel──────────────────────────────┤ join on player_uid
                                                                    ▼
                              per co-observed frame: (field XYZ, endzone uv)
                                                                    │
        aggregate over ALL first-half plays  (one shared C)         │
                                                                    ▼
    EndzonePrior (C box behind SEA end, |Y| small; focal range)     │
                                                                    ▼
              solve_fixed_center over the UNION of frames, view_deg=0,
              C bounded to the prior box (reflection unreachable)
                                                                    │
                          re-match tighter, re-solve (ICP)  → endzone camera
                                                                    │
   validation: same-field-point proof + held-out-play pixel prediction
   cross-check: vanishing-point rotation/focal (secondary, optional)
```

### Why it works where the last attempt failed

The solver is unchanged; only its input changes. Before: ~85% mispaired
correspondences from per-frame geometry. Now: correspondences carry a jersey-
verified `player_uid`, so they are correct pairs. Correct pairs + robust loss +
one shared C over many frames/plays → convergence. The prior bounds C to the
correct side, so the Y-reflection minimum is unreachable during the solve.

## Components

- **Precompute for calibration (new script `scripts/03c_identity_tracks.py`,
  or a `--track --jersey` path in `03b_detect_players.py`):** runs
  `detect_and_track` (not `detect_only`) → `jersey_ocr` → `team_color` →
  `identity.registry.resolve_tracks(id_col="track_id")` per camera, writing
  `track_id`, `jersey_number_ocr`, `team`, `player_uid` into `tracks.parquet`.
  Reuses existing modules; runs in the `nfl_smplx` env on PACE embers (GPU:
  detection + PaddleOCR). Roster optional: with `data/rosters/{season}/` present,
  `RosterSource`; otherwise `OcrOnlySource` synthesizes `{season}_{team}_{jersey}`
  uids (still cross-camera-joinable).
- **`nfl_gsplat/calibration/endzone_identity.py` (new):**
  - `field_positions_by_uid(tracks_df, sideline_track, *, cam) → {player_uid: {frame:(X,Y)}}`
    — project sideline feet to field, temporally smooth per uid (rolling median),
    drop off-field / low-confidence.
  - `endzone_pixels_by_uid(tracks_df, *, cam) → {player_uid: {frame:(u,v)}}`
    — native endzone foot pixels (no rotation).
  - `identity_correspondences(tracks_df, sideline_track, *, sideline_cam,
    endzone_cam) → {frame: (world (N,3), uv (N,2))}` — inner-join the two on
    `player_uid` per co-observed frame; exclude refs / OTHER_UID.
- **`nfl_gsplat/calibration/endzone_multiplay.py` (new):**
  - `solve_endzone_identity(corrs_by_play, image_size, prior, *, max_rounds,
    match_px) → per-play list[CalibrationResult|None]` — concatenate all plays'
    frames into one frame set sharing C; call `solve_fixed_center` with C bounded
    to `prior` and `view_deg=0`; ICP re-match/re-solve; split results back per
    play. Fail loud if too few usable frames.
- **`EndzonePrior` (small dataclass, in `endzone_multiplay.py`):**
  `x_range, y_range, z_range, focal_range`, sourced from `meta.yaml`
  (`endzone_prior:` block). One-time step maps "behind Seattle's end" → world-X
  sign by checking the sideline territory/goal-line orientation; documented in
  the play's meta.
- **`scripts/02_autocalibrate.py` (modify):** `--mode identity-endzone` accepts
  multiple play dirs (or a game dir), reads the prior, calls
  `solve_endzone_identity`, writes each play's `endzone_*` into its `cameras.npz`
  (sideline preserved, exactly like `build_endzone_from_sideline`).
- **`nfl_gsplat/calibration/field_vanishing.py` (new, SECONDARY / optional):**
  extract two vanishing points from endzone yard-line + cross-field line pencils
  → rotation + focal for a frame; used to cross-check the identity solve and,
  optionally, seed per-frame rotation. Not required for the core acceptance;
  scoped as a follow-on so the identity solve is the MVP.
- **Reuse unchanged:** `solve_fixed_center` (add C-bounds support if not already
  present — see below), `jersey_ocr`, `team_color`, `registry`,
  `project_foot_points_to_field`, `assemble_track_from_results`.

### solve_fixed_center change (minimal)

The solver currently multi-starts C over a plausibility grid. Add an optional
`center_bounds` (per-axis min/max) that (a) restricts the multi-start grid to the
prior box and (b) is passed as bounds to `least_squares`, so C cannot slide into
the reflected minimum. Default `None` preserves current behavior exactly
(sideline path untouched). This is the only solver change.

## Error handling

Fail loud (`CalibrationError`/`SetupError`), no silent fallback:
- Tracks lack `player_uid` (precompute not run with identity) → point to
  `03c_identity_tracks.py`.
- `EndzonePrior` missing or X-sign unresolved → fail with instructions.
- Too few jersey-matched players across all plays (e.g. < a floor of shared uids
  or < N total correspondences) → fail naming the shortfall (thin OCR / too few
  plays), suggesting adding plays or checking roster.
- Shared-C solve keeps too few frames → fail; the same-field-point proof is the
  numeric gate.

## Testing

- **Synthetic multi-play (identity given):** ground-truth endzone camera + several
  plays of players sharing C; assign known `player_uid`s to both cameras' tracks;
  add foot noise. Assert C recovered < 0.5 m, focals < 2 %, and a **reflection
  decoy** center is excluded by the prior bounds.
- **Identity join unit tests:** `identity_correspondences` inner-joins on
  `player_uid` correctly; refs/`OTHER_UID` excluded; only co-observed frames
  produce pairs; temporal smoothing reduces per-uid field-position variance.
- **`center_bounds` unit test:** with bounds set, `solve_fixed_center` cannot
  return a center outside the box; with `None`, output is byte-identical to
  today on an existing sideline fixture.
- **OCR-only identity path test:** with no roster, `resolve_tracks` synthesizes
  `{season}_{team}_{jersey}` uids that join across cameras for shared numbers.
- **Real acceptance (manual):** run identity precompute (PACE) on the first-half
  SEA@AZ plays, then `--mode identity-endzone` → `endzone_*` in the prior box,
  overlay tracks, same-field-point proof within a few meters / tens of px,
  held-out-play pixel prediction sane. Sideline unchanged.
- `pytest -m "not gpu and not slow"` green; ruff clean. Synthetic tests are
  CPU/fast and mock OCR (no PaddleOCR dependency in unit tests).

## Scope / decomposition

Single coherent goal (endzone camera from jersey identity, multi-play). One spec.
Explicitly **out of v1 scope** (future follow-ons):
- Vanishing-point calibration beyond a cross-check (it could later calibrate the
  endzone independently, or replace the sideline dependency).
- The geometric trajectory matcher as a fallback where OCR yield is thin.
- Populating `data/rosters/` for 2025 (use OCR-only until fetched).
- Second-half plays (camera may be repositioned; different shared C).

## Risks

- **OCR yield per play too thin after voting.** Mitigation: multi-play aggregation
  (few shared players × many plays still over-determines C); PaddleOCR > easyocr;
  per-track top-K voting >> per-detection. Fail-loud floor surfaces it early.
- **OCR misreads create false correspondences.** Mitigation: the `registry` +
  `team_color` (+ roster when present) snap misreads to valid team jerseys and
  gate by team; robust loss absorbs the residual few; ICP tightening drops them.
- **Foot-point semantics differ between views** (side vs. rear bbox-bottom) — a
  small systematic per-camera offset that does not average out. Mitigation:
  accept it as a bounded nuisance for v1; if it dominates residuals, add a
  per-camera foot offset later. Flagged, not solved in v1.
- **Shared-C assumption** (camera fixed across first-half plays) unverified;
  if violated the solve fights itself. Mitigation: first-half only; the
  per-play residuals will expose a drift.
- **X-sign / prior mis-set** would bound C to the wrong side and fail loud
  (not silently wrong) — acceptable.
