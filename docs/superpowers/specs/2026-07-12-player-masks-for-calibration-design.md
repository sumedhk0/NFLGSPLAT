# Player Masks for Calibration (endzone unblock)

**Date:** 2026-07-12
**Status:** Approved (design); implementation plan pending
**Builds on:** [2026-07-08-endzone-rotate90-calibration-design.md](2026-07-08-endzone-rotate90-calibration-design.md)
and the pretrained-hybrid + fixed-center calibration cycles.

## Problem

Endzone-camera calibration fails not for lack of field markings but because
**players corrupt hash-mark detection**. Measured on the rotated endzone frame
(SEA_at_AZ play_001, 2026-07-12): with players unmasked, `detect_hashes`
returns ~220 candidates — most false positives on white jerseys/numbers — and
`fit_hash_rows` RANSAC-fits two **diagonal** rows (angles −16° and −40°
instead of ~0°). Every yard-line × hash-row intersection is then wrong, so all
fused correspondences are wrong (uniform ~194 px, no consistent camera). This
single bug — not the perspective, not identity anchoring — defeated all five
earlier endzone prototypes, because every one ran detection without masking.

Yard-line detection is already clean (median 9 lines/frame). The pretrained
calibration pipeline HAS a `masks_provider` seam that zeroes player boxes out
of the white mask before detection, but the CLI never wires it (a TODO), and
the endzone rotate-90 cycle added a guard that *rejects* masks on rotated
cameras (player boxes would be in unrotated coordinates). Both must change.

## Goal

`--mode pretrained` on the two-camera play writes `cameras.npz` containing
BOTH `sideline_*` and `endzone_*`, with the endzone camera plausible
(|X| 50–150 m, |Y| ≤ 40 m, Z 10–80 m) and its grid overlay tracking the
painted field. Sideline result must be unchanged from the current merged main
(center ≈ (−3.6, 80.5, 35.9) m, ~816 kept frames) — masking sideline is
additive, never regressive.

## Approach

Precompute player boxes once per play (mirroring the Roboflow-keypoint cache),
feed them into calibration as masks (rotating boxes for the endzone view), and
re-measure endzone fusion on the now-clean hash rows before adding any new
fusion logic.

## Components

### 1. Player-box precompute (`scripts/03b_detect_players.py`, local)

- Runs the existing `nfl_gsplat.tracking.detect_track.detect_and_track`
  (YOLOv8 + BoT-SORT) locally per camera and writes `tracks.parquet` to the
  play dir — the same artifact downstream player placement/pose need, so no
  throwaway work. Masking reads only the per-frame boxes from it.
- Boxes stored in **original-frame pixel coordinates** (`bbox_x1..y2` in the
  existing `TRACK_COLUMNS` schema); calibration handles rotation.
- CPU fallback: if BoT-SORT tracking is unavailable/too slow on CPU, a
  `--detect-only` path runs per-frame YOLO detection (no track IDs) and writes
  the same columns with `track_id = -1`. Masking needs boxes, not IDs.
- Lazy `ultralytics` import with the existing `SetupError` pointer.

### 2. Boxes → masks in calibration (`run_autocalib.py` + CLI)

- `boxes_provider_from_tracks(tracks_path) -> Callable[[str], Callable[[int], list[box]]]`
  loads `tracks.parquet` once and returns a per-camera `boxes_for(frame_idx)`
  yielding `[(x1,y1,x2,y2), …]` in original pixels. Missing file → no masking
  (masks are an enhancement, not required) — but the CLI warns loudly so a
  forgotten precompute is visible.
- `build_autocalib_npz_pretrained` already accepts `masks_provider`; the CLI
  now builds it from the boxes cache (convention `<play_dir>/tracks.parquet`,
  or `--player-boxes PATH`) and passes it for both cameras.
- **Box rotation (replaces the reject-guard):** for a camera rotated by `deg`,
  each box is mapped into the working frame by rotating two opposite corners
  via `view_rotation.rotate_uv(...,deg,orig_wh)` and taking min/max. 90/180/270
  are axis-preserving, so the result is an exact axis-aligned box in
  rotated coordinates. The masking then happens on the rotated frame the
  detector actually sees. The `deg != 0` `CalibrationError` guard from the
  rotate-90 cycle is removed.

### 3. Endzone fusion — measure, then extend

- **First:** re-run the existing `fuse_frame` model-keypoint path on
  masked endzone frames. Masking gives correct (near-horizontal) hash rows, so
  line × hash-row intersections become valid; measure usable-frame count and
  whether the joint solve locks a plausible camera. If it clears the solve's
  gates → done, no new fusion code.
- **Only if it still falls short:** add an endzone fusion mode that runs
  `label_lines_by_consensus` (anchored by the highest-confidence identity-
  carrying model keypoint per frame, both directions/territories scored) over
  the masked dense classical lines. This is the approach that reached 89
  usable frames even with corrupted rows; with correct rows its labels should
  be right. Gated behind the measured result of the first step, wired as a
  selectable path (not replacing the sideline fusion).

## Data flow

```
[local, once per play]
  03b_detect_players.py video → tracks.parquet   (original-pixel boxes)

[local calibration]
  02_autocalibrate --mode pretrained
    per camera: boxes_for(fidx) → rotate boxes by deg → mask →
      detect_field_features(masked) → clean hash rows → fuse → joint solve
    → cameras.npz (sideline_* + endzone_*)
```

## Error handling

- Missing `tracks.parquet` when masks were expected → CLI warns (loud), runs
  unmasked (sideline still works; endzone likely fails loud at its solve gate,
  naming the camera — the existing behavior).
- `ultralytics` absent in the precompute script → `SetupError` with the env
  pointer.
- Box-rotation with an invalid `deg` → `view_rotation._check`'s `ValueError`.
- No silent numerical fallback: masking either applies or is absent; it never
  half-applies.

## Testing

- **Unit:** `rotate_box` round-trip (box rotated 90 then 270 = original, to
  ≤1 px); rotated box is axis-aligned and covers the rotated corners;
  `boxes_provider_from_tracks` yields the right per-frame boxes from a synthetic
  parquet; masks_provider path in `build_autocalib_npz_pretrained` receives
  rotated boxes for a rotated camera (monkeypatch capture, following the
  existing wiring-test pattern).
- **Synthetic:** a frame with a bright "player" blob over the hash region —
  unmasked `fit_hash_rows` tilts; masked → rows horizontal (angle ≈ 0).
- **Real acceptance (manual):** precompute boxes for SEA_at_AZ play_001; run
  two-camera calibration; assert `cameras.npz` has `endzone_*`, endzone center
  in the plausible box, overlay tracks; sideline center/coverage unchanged
  (regression guard). Record endzone kept-frame count.
- `pytest -m "not gpu and not slow"` green (detection itself is gpu/slow-marked
  or stubbed); ruff clean.

## Out of scope

- Cross-camera reID, jersey OCR, player 3D placement, pose — later slices
  (this slice's `tracks.parquet` feeds them).
- Masking the hint/learned calibration modes (pretrained only for now).
- Graceful per-camera calibration degradation (a good sideline surviving an
  endzone failure) — noted earlier, still a separate small improvement.
- FieldDetectConfig retuning for the endzone view and SkyCam-cable rejection —
  secondary cleanups, only if masked fusion still underperforms.

## Risks

- **Real hash marks partly occluded by players near the line of scrimmage** —
  masking removes false positives but can't recover truly hidden ticks; the
  joint solve pools frames across the pan, so frames where the hash region is
  clear carry the solve. Measured coverage in step 3 decides if this bites.
- **CPU YOLO too slow for a full game video** — `detect_and_track` runs on the
  whole per-game video for BoT-SORT continuity. Mitigation: the `--detect-only`
  per-frame path over just the play's frame range; or run detection on the
  already-extracted play clip.
- **Box rotation sign error** silently masks the wrong region — pinned by the
  round-trip unit test and the synthetic horizontal-rows test.
