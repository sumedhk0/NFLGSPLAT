# NFLGSPLAT — Project Handoff / Agent Context

**Last updated:** 2026-07-24. Purpose: give any AI agent (OpenHands, Claude Code,
etc.) full working context WITHOUT the original chat history. Read this first,
then `docs/agent-context/` (ported memory), then the specs/plans referenced below.

## What this project is

NFL broadcast **All-22** footage (synced sideline + endzone cameras, from
pro.nfl.com Film Room) → camera calibration → free-viewpoint **Gaussian
Splatting**. Local dev on Windows; GPU jobs on **GT PACE Phoenix**.

## Hard rules (do not violate)

- **Never commit real NFL video/frames.** `data/` and `kp_eval/` are gitignored.
  Diagnostic images go to `C:\Users\sumedh\diag\` or scratch, OUTSIDE the repo.
- **All GPU jobs run on PACE Phoenix `embers` partition** (account
  `paceship-pso`), checkpoint for preemption. See `docs/agent-context/`.
- **Fail loud** with `SetupError`/`CalibrationError` + an actionable pointer.
  No silent fallback that changes numerical results.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Workflow: feature branch off `main` → TDD (test first) → `--no-ff` merge →
  delete branch. Commit/push only when the user asks.
- Process skills used: brainstorming → writing-plans → subagent/inline execution
  (docs under `docs/superpowers/`).

## Current state (2026-07-24)

**Sideline camera: SOLVED & valid.** `cameras.npz` for
`data/2025/week_04/SEA_at_AZ/play_001` has a good sideline track (816/986
frames, camera ≈ (−3.6, 80.5, 35.9) m). See spec
`docs/superpowers/specs/…focal-pose-from-homography…` and
`docs/agent-context/next-cycle-focal-pose-from-homography.md`.

**Endzone camera: IN PROGRESS — jersey-identity approach just merged, real
acceptance PENDING.** History of three approaches (full detail in
`docs/agent-context/endzone-camera-bringup.md`):
1. Field markings — EXHAUSTED (yard-number identity unreadable at the endzone
   angle; ~9 prototypes). Do not retry.
2. Geometric cross-camera (per-frame foot-point Hungarian) — FAILED on real
   data: even the true camera reprojects foot correspondences at ~87px median,
   the Y-reflection outscores truth, 0 frames solve. Branch
   `cross-camera-roll-fix` left UNMERGED (superseded).
3. **Jersey-IDENTITY cross-camera — MERGED to main (2026-07-24).** This is the
   live approach.

### The jersey-identity endzone approach (merged, how it works)

- **Cross-camera identity is geometry-free.** `identity.registry.resolve_tracks(
  id_col="track_id")` per camera assigns `player_uid` from jersey OCR + team
  color (roster optional; `OcrOnlySource` synthesizes `{season}_{team}_{jersey}`
  uids). JOIN both cameras on `player_uid`. This breaks the chicken-and-egg that
  `tracking.cross_cam_reid.reid_pipeline` has (it needs both cameras calibrated).
- **Correspondences:** per shared `player_uid`, per co-observed frame →
  (sideline foot projected to field `(X,Y,0)`, endzone foot pixel). Sideline
  field point is temporally smoothed (frame-aware rolling median).
- **Multi-play shared-center solve:** the endzone camera is a fixed tripod, so
  ONE center C is shared across all first-half plays. Concatenate all plays'
  frames, solve via the existing `joint_solve.solve_fixed_center`
  (`_frame_data_override` path). **Native endzone pixels: `view_deg=0`, NO
  rotation, NO roll** (the old 90° rotation only served the dead field-marking
  detector).
- **EndzonePrior** (from `meta.yaml` `endzone_prior:` block) bounds C to the
  correct side (behind the target endzone, X<0 in this play's world frame),
  excluding the Y-reflection.

### Key files (jersey-identity)

- `nfl_gsplat/calibration/endzone_identity.py` — `identity_correspondences`,
  `field_positions_by_uid`, `endzone_pixels_by_uid` (join on player_uid).
- `nfl_gsplat/calibration/endzone_multiplay.py` — `EndzonePrior` +
  `solve_endzone_identity` (multi-play shared-C driver).
- `nfl_gsplat/calibration/identity_precompute.py` — `assign_identity_columns`
  (global team split + player_uid, geometry-free).
- `nfl_gsplat/calibration/run_autocalib.py` — `build_endzone_identity_from_plays`.
- `scripts/03c_identity_tracks.py` — PACE precompute (detect+track + jersey OCR
  + identity → tracks.parquet with player_uid). Runs in `nfl_smplx` env.
- `scripts/02_autocalibrate.py` — `--mode identity-endzone --play-dirs a,b,c`.
- `joint_solve.solve_fixed_center` gained default-preserving `center_bounds`
  (bounds the multi-start to the prior box) + `audit_drop_px` (frame-consistency
  gate; player feet need ~15px, not the 6px of sub-pixel field markings).
- Spec: `docs/superpowers/specs/2026-07-22-jersey-identity-endzone-calibration-design.md`
- Plan: `docs/superpowers/plans/2026-07-22-jersey-identity-endzone-calibration.md`

### Validation status

- **Synthetic multi-play: recovers C to 0.062 m**, 75/75 frames, reflection
  excluded (load-bearing test). All fast tests pass (+2 pre-existing OpenCV
  `test_field_detect.py` failures, unrelated/environmental).
- **Real single-play smoke test (local, easyocr proxy):** the cross-camera JOIN
  works on real footage — 8 shared player_uids across cameras on play_001. (It
  also caught a real crash bug in the frame-aware smoothing on duplicate
  uid-in-frame, since FIXED + merged.) Full residual measurement was in progress
  at handoff time.

## NEXT STEP — real acceptance (the actual gate, NOT yet passed)

1. **Get several first-half SEA@AZ plays** (clean `sideline.mp4`/`endzone.mp4`
   like play_001) into `data/2025/week_04/SEA_at_AZ/play_NNN/`. First half only
   (camera may move at halftime). More plays = better-pinned shared C.
2. **PACE precompute per play** (env `nfl_smplx`, GPU on `embers`, PaddleOCR):
   `python scripts/03c_identity_tracks.py <play_dir> --season 2025 --device cuda`
   Fills `track_id`, `jersey_number_ocr`, `team`, `player_uid` in tracks.parquet.
   (Local `tracks.parquet` for play_001 is DETECT-ONLY: track_id=-1, needs this.)
3. **Add `endzone_prior:` to each `meta.yaml`:**
   ```yaml
   endzone_prior:
     x_range: [-150, -60]   # X<0 = behind the target endzone in this world frame
     y_range: [-15, 15]
     z_range: [10, 60]
     focal_range: [1500, 3500]
   ```
4. **Solve:** `python scripts/02_autocalibrate.py --play-dir <play_001>
   --play-dirs <p1,p2,...> --mode identity-endzone`
5. **Verify:** endzone center in the prior box; same-field-point proof (a
   jersey-matched player's sideline field point and its endzone back-projection
   agree within a few meters). If back-projection lands on the wrong side, flip
   the prior X-sign.

### MEASURED BLOCKER (2026-08-10, real local run on play_002)

Ran the full identity precompute on the local GPU (`03c`, easyocr on CUDA,
6 min/play). Result:

| metric | value |
|---|---|
| tracks with a voted jersey | 49 / 236 |
| distinct uids: sideline / endzone | 16 / 26 |
| **uids shared across BOTH cameras** | **5** |
| **frames with >= 4 shared uids in both** | **0 / 327** (median 1) |

`solve_fixed_center` needs **>= 4 correspondences per frame**
(`build_frame_data(min_corrs=4)`), so as it stands almost no frame qualifies and
the solve cannot run — this is the same wall the geometric approach hit, now
from the identity side.

Note **multi-play aggregation does NOT fix this**: more plays add more frames,
each still carrying ~1 shared uid. The binding constraint is *per-frame*
density, not total correspondence count.

Why yield is thin: the endzone sees players' backs (numbers legible, 26 uids)
while the sideline sees side-on profiles where the number is often not readable
(16 uids) — and a correspondence needs the SAME player read in BOTH views. The
torso-crop + upscale fix lifted per-crop reads ~15x and is already in, but the
sideline viewing angle is the real limiter.

Options, roughly in order of cost:
1. **Relax the per-frame requirement.** Add temporal smoothness on rotation/
   focal so frames with 1-2 correspondences still constrain the solve, and drop
   `min_corrs` accordingly. Touches `joint_solve` (the load-bearing solver) —
   needs care and its own spec.
2. **Raise sideline yield**: OCR more crops per track (`--ocr-top-k`), lower
   `min_ocr_conf`, or propagate one confident read along a whole track (already
   done per track — the gap is tracks never read at all).
3. **Constrain with the roster** (`data/rosters/2025/rosters.parquet` is
   fetched): snap OCR misreads to valid SEA/ARI jerseys and use team colour to
   halve the candidate set, converting weak reads into usable identities.
4. Fall back to the geometric trajectory matcher for the frames identity cannot
   cover, using identity-matched players as anchors.

### Watch during acceptance (data-dependent review findings)

- **Team-color split may partition by CAMERA not team** (lighting differs
  sideline vs endzone). It fails loud (too-few-frames `CalibrationError`) rather
  than silently wrong. If so: compute the 2-color split per camera then align
  labels, or use a roster.
- **Real jersey-OCR yield must clear the `min_frames` floor.** easyocr proxy was
  thin (per-detection); PaddleOCR + per-track voting is much stronger.

## Deferred / backlog

- Vanishing-point field geometry as an independent endzone cross-check (spec
  out-of-scope v1).
- Populate `data/rosters/2025/` via `scripts/fetch_roster.py --season 2025`
  (needs `nfl_data_py` + network) → stronger identity than OCR-only.
- Graceful per-camera calibration degradation (a good sideline surviving an
  endzone failure).
- Downstream: player 3D placement, pose (SMPL-X on PACE embers), splat training.
- `cross-camera-roll-fix` branch is superseded; delete if not revisiting field
  markings.

## Environment

**Local CUDA box (set up 2026-08-10) — precompute no longer needs PACE.**
RTX 4080 Laptop, 12 GB, driver 595.71 / CUDA 13.2. Calibration + tracking + OCR
peak ~2-3 GB, so the whole precompute runs here. Keep PACE for Gaussian-Splat
training, where 12 GB is tight.

- **Interpreter: use the venv, not the system Python.**
  `C:\venvs\nflgsplat\Scripts\python.exe` (Python 3.14.7). It lives at a SHORT
  path on purpose: `LongPathsEnabled=0` on this machine, and deep CUDA wheel
  paths blow the 260-char limit when installed under the OneDrive repo path.
- Stack: torch 2.11.0+cu128 (CUDA verified), ultralytics 8.4.117, numpy 2.4.4,
  pandas 3.0.5, opencv-python 5.0.0, scipy 1.18, pyarrow 25.0.1, easyocr,
  rapidocr + onnxruntime-gpu, omegaconf 2.3.1.
- ffmpeg/ffprobe 9.0 via winget (`Gyan.FFmpeg`). If `ffprobe` is not found, add
  `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-9.0-full_build\bin`
  to PATH (winget edits PATH only for new shells).
- **Do NOT `pip install -e .` here:** `pyproject` pins `numpy<2` for paddle/mmcv,
  which would downgrade numpy out from under torch 2.11. Install deps directly;
  `python -m pytest` from the repo root puts the repo on `sys.path` anyway.
- **Jersey OCR: paddlepaddle has no Python 3.14 wheels at all.** Backends are
  pluggable (`JerseyOCRConfig.backend`, `03c --ocr-backend`); `auto` resolves to
  **easyocr on CUDA** here — measured 37.7 crops/s vs rapidocr's 1.3 (rapidocr's
  ONNX CUDA provider never engages, so it stays CPU-bound).
- Tests: `C:\venvs\nflgsplat\Scripts\python.exe -m pytest -m "not gpu and not slow" -q`
  (359 passing on this stack). Lint: `... -m ruff check nfl_gsplat tests scripts`
  — one pre-existing `B008` (typer.Option defaults) repo-wide.

Videos: `data/all22/sea_at_az_wk4` and the play dirs live in OneDrive and arrive
as **cloud-only placeholders** on a new machine. Copying/reading hydrates them
(~1.4 GB). Hardlinks do NOT survive a OneDrive sync — repopulate play dirs by
copying, not linking.
