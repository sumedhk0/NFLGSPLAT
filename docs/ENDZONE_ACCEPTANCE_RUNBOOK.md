# Endzone Identity Calibration — Acceptance Runbook

Goal: recover the endzone camera from jersey-identity correspondences across
several first-half SEA@AZ plays. See `docs/HANDOFF.md` for the approach.

## Plays (organized 2026-07-24)

`data/2025/week_04/SEA_at_AZ/play_001` … `play_043` (play_002–043 hardlinked
from `data/all22/sea_at_az_wk4`, all FIRST HALF, 59.94 fps, 1080p). Each has
`sideline.mp4`, `endzone.mp4`, `meta.yaml` (with an `endzone_prior:` block).

- **34 scrimmage** plays; **8 special-teams** (play_006/021/022/023/031/032/036/042
  — kicks/XP/punts, sparser player spread — skip for the first run).
- **play_001** already has a calibrated sideline + roboflow kps; it needs only
  step 3 (identity). play_002+ need all steps.

**Recommended first batch (~10, varied pass/run, good player spread):**
`play_001 play_002 play_004 play_005 play_009 play_011 play_014 play_016 play_026 play_029`

## Where this runs now

**All of it runs on the local CUDA box** (RTX 4080 12 GB) — PACE is no longer
required for precompute. Use the venv interpreter, referred to below as `PY`:

```
set PY=C:\venvs\nflgsplat\Scripts\python.exe
```
Jersey OCR uses **easyocr on CUDA** (paddlepaddle has no Python 3.14 wheels);
pass `--ocr-backend easyocr` explicitly if you want to pin it. See
`docs/HANDOFF.md` → Environment for the full stack and gotchas.

**Running a stage script directly needs `PYTHONPATH`** — `python scripts/x.py`
puts only `scripts/` on `sys.path`, and this repo is deliberately not
pip-installed here (its `numpy<2` pin would break the cu128 torch):
```
set PYTHONPATH=%CD%
```
`run_precompute_batch.py` sets this for its children automatically, so the fast
path below needs nothing extra.

## FAST PATH — batch driver (recommended)

`scripts/run_precompute_batch.py` runs one stage across the whole batch
(`--plays BATCH` = the recommended set; play_001's roboflow/sideline are
auto-skipped since already done). Three commands, in order:

```
set PY=C:\venvs\nflgsplat\Scripts\python.exe
set ROBOFLOW_API_KEY=...                                          # step 1 only: internet + API key
%PY% scripts/run_precompute_batch.py --stage roboflow --plays BATCH
%PY% scripts/run_precompute_batch.py --stage sideline --plays BATCH
%PY% scripts/run_precompute_batch.py --stage identity --plays BATCH --device cuda
```
Each stage continues past a failed play and prints an ok/failed summary. Then
the controller runs the final `--mode identity-endzone` solve.

The per-play commands below are the same steps, for debugging a single play.

## Per-play precompute (each play in the batch)

### Step 1 — Roboflow keypoints for the sideline (LOCAL, Windows, needs internet + `ROBOFLOW_API_KEY`)
Only step needing the API. play_001 already has this.
```
set ROBOFLOW_API_KEY=...
python scripts/03_roboflow_precompute.py "data/2025/week_04/SEA_at_AZ/<play>/sideline.mp4" ^
    --out "data/2025/week_04/SEA_at_AZ/<play>/roboflow_kps_sideline.json" ^
    --model-id football-field-key-points-mvmjf/2
```

### Step 2 — Sideline calibration (local or PACE)
Writes the play's `cameras.npz` with the `sideline` track (the endzone solve
projects each play's sideline feet to the field). play_001 already done.
```
python scripts/02_autocalibrate.py --play-dir data/2025/week_04/SEA_at_AZ/<play> \
    --mode pretrained --cameras sideline
```

### Step 3 — Identity precompute (PACE, `nfl_smplx` env, GPU on `embers`)
detect+track + PaddleOCR jersey vote + team/player_uid → `tracks.parquet`.
Needed for EVERY play including play_001 (its tracks are detect-only).
```
python scripts/03c_identity_tracks.py data/2025/week_04/SEA_at_AZ/<play> \
    --season 2025 --device cuda
```
Sanity after: `tracks.parquet` should have non-`-1` `jersey_number_ocr` for a
healthy fraction of tracks and a `player_uid` per track.

## Final — multi-play endzone solve (local or PACE, CPU is fine)
Once the batch has cameras.npz (sideline) + tracks.parquet (player_uid):
```
python scripts/02_autocalibrate.py \
    --play-dir  data/2025/week_04/SEA_at_AZ/play_001 \
    --play-dirs "data/2025/week_04/SEA_at_AZ/play_001,data/2025/week_04/SEA_at_AZ/play_002,...(the batch)" \
    --mode identity-endzone
```
Writes an `endzone` track into each play's `cameras.npz` (sideline preserved).

## Verify (acceptance gate)
- Endzone center in the prior box (`|X|` 60–150, `|Y|` ≤ 15, Z 10–60 m).
- Same-field-point proof: a jersey-matched player's sideline field point and its
  endzone back-projection agree within a few meters. If it lands on the WRONG
  side, flip the `endzone_prior.x_range` sign in every meta.yaml.

## Known risks (from review + the easyocr smoke test)
- **Per-frame density:** the solve needs ≥4 jersey-matched players per frame.
  easyocr proxy gave ~1/frame (too thin, 8 shared uids). PaddleOCR + roster
  should identify most players → enough. If a play still starves, its residuals
  fail loud — add plays / check OCR.
- **Team color split may cut by CAMERA not team** (lighting). Fails loud
  (too-few-frames), not silently wrong. Mitigate with a roster
  (`scripts/fetch_roster.py --season 2025`) or per-camera split + label align.
