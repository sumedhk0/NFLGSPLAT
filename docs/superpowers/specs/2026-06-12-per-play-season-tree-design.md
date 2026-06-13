# Per-Play Season Tree — Data Layout Redesign

**Date:** 2026-06-12
**Status:** Approved (design); implementation plan pending

## Problem

The current pipeline assumes **one continuous video per game** plus a `plays.yaml`
that defines per-play **frame windows**. Artifacts are split across two trees:
raw input under `data/raw/{game}/` and derived data under `outputs/{game}/{play}/`.
Calibration and field are **per-game** (`outputs/{game}/calib/cameras.json`,
`outputs/{game}/field/field.ply`).

We want a season-scale layout organized by **week → matchup → play**, where each
play folder holds its own two video clips plus all of its created data. This
inverts two core assumptions (continuous video → per-play clips; per-game calib →
per-play calib) and collapses the input/output split into one self-contained
play folder.

## Decisions (locked)

1. **Per-play clips, no frame windows.** Each play folder's `sideline.mp4` +
   `endzone.mp4` ARE the play, already trimmed to the snap. Stages read the whole
   clip; there is no `start_frame`/`end_frame` slicing. `plays.yaml` frame windows
   are removed.
2. **Per-play calibration and per-play field.** Broadcast cameras pan-tilt-zoom
   every snap, so `cameras.json` and `field.ply` live inside each play folder.
3. **Naming:** `data/{season}/week_NN/AWAY_at_HOME/play_NNN/` — zero-padded week
   and play (correct lexical sort), NFL-standard "away at home" with team
   abbreviations.
4. **Season discovery by tree-walk.** The filesystem is the source of truth; the
   season DAG discovers plays by globbing the tree rather than reading an explicit
   `games:` manifest.

## Directory Layout

```
data/2024/                          # season root  (data/{season})
  _library/                         # season-shared avatar/shape cache (cross-play)
  _rosters/                         # nflverse roster data for the season
  _registry.json                    # season identity registry (cross-play vote histograms)
  week_01/
    NO_at_ATL/                      # {away}_at_{home}, team abbreviations
      play_001/
        sideline.mp4                # the two clips ARE the play (pre-trimmed)
        endzone.mp4
        cameras.json                # per-play calibration (CameraIntrinsics/Pose per cam)
        field.ply                   # per-play field reconstruction
        tracks.parquet              # detection -> reid -> jersey progressive enrichment
        entities.json               # resolved player_uid / team / referee / football
        smplestx/                   # per-(cam,frame,uid) SMPLest-X inference cache NPZs
        poses/{uid}.npz             # joint_tfms[T,J,4,4] per uid (FK output)
        ball.npz                    # {xyz[T,3], vel[T,3], visible[T]}
        render.mp4                  # final free-viewpoint render
        meta.yaml                   # fps, gsis_play_id, home/away, season, week
      play_002/
  week_02/
    ...
```

The three `_`-prefixed season-shared resources sit under the season root (the `_`
sorts them above `week_*` and signals "not a week"). The avatar library MUST stay
cross-play: building each `player_uid` once and reusing it across every play in
the season is the central efficiency design, so it cannot live inside play folders.

## `meta.yaml` Schema

One per play. `season` / `week` / `home_team` / `away_team` are also derivable
from the path, but `meta.yaml` is the authoritative record and carries the two
fields that are NOT in the path (`fps`, `gsis_play_id`).

```yaml
season: 2024
week: 1
home_team: ATL
away_team: "NO"        # quote abbreviations — bare NO/ON/NA parse as YAML booleans
fps: 30.0
gsis_play_id: 36       # optional; only used for nflverse participation alignment
```

Validation (fail-loud, per project philosophy): missing file → `SetupError` naming
the path; `home_team`/`away_team` parsed as bool → `SetupError` telling the user to
quote; path-derived teams that disagree with `meta.yaml` → `SetupError` (catch a
mis-filed clip).

## Component Changes

### `nfl_gsplat/paths.py` (rewrite)
Replace `GamePaths` / `PlayPaths` with a single `PlayDir` resolver keyed by
`(season, week, matchup, play_id)`. It exposes every artifact path as a property,
all rooted at the one play directory, plus the three season-shared roots:

- `PlayDir.dir` → `data/{season}/week_{NN}/{matchup}/play_{NNN}`
- `video(cam)`, `cameras_json`, `field_ply`, `tracks`, `entities`, `smplestx_dir`,
  `poses_dir`, `pose(uid)`, `ball`, `render_mp4`, `meta_yaml`
- `library_root` → `data/{season}/_library`
- `rosters_root` → `data/{season}/_rosters`
- `registry_path` → `data/{season}/_registry.json`

A `matchup` is `{away}_at_{home}`. A constructor `play_dir(cfg, season, week,
matchup, play_id)` reads the roots from config with sensible defaults, mirroring
the existing `game_paths` / `play_paths` factory style.

### `nfl_gsplat/utils/plays.py` → `nfl_gsplat/utils/meta.py`
- Remove `PlayWindow` / frame-window logic.
- `PlayMeta` dataclass: `season`, `week`, `home_team`, `away_team`, `fps`,
  `gsis_play_id`, `game_teams` property.
- `load_meta(path) -> PlayMeta` with the validation above.
- Keep the YAML-boolean-abbreviation guard from the current loader.

### Stages (read whole clip)
Stages that currently iterate `[start_frame, end_frame]` now iterate the full clip
via the existing `utils.video.iter_frames(video)`. The `(cam, frame, instance)`
addressing is unchanged; only the frame range source changes.

### Season discovery — `scripts/run_season.py`
Discover plays by globbing `data/{season}/week_*/*_at_*/play_*/` (each must contain
`sideline.mp4` + `endzone.mp4` + `meta.yaml`). Build the per-play work list the
staged DAG submits. Remove the `games:` list from `configs/season.yaml`; keep the
SLURM knobs.

### DAG — `nfl_gsplat/season/dag.py`
- Field recon (old S1, per game) folds into per-play perception (per-play field).
- Calibration remains a **manual, interactive per-play pre-step** (not a SLURM
  stage) — it needs a display and human landmark clicks.
- Staged DAG becomes: **per-play perception+field** (array over plays) →
  **collect_uids** (CPU barrier) → **avatar-build** (array over unique uids) →
  **render** (array over plays). `--dependency=afterok` between stages, `--array`
  within. The one-task-per-uid avatar build keeps the shared `_library` write-safe.

### Scaffolder — `scripts/new_play.py`
CLI that creates a play folder and a `meta.yaml` stub from
`--season --week --away --home --play` (+ optional `--fps --gsis-play-id`), so the
user just drops the two clips in. Idempotent; refuses to overwrite an existing
`meta.yaml` unless `--force`.

### Calibration — `scripts/02_calibrate_cameras.py`
Re-target from `--game` to a play directory (`--play-dir` or
`--season/--week/--matchup/--play`); writes `cameras.json` into the play folder.

## Migration

The in-progress single play maps as:

```
data/raw/game_001/{sideline,endzone}.mp4   ->  data/2024/week_01/NO_at_ATL/play_001/{sideline,endzone}.mp4
data/raw/game_001/plays.yaml               ->  data/2024/week_01/NO_at_ATL/play_001/meta.yaml  (reshaped)
outputs/game_001/play_001/*                ->  data/2024/week_01/NO_at_ATL/play_001/*           (colocated)
library/2024/                              ->  data/2024/_library/
```

No production data exists yet on the new tree, so this is a convention change plus
a code refactor, not a data migration script. The single-play bring-up will be the
first content authored under the new layout.

## Testing

CPU-only unit tests, consistent with the existing suite:
- `paths.py`: `PlayDir` resolves every artifact under the right play dir; season
  roots resolve under `data/{season}/_*`; matchup string round-trips
  `{away}_at_{home}`.
- `meta.py`: loads a valid `meta.yaml`; raises on missing file, boolean team
  abbreviation, and path/meta team disagreement.
- Season discovery: a synthetic tree with two weeks / three plays yields the right
  ordered play list; a folder missing a required file is skipped with a warning.
- `new_play.py`: creates the folder + `meta.yaml`; refuses overwrite without
  `--force`.
- DAG: the per-play submission plan over a discovered tree carries the right
  dependencies and array lengths; `_library` path is the season-shared one.

Keep `pytest -m "not gpu and not slow"` green and ruff clean.

## Out of Scope

- No re-encoding/trimming tooling — the user supplies pre-trimmed per-play clips.
- No change to the avatar/identity/pose algorithms — only where their inputs and
  outputs live and how plays are addressed.
- No multi-season indexing beyond the `data/{season}/` root (one season per root).
