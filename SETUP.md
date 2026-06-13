# Setup

Three things must be in place before the pipeline will run. None of them can be automated. All of them are detected at runtime and produce a precise pointer back to the relevant section below.

---

## §1. Conda environments

The pipeline uses four conda environments to isolate conflicting CUDA / PyTorch pins across SMPLest-X, gsplat+nerfstudio, LHM++, and 3DGS-Avatar.

```bash
bash scripts/00_setup_environments.sh
```

This creates:

| Env | Purpose | PyTorch | CUDA |
|---|---|---|---|
| `nfl_smplx` | SMPLest-X pose inference | 2.1 | 12.1 |
| `nfl_gsplat` | Field reconstruction + compositing | 2.3 | 12.1 |
| `nfl_lhm` | LHM++ feed-forward avatars | 2.1 | 12.1 |
| `nfl_avatar` | 3DGS-Avatar per-hero optimization | 2.0 | 11.8 |

Version pins are exact in `envs/*.yml`. Do not upgrade blindly — `gsplat` and `nerfstudio` periodically break each other.

**`ERROR: Failed to build 'mmcv'` (login nodes / no nvcc).** `mmcv` has compiled
CUDA ops; from PyPI it builds from source and needs a CUDA toolchain the login
node lacks. `environment_smplx.yml` already points pip at OpenMMLab's prebuilt
wheel index (`--find-links …/cu121/torch2.1/…`). If it still tries to compile
(e.g. a different torch), install it with `mim` inside the env, which auto-detects
the right binary wheel:

```bash
conda activate nfl_smplx
pip install -U openmim && mim install "mmcv==2.1.0"
# then re-run: bash scripts/00_setup_environments.sh --only nfl_smplx
```

**`ERROR: Failed to build 'chumpy'` → `ModuleNotFoundError: No module named 'pip'`.**
`chumpy` 0.70's `setup.py` imports `pip`, which isn't available inside pip's
isolated build env. `00_setup_environments.sh` installs it (and VPoser) in a
post-build step with `--no-build-isolation`; if you hit it manually:

```bash
conda activate nfl_smplx
pip install -U pip setuptools wheel
pip install --no-build-isolation chumpy==0.70
```

(VPoser / `human-body-prior` is intentionally not installed — the pose fit uses
an L2 prior surrogate, so it is not a dependency.)

---

## §2. Body models (SMPL-X and SMPL)

These are license-gated and require manual download.

1. Register at **https://smpl-x.is.tue.mpg.de/** and accept the license.
2. Download **"SMPL-X v1.1 (NPZ+PKL)"**.
3. Register at **https://smpl.is.tue.mpg.de/** and accept the license.
4. Download **"SMPL v1.1.0"** (needed for SMPLest-X compatibility).
5. Place files as:

```
data/body_models/
├── smplx/
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
└── smpl/
    ├── SMPL_NEUTRAL.pkl
    ├── SMPL_MALE.pkl
    └── SMPL_FEMALE.pkl
```

The pipeline checks for `data/body_models/smplx/SMPLX_NEUTRAL.npz` at startup. Missing file → `SetupError` pointing to this section.

---

## §3. Camera calibration (human-in-the-loop)

Every camera pair must be calibrated once per keyframe. NFL broadcast cameras pan and zoom during plays, so we re-calibrate at keyframes (default: once per play start) and linearly interpolate extrinsics between them. Calibration is a **manual per-play pre-step** — run it on a play folder before submitting that play to the pipeline.

### First-time run

```bash
python scripts/02_calibrate_cameras.py --play-dir data/2024/week_01/NO_at_ATL/play_001
```

For each camera, a clickable OpenCV window shows a reference frame overlaid with the list of expected NFL field landmarks (yard-line intersections with sidelines and hash marks, pylons, goalpost bases). Click each visible landmark. Press `s` to save, `q` to quit.

Annotations are written to `<play-dir>/cameras.json`. The PnP solver runs immediately and reports the reprojection RMS. If RMS > 5 px the script refuses to save — re-click or add more landmarks.

**Minimum annotations per camera: 6.** With 10+, bundle adjustment via `cv2.calibrateCamera` refines intrinsics too.

### Adding a keyframe for mid-play zoom changes

```bash
python scripts/02_calibrate_cameras.py --play-dir data/2024/week_01/NO_at_ATL/play_001 --camera sideline --keyframe-frame 4200
```

Extrinsics between keyframes are linearly interpolated. This is the documented limitation — for smoother tracking of aggressive pans, annotate more keyframes.

---

## §4. Non-gated model weights

```bash
bash scripts/01_download_models.sh
```

Downloads:

- **SMPLest-X-H32** checkpoint (from SMPLer-X release on GitHub)
- **LHM-1B** and **LHM-MINI** weights (Alibaba OSS — slow from US, `aria2c` with resume is used)
- **3DGS-Avatar** release weights
- **YOLOv8x** person detector (auto-download via ultralytics on first use)
- **Football-tuned YOLOv8** ball detector (roboflow public dataset, fine-tuned locally — weights cached at `weights/ball_yolov8.pt`)
- **ViTPose** weights via mmpose

All downloads are resumable. Re-running the script is idempotent.

---

## §5. Raw video / data layout

Each play lives in its own self-contained folder. Scaffold one with `scripts/new_play.py`, then drop the two pre-trimmed clips in.

```bash
python scripts/new_play.py --season 2024 --week 1 --away NO --home ATL --play play_001
# creates data/2024/week_01/NO_at_ATL/play_001/ with a meta.yaml stub
```

### Per-play tree

```
data/2024/                          # season root
  _library/  _rosters/  _registry.json   # season-shared (cross-play)
  week_01/
    NO_at_ATL/                      # AWAY_at_HOME
      play_001/
        sideline.mp4  endzone.mp4   # the two clips ARE the play (pre-trimmed)
        cameras.json  field.ply     # per-play calibration + field
        tracks.parquet  entities.json  smplestx/  poses/{uid}.npz  ball.npz
        render.mp4
        meta.yaml                   # season/week/home_team/away_team/fps[/gsis_play_id]
```

### `meta.yaml` schema

```yaml
season: 2024
week: 1
home_team: "ATL"   # abbreviation, quoted
away_team: "NO"
fps: 29.97
# gsis_play_id: "2024090800-1234"   # optional; links to nflverse play-by-play
```

After scaffolding, drop the two pre-trimmed broadcast clips (`sideline.mp4`, `endzone.mp4`) into the play folder and run calibration (§3). The pipeline validates that both clips exist and have matching frame counts (±2 frames tolerance for broadcast jitter).

---

## §6. SLURM configuration

Edit `configs/pipeline.yaml` under the `slurm:` key to match your cluster:

```yaml
slurm:
  account: your_account
  partition: your_gpu_partition
  gpu: h100:1
  qos: embers              # PACE: free, preemptible backfill QOS (vs paid inferno)
  requeue: true            # auto-restart jobs preempted off embers
  time_field: "02:00:00"
  time_play: "01:00:00"
  mem: "64G"
```

The `scripts/slurm/*.sbatch` files template these values via `envsubst` on launch.

**embers vs inferno (PACE Phoenix).** `embers` is the free, preemptible backfill
QOS: jobs run on idle nodes at no charge but are killed/requeued when a paid
`inferno` job needs the node. The season DAG sets `qos: embers` + `requeue: true`
by default, so every job (`run_season.py` and direct `sbatch scripts/slurm/*.sbatch`)
lands on embers and auto-restarts on preemption — safe because each stage skips
already-cached work. Set `qos: inferno` to run against your paid allocation
instead (faster start, no preemption). Note embers caps walltime at ~8h, which the
4h avatar stage fits under.

---

## §7. Verifying the setup

```bash
# Should pass on CPU:
pytest tests/test_calibration.py -v
pytest tests/test_triangulation.py -v
pytest tests/test_ply_merge.py -v

# Full synthetic pipeline, CPU-only with mocked LHM++:
pytest tests/test_pipeline_smoke.py -v
```

If all four suites pass, scaffolding and the numerical foundations are sound. Real-video runs require the GPU envs + body models + annotations above.

---

## §8. Env-gated adapters (SMPLest-X, LHM++, 3DGS-Avatar)

Three heavy external models are treated as *env-gated adapters* rather than vendored code:

| Model | Wrapper module | External repo path (default) | Runs in |
|---|---|---|---|
| SMPLest-X-H32 | `nfl_gsplat/pose/smplestx_infer.py` | `third_party/SMPLer-X/` | `nfl_smplx` |
| LHM-1B / LHM-MINI | `nfl_gsplat/avatars/lhm_wrapper.py` | `third_party/LHM/` | `nfl_lhm` |
| 3DGS-Avatar | `nfl_gsplat/avatars/gdgs_avatar_train.py` | `third_party/3dgs-avatar-release/` | `nfl_avatar` |

Each wrapper in-tree is a **stable seam**: it validates prerequisites (weights + repo checkout), sets up the call signature, and then shells out to the real adapter via `scripts/04_process_play.sh`. The body of each `infer_*` / `generate_*` / `train_*` function raises `NotImplementedError` if imported outside its conda env — this is deliberate so unit tests and CI can still import the pipeline on CPU without pulling torch, CUDA, or gated weights.

If you see:

```
NotImplementedError: SMPLest-X adapter is env-gated; run inside the nfl_smplx conda env
via scripts/04_process_play.sh. See SETUP.md §8 for the adapter wiring.
```

Either you are running a production command outside the right conda env, or `scripts/01_download_models.sh` has not finished cloning the external repo. Check:

1. `conda activate nfl_smplx` (or `nfl_lhm`, `nfl_avatar` depending on the stage).
2. `ls third_party/SMPLer-X/` (or `LHM/`, `3dgs-avatar-release/`) is non-empty.
3. The stage is being invoked through `scripts/04_process_play.sh`, not imported directly from the CI env.

**Why this design.** The adapters pin incompatible PyTorch / CUDA versions; vendoring any one of them would force the entire pipeline onto its pins and break the others. Env-gating is what lets all four coexist.

## §9 — Roster prior + per-player avatar/shape library (season-scale reuse)

For multi-game runs (e.g. one team's 17-game season), the pipeline reconstructs
each player **once** and reuses them across every play and game. Two pieces:

### Roster / participation prior (optional, recommended)

The roster turns player recognition from open-set re-ID into constrained
classification against the ~22 players actually on the field per play. Fetch it
once per season via nflverse:

```bash
conda activate nfl_smplx
pip install nfl_data_py          # one-time
python scripts/fetch_roster.py --season 2024
# → data/2024/_rosters/rosters.parquet  (+ participation.parquet if available)
```

Set `identity.season=2024` (and `identity.source=roster`) in your config. Per-play
home/away abbreviations and other metadata are read from each play's `meta.yaml`
(see §5). If participation data is missing for a play, the full per-game roster is
used as the candidate set; with no roster at all, `identity.source=ocr_only` falls
back to OCR + jersey-color identities (coarser, no cross-game guarantees).

`data/{season}/_rosters/` is gitignored — nflverse data is not ours to redistribute.

### Avatar/shape library

The library at `data/{season}/_library/{player_uid}/` caches each player's canonical
avatar (`avatar.npz`) + frozen shape (`betas.npz`) once. On later appearances the
avatar stage loads instead of re-running LHM++, and the pose stage reuses the
frozen `betas` (`pose.refit.use_library_betas: true`) so the cached avatar's rig
and the per-play pose skeleton share bone lengths. Generic assets live under
reserved uids: `__referee__` (a striped-shirt avatar for officials) and
`__football__` (the canonical football, oriented along the Kalman velocity).

Force a rebuild with `avatars.library.rebuild=true`. `data/{season}/_library/` is
gitignored (derived data). Author the one-time generic referee avatar before
processing plays, or referee tracks raise a `SetupError`.

## §10 — Running the full season on a GPU cluster (PACE)

End-to-end season pipeline, staged so each player's avatar is built **once** and
reused across every play/game (the library on shared scratch is the cache).

### One-time setup
```bash
# Put data + library + conda envs on scratch (home quota is small); symlink in.
ln -s ~/scratch/nflgsplat/data    data
ln -s ~/scratch/nflgsplat/library library

module load anaconda3
bash scripts/00_setup_environments.sh        # 4 conda envs
bash scripts/01_download_models.sh            # SMPLer-X, LHM, 3dgs-avatar, weights
python scripts/fetch_roster.py --season 2024  # roster/participation prior (§9)
python scripts/build_assets.py --season 2024  # generic referee + football into library
```
Set `slurm.account`, `slurm.partition`, and `slurm.gpu` in `configs/season.yaml`
to your PACE allocation. Plays are discovered by walking the `data/{season}/` tree —
no `games:` list is needed.

### Submit the staged DAG
```bash
python scripts/run_season.py --config configs/season.yaml --dry-run   # inspect
python scripts/run_season.py --config configs/season.yaml --submit    # go
```
Stages: **S1** per-play perception (field recon folded in: tracks → identity →
SMPLest-X → triangulate → fuse → smooth → FK → ball; plays discovered by walking
the tree) → **tail** (`collect_uids` → submit S2) → **S2** avatar build (one task
per unique `player_uid`; heroes via 3DGS-Avatar, others via LHM++) → **S3** render
array (per play). The one-task-per-uid design in S2 makes concurrent library
writes race-free. Calibration (`scripts/02_calibrate_cameras.py --play-dir <play folder>`)
is a manual per-play pre-step done before submitting the DAG — it is not a SLURM stage.

### Single play (debug)
```bash
bash scripts/04_process_play.sh --play-dir data/2024/week_01/NO_at_ATL/play_001          # all the way to render.mp4
bash scripts/04_process_play.sh --play-dir data/2024/week_01/NO_at_ATL/play_001 --perception-only
```

### Stage CLIs
Each step of `04_process_play.sh` is a real `python -m nfl_gsplat.<stage>
--play-dir <path to play folder> [--config ...]` entry point that loads the
calibrated cameras (`cameras.json` inside the play folder via
`calibration.cameras_io.load_cameras`) and reads play metadata from `meta.yaml`
before running the stage:

| Step | Module | Reads → writes |
|---|---|---|
| detect + track | `tracking.detect_track` | video → `tracks.parquet` |
| cross-cam re-ID | `tracking.cross_cam_reid` | enriches `tracks.parquet` (`global_player_id`) |
| jersey OCR | `tracking.jersey_ocr` | enriches `tracks.parquet` (`jersey_number_ocr`) |
| identity | `identity.assign_stage` | `tracks.parquet` + roster → `entities.json` |
| pose | `pose.run_pose` | SMPLest-X → triangulate → fuse → FK → `poses/{id}.npz` |
| ball | `ball.run_ball` | football YOLO → 3D Kalman → `ball.npz` |
| avatars | `avatars.build_play` | `entities.json` → library avatars (single-play path) |

The numerical cores (camera loading, FK fit-forward, triangulation/fuse/smooth,
ball assembly, identity classification, avatar loop) are CPU-unit-tested
(`tests/test_stage_clis.py`). The remaining seams are the three GPU model
adapters (SMPLest-X / LHM++ / 3DGS-Avatar) and the per-frame video crop
extraction, which run inside their conda envs and are verified on PACE — the
first single-play run is where those are exercised end-to-end against real
weights + data.
