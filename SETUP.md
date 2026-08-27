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

## §3. Camera calibration (automatic-with-hint, per-frame)

Calibration is **automatic and per-frame**: the pipeline detects yard lines in each frame and solves the camera using a one-time hint from `meta.yaml` that identifies a single line. No display, no annotation, no keyframes.

### Step 1 — Find a reference yard-line x-position

```bash
python scripts/diag_calib.py --play-dir data/2024/week_01/NO_at_ATL/play_001 \
    --frame 0 --cam sideline --out-dir ~/scratch/diag
```

Open the saved PNG (`diag_sideline_f00000.png`) to see which line is which yard. The script also prints the detected line x-positions — pick the x of a yard line you can identify by eye (e.g. a painted number or a distinctive gap between lines).

### Step 2 — Add `calib_hints` to `meta.yaml`

```yaml
calib_hints:
  sideline: {ref_frame: 0, ref_x: 866, yard: 30, side: away, increasing: right}
  endzone:  {ref_frame: 0, ref_x: 540, yard: 35, side: home, increasing: left}
```

- `ref_x` — image-x of the yard line you identified (from the diagnostic above).
- `yard` / `side` — the absolute yard number for that line and which side of the field (home/away end zone).
- `increasing` — which direction yard numbers grow in the image (`left` or `right`).

Repeat for each camera. A missing hint causes a `SetupError` naming the camera.

### Step 3 — Run automatic calibration

```bash
python scripts/02_autocalibrate.py --play-dir data/2024/week_01/NO_at_ATL/play_001
```

Writes `<play-dir>/cameras.npz`. **Fails loudly** if a long run of consecutive frames cannot be registered — this usually means the hint is wrong. Flip `side` or `increasing` and re-run.

`02_autocalibrate.py` runs automatically as **step [2/9]** in `scripts/04_process_play.sh` (after player detect+track, before field reconstruction), so `meta.yaml` must have a valid `calib_hints` block before you submit the DAG.

> **Note:** number-OCR was replaced by the hint because painted numbers are not reliably OCR-able on this footage. The keyframe+tracking path (`02_calibrate_cameras.py` + `02b_track_calibration.py`) remains as a fallback. The YOLO player-mask wiring for line de-cluttering is finalized at bring-up.

### Endzone camera — the static-mosaic route (`--mode mosaic-endzone`)

**Use this for the endzone camera.** The earlier routes are superseded: field
markings alone mislabel yard lines by 67-210 px from that angle, per-frame
geometric cross-camera matching mispairs ~85% of feet, and the jersey-identity
route needs >=4 shared player IDs in a single frame, which real footage did not
supply (measured: 0 frames out of 327).

The mosaic route works because a broadcast endzone camera sits on a **tripod**:
it pans and zooms but never translates, so every frame is related to every other
by a pure homography. Register them all into one reference frame, accumulate the
white paint with players masked out (paint reinforces, movers wash out), then
solve one camera against the field model and propagate to every frame.

**Prerequisites**

* the **sideline** camera already solved in `cameras.npz`
* `tracks.parquet` with `cam=="endzone"` rows (from `03b_detect_players.py`) —
  without player boxes, white jerseys are indistinguishable from paint
* an `endzone_prior:` block in `meta.yaml`

```yaml
endzone_prior:
  x_range: [60, 200]        # camera centre bounds, METRES. Get the SIGN right:
  y_range: [-20, 20]        #   it is which end zone the camera is behind.
  z_range: [10, 60]         # must exclude negatives -- this is what rejects the
                            #   mirror solution, and nothing else does
  focal_range: [1200, 30000]  # a long telephoto that zooms OUT to follow the
                              #   play; measured span on one play was 1532..23200
```

**First run per game — read two anchors off the mosaic**

Run it once with no `endzone_anchor:`. It writes `<play>_mosaic.png` (detected
lines in red, midpoints in green) and fails loud. Equally spaced parallel lines
are translation-invariant, so which yard line is which cannot be recovered from
geometry — it needs one absolute reference. Identify the **outermost two** lines
and add:

```yaml
endzone_anchor:
  lines:
    - {point_px: [962.5, 2.9],   world_x_m: -45.720}   # goal line
    - {point_px: [956.5, 982.8], world_x_m:  -9.144}   # the 40
```

This is **once per game, not per play** — the tripod holds one centre all half.
The anchors must be the outermost detected lines, so the spacing check spans
every line rather than just the interval between them.

First run for a game (solves the reference camera from hash marks):

```bash
python scripts/02_autocalibrate.py --play-dir <play> --mode mosaic-endzone \
    --stride 6 --ref-frame 648 --propagate-stride 5 --refine-stride 5 \
    --max-gap 120
```

Every later run, once a reference camera has been solved and verified:

```bash
python scripts/02_autocalibrate.py --play-dir <play> --mode mosaic-endzone \
    --stride 6 --ref-frame 648 --reuse-reference \
    --propagate-stride 1 --refine-stride 1 --max-gap 200
```

Stride 1 is the coverage setting and it is worth a great deal: on play_001 it
took the endzone from 104 verified frames to 473, and frames verified in BOTH
cameras from 86 to 364, at the same accuracy (median 0.94 px against 0.96 px).
At stride 5, four frames in five were never given a camera at all -- they were
not rejected, they were never candidates. Chain homographies also improve,
because adjacent frames genuinely are adjacent rather than five apart.

* `--stride` — frames used to BUILD the mosaic. Coarse is fine; 6-12 works.
* `--ref-frame` — the frame everything registers into. It is added to the
  propagation and refinement grids explicitly, so it need not be a multiple of
  any stride.
* `--propagate-stride` — frames in the OUTPUT track. Defaults to `--stride`;
  set `1` for a camera on every frame, which is what compositing needs. It adds
  coverage, **not** accuracy: a tripod's frames all share one centre, so extra
  frames add no parallax.
* `--refine-stride` — turns on the bundle-adjusted refinement pass (below). Use
  a stride that divides whatever stride the splat samples at, so every frame you
  intend to train on is a node.
* `--reuse-reference` — take the reference camera already in `cameras.npz` at
  `--ref-frame` instead of re-solving it. **Use this for every run after the
  first.** The reference solve fits a ~19,000 px focal from hash-mark centroids
  in accumulated paint, which makes it by far the most environment-sensitive
  stage in the pipeline: upgrading OpenCV 4 → 5.0 moved the accumulation by
  about 15 cm, which is 25 px at that focal, and the solve then failed its own
  gates with no code change while every stage after it still worked. Reuse is
  safe because everything downstream is self-checking — a frame that cannot be
  verified against paint it can see keeps `conf = 0`, so a bad reference costs
  coverage, not correctness. It refuses a `--ref-frame` whose `conf` is 0, since
  that pose was interpolated from a neighbour rather than solved.
* `--max-gap` — longest run of uncalibrated frames tolerated. Raise it only
  when you have looked at the span: a fast pan can break the pure-rotation model
  outright (rolling shutter shears the frame mid-slew). play_001 has a real
  94-frame uncovered pan, hence `120` here.

**The refinement pass (`--refine-stride`).** Propagation carries every frame's
camera through ONE homography onto the reference, so error grows with distance
from it. The refinement re-solves every node's focal and rotation jointly — tied
to its neighbours by directly measured consecutive homographies, and to the field
by yard-line anchors — then checks each result against paint that frame can
actually see. The camera centre is shared and held fixed, so a node costs 4
parameters rather than 7.

Both residual families are required. The chain alone is satisfied by a solution
that is rigidly wrong: the whole network can rotate or zoom as one, and every
chain residual stays perfect while the model sits a yard line out. An earlier
sweep did exactly that and reported a flattering 1.59 px.

**Accepting the result.** The run prints what it did; these are play_001's
numbers:

```
endzone mosaic: hash rows mirrored; yard-line reprojection 1.46 px
endzone refine: 214 frames read, 213 chain pairs, 56 anchor nodes
endzone bundle: median |residual| chain 92.606 -> 2.929 px, anchor 40.713 -> 1.758 px
endzone refine: verified 104/214 frames (median 0.96 px, worst 5.66 px)
```

Read them in this order:

1. **yard-line reprojection** — the reference camera against the accumulated
   paint. Want under ~3 px. If this is wrong, nothing downstream can be right.
2. **bundle chain / anchor** — reported separately on purpose: they answer
   different questions (does the network agree with the measured homographies /
   do the anchored nodes sit on the field). One pooled median reads ~0.00 px
   regardless of the fit, because most anchor rows are structurally zero.
3. **verified N/M** — frames whose camera was checked against paint they can
   see. Everything else is left at `conf = 0`.

**Roughly half the frames not verifying is expected, not a failure.** The endzone
camera zooms tight on the line of scrimmage for much of a play, leaving frames
with no painted yard line in view at all. There is no evidence in such a frame to
check a camera against, and a camera that cannot be checked is not shipped.
Overlay the model on a few rejects before assuming the gate is too strict — that
check is what showed the rejects were genuinely paint-poor rather than badly
calibrated.

**`conf = 0` does not mean "no camera".** `CameraTrack.at()` returns a
well-formed pose for every index; unsolved ones are filled from the nearest valid
neighbour. Nothing about the returned pose says "interpolated" — one measured
8 px off the paint — so always filter on `conf > 0` before using a camera for
anything that matters. `build_transforms` and `extract_static_frames` now do this
by default.

**Then extract and build transforms:**

```bash
python -m nfl_gsplat.field.extract_static_frames --play-dir <play>
python -m nfl_gsplat.field.build_transforms      --play-dir <play>
```

`extract_static_frames` defaults to `--verified-only`, spending the per-camera
frame budget on frames that have a trustworthy camera rather than on whatever a
fixed sampling clock lands on. On play_001 the clock kept 44 endzone frames of
which 16 verified, while 104 verified cameras existed. Pass `--all-frames` to
restore clock sampling. `build_transforms` drops any frame still at `conf = 0`
and reports the count.

play_001 end to end: **120 training views (60 sideline + 60 endzone), all
verified**, against 43 before.

### Learned calibration (field-landmark detector)

If you have labelled training data and a trained `LandmarkNet`, the learned path
bypasses the hint/consensus flow entirely — the network outputs named, spread
correspondences directly, so no `calib_hints` block is required in `meta.yaml`.

1. **Label ~100–150 frames/clip** (one pass per distinct camera angle):

   ```bash
   python scripts/label_landmarks.py \
       --video sideline.mp4 --out labels/sideline.json
   ```

2. **Train on embers** (preemptible; resumes from `ckpt_last.pt` on requeue):

   ```bash
   sbatch scripts/train_landmarks.sbatch \
       labels/sideline.json frames_dir/ out/landmarks/ <yard_min> <yard_max>
   ```

3. **Run learned autocalibration:**

   ```bash
   python scripts/02_autocalibrate.py \
       --play-dir data/2024/week_01/NO_at_ATL/play_001 \
       --mode learned \
       --model-ckpt out/landmarks/ckpt_best.pt
   ```

4. **Validate** with the field-overlay diagnostic:

   ```bash
   python scripts/diag_calib.py \
       --play-dir data/2024/week_01/NO_at_ATL/play_001 \
       --frame 0 --cam sideline --out-dir ~/scratch/diag
   ```

### Manual fallback (if automatic registration fails loud on a clip)

The original two-step interactive path is kept as a fallback:

**Step 1 — Annotate keyframe anchors (needs a display: PACE OnDemand Interactive Desktop, laptop, or X-forwarding)**

```bash
python scripts/02_calibrate_cameras.py \
    --play-dir data/2024/week_01/NO_at_ATL/play_001 \
    --keyframe 0 --keyframe <mid> --keyframe <last>
```

Click NFL field landmarks for each keyframe/camera; press `s` to save, `q` to quit. Annotations land in `{cam}_keyframes.json`. Minimum 6 landmarks per keyframe per camera; PnP rejects above 5 px RMS.

**Step 2 — Batch homography tracking (headless)**

```bash
python scripts/02b_track_calibration.py --play-dir data/2024/week_01/NO_at_ATL/play_001
```

Reads `{cam}_keyframes.json` and tracks the field homography frame-by-frame, writing `cameras.npz`. If it reports it cannot cover a frame range, add a keyframe anchor in that range and re-run.

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

## §9b — Quickstart: one field splat on PACE

Before committing to the full season DAG, train ONE play and look at it. This is
the first honest test of whether the reconstruction is worth scaling.

```bash
# on PACE, from the repo root
module load anaconda3
bash scripts/00_setup_environments.sh --only nfl_gsplat   # ~20 min, once
conda activate nfl_gsplat && ns-train --help              # sanity: nerfstudio present
```

The play needs `sideline.mp4`, `endzone.mp4`, `cameras.npz` and `meta.yaml`.
Calibration itself runs fine on a laptop — only the splat needs the GPU — so the
usual flow is calibrate locally, then copy the play directory up.

```bash
sbatch scripts/slurm/field_recon.sbatch data/2025/week_04/SEA_at_AZ/play_001
```

The argument is a play **directory**, not a game id. Output is
`<play>/field.ply`; the job runs on `--qos=embers` (free, preemptible,
auto-requeued) and logs to `logs/nfl-field-<jobid>.out`.

**Set expectations before you look at it.** Both feeds are tripods, so the whole
capture has exactly **two camera centres** (130 m apart). Gaussian Splatting
infers depth from parallax, and two viewpoints is the bare minimum. The field is
planar and should come out clean; anything with real depth — stands, goalposts,
crowd — is under-constrained, and the optimiser will place blobs that look right
from both cameras and wrong from anywhere between them. Players are deliberately
not splatted at all; they are posed avatars composited later.

## §9b — Quickstart: one field splat on PACE

Before committing to the full season DAG, train ONE play and look at it. This is
the first honest test of whether the reconstruction is worth scaling.

```bash
# on PACE, from the repo root
module load anaconda3
bash scripts/00_setup_environments.sh --only nfl_gsplat   # ~20 min, once
conda activate nfl_gsplat && ns-train --help              # sanity: nerfstudio present
```

The play needs `sideline.mp4`, `endzone.mp4`, `cameras.npz` and `meta.yaml`.
Calibration itself runs fine on a laptop — only the splat needs the GPU — so the
usual flow is calibrate locally, then copy the play directory up.

```bash
sbatch scripts/slurm/field_recon.sbatch data/2025/week_04/SEA_at_AZ/play_001
```

The argument is a play **directory**, not a game id. Output is
`<play>/field.ply`; the job runs on `--qos=embers` (free, preemptible,
auto-requeued) and logs to `logs/nfl-field-<jobid>.out`.

**Set expectations before you look at it.** Both feeds are tripods, so the whole
capture has exactly **two camera centres** (130 m apart). Gaussian Splatting
infers depth from parallax, and two viewpoints is the bare minimum. The field is
planar and should come out clean; anything with real depth — stands, goalposts,
crowd — is under-constrained, and the optimiser will place blobs that look right
from both cameras and wrong from anywhere between them. Players are deliberately
not splatted at all; they are posed avatars composited later.

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
