  Here's the runbook for the per-play season-tree layout.

  ---
  Part 0 — Decide/obtain before you touch any machine

  1. Hardware target. Minimum: 1× NVIDIA GPU with ≥ 16 GB VRAM (LHM-1B). 24 GB cards work; 12 GB forces LHM-MINI. Below 8 GB is a hard floor (LHMVRAMError).
  2. Disk budget. ~25 GB conda envs + ~15 GB weights + your clips + outputs ≈ 60–80 GB.
  3. Two synchronized broadcast clips per play. One sideline, one endzone, the same play, same start frame, same FPS — already trimmed to the snap. Confirm with ffprobe. Each play is its own pair of clips; there are no whole-game videos and no frame windows.
  4. SMPL-X license acceptance. Register at https://smpl-x.is.tue.mpg.de/, download SMPL-X v1.1 (NPZ+PKL). (SMPL is only needed if a third_party model demands it; our code reads SMPL-X.)
  5. Season / week / matchup. Know each play's season, week number, and the home/away team abbreviations (NFL standard is "away at home", e.g. NO at ATL).

  ---
  Part 1 — Shared setup

  1.1 Clone + Conda

  git clone <repo-url> nfl-gsplat
  cd nfl-gsplat
  # Install Miniforge if needed (see SETUP.md §1)
  bash scripts/00_setup_environments.sh          # ~30–60 min, builds 4 envs

  1.2 Download non-gated models

  bash scripts/01_download_models.sh             # ~2–6 h; LHM OSS is the slow part

  1.3 Place SMPL-X body models

  Where: data/body_models/

  data/body_models/
  └── smplx/
      ├── SMPLX_NEUTRAL.npz
      ├── SMPLX_MALE.npz
      └── SMPLX_FEMALE.npz

  The SMPL-X archive you downloaded contains a models/smplx/ folder — copy its contents into data/body_models/smplx/.

  # Example, after scp'ing the archive in:
  unzip smplx_v1_1.zip -d /tmp/smplx
  cp /tmp/smplx/models/smplx/*.npz data/body_models/smplx/
  ls data/body_models/smplx/                     # confirm 3 .npz files

  1.4 Lay out a play — the per-play tree

  Each play is a self-contained folder:
  data/{season}/week_NN/{away}_at_{home}/play_NNN/

  Scaffold the folder + a meta.yaml stub, then drop the two clips in:

  python scripts/new_play.py --season 2024 --week 1 --away NO --home ATL --play play_001 --fps 30
  # creates data/2024/week_01/NO_at_ATL/play_001/ + meta.yaml
  # then copy/scp the two clips into it:
  #   data/2024/week_01/NO_at_ATL/play_001/sideline.mp4
  #   data/2024/week_01/NO_at_ATL/play_001/endzone.mp4

  The resulting layout (everything for one play lives together):

  data/2024/
  ├── _library/        # season-shared avatar/shape cache (cross-play, auto-built)
  ├── _rosters/        # nflverse roster data for the season
  ├── _registry.json   # season identity registry
  └── week_01/
      └── NO_at_ATL/
          └── play_001/
              ├── sideline.mp4  endzone.mp4   # the two clips ARE the play
              ├── meta.yaml                   # season/week/teams/fps[/gsis_play_id]
              ├── cameras.json  field.ply     # produced by calibration + field recon
              ├── tracks.parquet  entities.json  smplestx/  poses/  ball.npz
              └── render.mp4                  # final output

  meta.yaml (written by new_play.py; edit as needed):

  season: 2024
  week: 1
  home_team: ATL
  away_team: "NO"        # quote abbreviations — bare NO/ON/NA parse as YAML booleans
  fps: 30.0
  gsis_play_id: 36       # optional; only for nflverse participation alignment

  There are no frame windows and no configs/plays list — each clip is already the play, and the season runner discovers plays by walking the tree.

  1.5 Smoke-test the install (CPU, no GPU needed)

  conda activate nfl_smplx
  pip install -e ".[test]"
  pytest -m "not gpu and not slow and not real_video" -q
  Expect 158 passing in ~10 s. If this fails, stop — nothing downstream will work.

  ---
  Part 2 — Calibration (automatic-with-hint, per-frame; per play)

  Calibration is automatic and per-frame: the pipeline detects yard lines in each frame and solves the camera using a one-time hint from meta.yaml that identifies a single line. No display, no annotation, no keyframes.

  Step 1 — find a reference yard-line x-position (headless, any node)

  conda activate nfl_gsplat
  python scripts/diag_calib.py --play-dir data/2024/week_01/NO_at_ATL/play_001 \
      --frame 0 --cam sideline --out-dir ~/scratch/diag

  Open the saved PNG (diag_sideline_f00000.png) to see which line is which yard. The script prints detected line x-positions — pick the x of a yard line you can identify by eye.

  Step 2 — add calib_hints to meta.yaml

  Edit data/2024/week_01/NO_at_ATL/play_001/meta.yaml and add:

  calib_hints:
    sideline: {ref_frame: 0, ref_x: 866, yard: 30, side: away, increasing: right}
    endzone:  {ref_frame: 0, ref_x: 540, yard: 35, side: home, increasing: left}

  ref_x = the image-x of a yard line you can identify (from the diagnostic above).
  yard / side = that line's absolute yard number and which end of the field (home/away).
  increasing = which direction yard numbers grow in the image (left or right).
  Repeat for each camera. A missing hint raises a SetupError naming the camera.

  Note: number-OCR was replaced by the hint because painted numbers are not reliably
  OCR-able on this footage. The YOLO player-mask wiring for line de-cluttering is
  finalized at bring-up.

  Step 3 — run automatic calibration (headless, any node)

  conda activate nfl_gsplat
  python scripts/02_autocalibrate.py --play-dir data/2024/week_01/NO_at_ATL/play_001

  Detects yard lines each frame and solves the camera per frame, writing cameras.npz.
  Fails loudly (naming a frame range) if a long run of consecutive frames cannot be
  registered. If that happens: check that side / increasing match the actual camera view
  and re-run.

  This step runs automatically as [2/9] inside scripts/04_process_play.sh (after player
  detect+track, before field reconstruction). meta.yaml must have a valid calib_hints
  block before you submit the DAG.

  Manual fallback (if automatic registration fails loud on a clip)

  The original interactive two-step path is kept as a fallback:

  Step 1 — annotate keyframe anchors (needs a display: PACE OnDemand Interactive Desktop, laptop, or X-forwarding)

  conda activate nfl_gsplat
  python scripts/02_calibrate_cameras.py \
      --play-dir data/2024/week_01/NO_at_ATL/play_001 \
      --keyframe 0 --keyframe <mid> --keyframe <last>

  For each keyframe and each camera, click NFL field landmarks (yard-line intersections, sideline/hash marks, pylons, goalpost bases). Press s to save, q to quit. Targets < 1 px reprojection RMS; rejects above 5 px. Annotations land in {cam}_keyframes.json inside the play folder.

  Step 2 — batch homography tracking (headless; any node)

  conda activate nfl_gsplat
  python scripts/02b_track_calibration.py --play-dir data/2024/week_01/NO_at_ATL/play_001

  Reads {cam}_keyframes.json and tracks the field homography across every frame, writing cameras.npz. Fails loudly if keyframe JSON files are missing — run step 1 first. If 02b reports it cannot cover a frame range, add a keyframe anchor in that range and re-run step 2.

  ---
  Part 3a — Run a single play on one GPU box

  # Full per-play pipeline: field → tracking → identity → pose → ball → avatars → render
  bash scripts/04_process_play.sh data/2024/week_01/NO_at_ATL/play_001

  # Or render only (after a processed play):
  python scripts/05_render_novel_view.py \
      --play-dir data/2024/week_01/NO_at_ATL/play_001 \
      --trajectory configs/trajectories/fly_through.yaml
  # Result: data/2024/week_01/NO_at_ATL/play_001/render.mp4

  Each stage is idempotent — rerunning skips completed work. The avatar/shape library at data/2024/_library is shared across all plays of the season, so a recurring player is reconstructed once and reused.

  ---
  Part 3b — Run a season on a SLURM cluster

  3b.1 Cluster-specific knobs (configs/season.yaml)

  season: 2024
  paths:
    data_root: data
  slurm:
    account: your_pace_account
    partition: gpu-h100      # adjust to your cluster
    gpu: "h100:1"
    qos: embers              # PACE free/preemptible backfill (vs paid inferno)
    requeue: true
    time_perception: "01:00:00"
    time_avatar: "04:00:00"
    time_render: "01:00:00"

  3b.2 Calibrate each play first (manual pre-step; not a SLURM stage), then submit the staged DAG

  python scripts/run_season.py --config configs/season.yaml --dry-run   # preview
  python scripts/run_season.py --config configs/season.yaml --submit

  run_season.py walks data/{season}/week_*/*_at_*/play_* (each must have both clips + meta.yaml), writes outputs/play_worklist.txt, and submits: per-play perception (field folded in) → collect_uids → per-uid avatar build → per-play render, wired with --dependency=afterok and --array.

  3b.3 Cluster gotchas

  - Quota. Move repo to scratch (~/scratch) and symlink. data/ (clips + _library), conda envs, and outputs/ blow past small home quotas.
  - Conda on login nodes. Some clusters forbid heavy installs there. Use an interactive GPU alloc for env builds.
  - Module system. module load anaconda3 (add to ~/.bashrc). If CUDA is needed, module load it at the top of each sbatch.
  - OSS firewalled. If LHM downloads fail, run 01_download_models.sh on your laptop and scp third_party/LHM/pretrained_models up.

  ---
  Part 4 — Fetching results

  PLAY=data/2024/week_01/NO_at_ATL/play_001
  ls -lh "$PLAY/render.mp4"
  ffprobe "$PLAY/render.mp4"

  # Sanity checks
  jq '.reprojection_error_px' "$PLAY/cameras.json"     # expect < 5 px, ideally < 2 px

  python -c "from nfl_gsplat.field.train_field import read_ply_gaussian_count; \
      print(read_ply_gaussian_count('$PLAY/field.ply'))"   # expect > 50_000

  # Pull off cluster
  scp cluster:nfl-gsplat/"$PLAY"/render.mp4 .

  ---
  Recap: where each user-supplied file goes

  ┌─────────────────────────┬───────────────────────────────────────────────────────────────┬──────────┬─────────────────────────────────────┐
  │    What you provide     │                             Path                              │ One per  │                Format               │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────┤
  │ SMPL-X body models      │ data/body_models/smplx/SMPLX_*.npz                            │ once     │ .npz from MPI                       │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────┤
  │ Sideline clip           │ data/{season}/week_NN/{away}_at_{home}/play_NNN/sideline.mp4   │ per play │ pre-trimmed MP4                     │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────┤
  │ Endzone clip            │ data/{season}/week_NN/{away}_at_{home}/play_NNN/endzone.mp4    │ per play │ pre-trimmed MP4                     │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────┤
  │ Play metadata           │ data/{season}/week_NN/{away}_at_{home}/play_NNN/meta.yaml      │ per play │ YAML (season/week/teams/fps[/gsis]) │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────┤
  │ Calibration (cameras)   │ data/{season}/week_NN/{away}_at_{home}/play_NNN/cameras.json   │ per play │ JSON, from 02_calibrate_cameras.py  │
  └─────────────────────────┴───────────────────────────────────────────────────────────────┴──────────┴─────────────────────────────────────┘

  Use scripts/new_play.py to scaffold each play folder + meta.yaml, then drop the two clips in. Everything else (cameras.json, field.ply, tracks/entities/poses/ball, render.mp4, _library/, _rosters/, weights, third_party/, logs/) is generated by the pipeline and gitignored.
