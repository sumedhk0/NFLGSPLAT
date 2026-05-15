  Here's the updated runbook with these three layouts spelled out and integrated.

  ---
  Part 0 — Decide/obtain before you touch any machine

  1. Hardware target. Minimum: 1× NVIDIA GPU with ≥ 16 GB VRAM (LHM-1B). 24 GB cards work; 12 GB forces LHM-MINI. Below 8 GB is a hard floor (LHMVRAMError).
  2. Disk budget. ~25 GB conda envs + ~15 GB weights + your raw video + outputs ≈ 60–80 GB.
  3. Two synchronized broadcast clips per game. One sideline, one endzone, same start frame, same FPS. Confirm with ffprobe. If you only have per-play clips, concatenate them into one
  continuous game clip per camera (see Part 1.6).
  4. SMPL-X + SMPL license acceptance.
    - Register at https://smpl-x.is.tue.mpg.de/, download SMPL-X v1.1 (NPZ+PKL).
    - Register at https://smpl.is.tue.mpg.de/, download SMPL v1.1.0.
  5. Per-play time windows. Know the start/end frame of each play in the game video. Use ffmpeg/your video editor to find these.
  6. Game ID slug. A short identifier like game_001 — used as a directory name.

  ---
  Part 1 — Shared setup

  1.1 Clone + Conda

  git clone <repo-url> nfl-gsplat
  cd nfl-gsplat
  # Install Miniforge if needed (see SETUP.md §1)
  bash scripts/00_setup_environments.sh          # ~30–60 min, builds 4 envs

  1.2 Download non-gated models

  bash scripts/01_download_models.sh             # ~2–6 h; LHM OSS is the slow part

  1.3 Place SMPL-X / SMPL body models

  Where: data/body_models/

  data/body_models/
  ├── smplx/
  │   ├── SMPLX_NEUTRAL.npz
  │   ├── SMPLX_MALE.npz
  │   └── SMPLX_FEMALE.npz
  └── smpl/
      ├── SMPL_NEUTRAL.pkl
      ├── SMPL_MALE.pkl
      └── SMPL_FEMALE.pkl

  The SMPL-X archive you downloaded contains a models/smplx/ folder — copy its contents into data/body_models/smplx/. Same for SMPL.

  # Example, after scp'ing the archive in:
  unzip smplx_v1_1.zip -d /tmp/smplx
  cp /tmp/smplx/models/smplx/*.npz data/body_models/smplx/
  ls data/body_models/smplx/                     # confirm 3 .npz files

  1.4 Place broadcast video — per game, NOT per play

  Where: data/raw/{game_id}/

  data/raw/game_001/
  ├── sideline.mp4               # whole-game broadcast, sideline angle
  ├── endzone.mp4                # whole-game broadcast, endzone angle
  └── plays.yaml                 # play_id → frame window (you write this)

  sideline.mp4 and endzone.mp4 are single continuous clips for the whole game (or whatever portion you want to process). The pipeline reads them once for field reconstruction and slices       
  time-windows out for each play.

  1.4a If you only have per-play clips, concatenate them

  mkdir -p data/raw/game_001
  cd /path/to/your/per_play_clips/

  # One file per play, sorted in chronological order
  for c in sideline endzone; do
      for p in play_001 play_002 play_003; do
          echo "file '$PWD/${p}_${c}.mp4'"
      done > /tmp/concat_${c}.txt
      ffmpeg -f concat -safe 0 -i /tmp/concat_${c}.txt -c copy \
          /path/to/repo/data/raw/game_001/${c}.mp4
  done

  After concatenation, ffprobe each output and note the frame counts of each segment — you'll use them to populate plays.yaml next.

  1.4b Write plays.yaml (per-play frame windows)

  # data/raw/game_001/plays.yaml
  # Map each play_id to its (start_frame, end_frame) inside sideline.mp4 / endzone.mp4.
  # Frames are 0-indexed. The pipeline uses these windows for tracking, pose, ball.

  play_001:
    start_frame: 0
    end_frame: 449          # ~15 s @ 30 fps
  play_002:
    start_frame: 450
    end_frame: 899
  play_003:
    start_frame: 900
    end_frame: 1349

  If your videos are clean 30 fps, frame numbers = seconds × 30. Use ffprobe -show_packets data/raw/game_001/sideline.mp4 | head to verify FPS.

  1.5 Write the play list (which plays to run)

  Where: configs/plays/{game_id}.txt

  mkdir -p configs/plays
  cat > configs/plays/game_001.txt <<'EOF'
  play_001
  play_002
  play_003
  EOF

  One play_id per line. Each must have a matching key in plays.yaml. The SLURM array job and full_game.sbatch read this file to know how many tasks to spawn (#SBATCH --array=1-N is set        
  automatically from wc -l).

  1.6 Smoke-test the install (CPU, no GPU needed)

  conda activate nfl_gsplat
  pip install -e ".[test]"
  pytest -m "not gpu and not slow and not real_video" -v
  Expect 60 passing in ~10 s. If this fails, stop — nothing downstream will work.

  ---
  Part 2 — Calibration (one-time per game, requires a GUI)

  # On the box where you can run X (your laptop with X-forwarding, or a desktop session)
  conda activate nfl_gsplat
  python scripts/02_calibrate_cameras.py --game game_001

  For each camera, click NFL field landmarks (yard lines × sidelines/hashes, pylons, goalpost bases). Press s to save, q to quit. Targets < 1 px reprojection RMS; rejects above 5 px.

  If your camera pans/zooms during the game, add keyframes:
  python scripts/02_calibrate_cameras.py --game game_001 --camera sideline --keyframe-frame 4200

  Annotations land at data/annotations/game_001/{cam}_landmarks.json; calibration result at outputs/game_001/calib/cameras.json. After this step, the rest is non-interactive.

  If you're on a headless cluster, calibrate on your laptop and scp -r data/annotations cluster:nfl-gsplat/data/.

  ---
  Part 3a — Run on a single GPU box

  # Field reconstruction: once per game (~25 min on H100, longer on smaller cards)
  bash scripts/03_reconstruct_field.sh game_001

  # Per play: tracking → pose → avatars → ball → composite (~40 min/play on H100)
  bash scripts/04_process_play.sh game_001 play_001
  bash scripts/04_process_play.sh game_001 play_002

  # Render the novel-view MP4
  python scripts/05_render_novel_view.py \
      --game game_001 --play play_001 \
      --trajectory configs/trajectories/fly_through.yaml
  # Result: outputs/game_001/play_001/render.mp4

  Each stage is idempotent — rerunning skips completed work unless you pass --force.

  ---
  Part 3b — Run on a SLURM cluster

  3b.1 Cluster-specific knobs (configs/pipeline.yaml)

  slurm:
    account: your_pace_account
    partition: gpu-h100      # adjust to your cluster
    gpu: h100:1
    time_field: "02:00:00"
    time_play: "01:00:00"
    mem: "64G"
  Also edit #SBATCH --partition= in scripts/slurm/*.sbatch to match your cluster.

  3b.2 Submit the whole game in one shot

  sbatch scripts/slurm/full_game.sbatch game_001
  squeue -u $USER
  This chains: field reconstruction (1 GPU, ~2 h) → per-play array (1 GPU × N plays, 1 h each, depending on the field job via afterok:). N is read from configs/plays/game_001.txt.

  3b.3 Render each play

  conda activate nfl_gsplat
  python scripts/05_render_novel_view.py --game game_001 --play play_001 \
      --trajectory configs/trajectories/fly_through.yaml
  Or wrap in its own sbatch if many plays.

  3b.4 Cluster gotchas

  - Quota. Move repo to scratch (/storage/scratch/$USER) and symlink. data/body_models/, conda envs, and outputs/ blow past 10 GB home quotas easily.
  - Conda on login nodes. Some clusters forbid heavy installs there. Use salloc --gres=gpu:1 --time=2:00:00 for env builds.
  - Module system. If your cluster needs module load cuda/12.1, add it at the top of each sbatch file.
  - OSS firewalled. If LHM downloads fail, run 01_download_models.sh on your laptop and scp -r data/weights/lhm cluster:.../data/weights/.

  ---
  Part 4 — Fetching results

  ls -lh outputs/game_001/play_001/render.mp4
  ffprobe outputs/game_001/play_001/render.mp4

  # Sanity checks
  jq '.sideline.rms_px, .endzone.rms_px' outputs/game_001/calib/cameras.json
  # Expect < 5 px, ideally < 2 px

  python -c "from nfl_gsplat.field.train_field import read_ply_gaussian_count; \
      print(read_ply_gaussian_count('outputs/game_001/field/field.ply'))"
  # Expect > 50_000

  # Pull off cluster
  scp cluster:nfl-gsplat/outputs/game_001/play_001/render.mp4 .

  ---
  Recap: where each user-supplied file goes

  ┌─────────────────────────┬─────────────────────────────────────────────────┬─────────────────────────────────────┬────────────────────────────────────────────┐
  │    What you provide     │                      Path                       │               One per               │                   Format                   │
  ├─────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────┤
  │ SMPL-X body models      │ data/body_models/smplx/SMPLX_*.npz              │ once total                          │ .npz from MPI                              │
  ├─────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────┤
  │ SMPL body models        │ data/body_models/smpl/SMPL_*.pkl                │ once total                          │ .pkl from MPI                              │
  ├─────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────┤
  │ Sideline broadcast      │ data/raw/{game_id}/sideline.mp4                 │ per game                            │ continuous MP4                             │
  ├─────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────┤
  │ Endzone broadcast       │ data/raw/{game_id}/endzone.mp4                  │ per game                            │ continuous MP4                             │
  ├─────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────┤
  │ Per-play frame windows  │ data/raw/{game_id}/plays.yaml                   │ per game                            │ YAML, play_id: {start_frame, end_frame}    │
  ├─────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────┤
  │ Calibration annotations │ data/annotations/{game_id}/{cam}_landmarks.json │ per game (per camera, per keyframe) │ JSON, generated by 02_calibrate_cameras.py │
  ├─────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────────────────────────┤
  │ Plays to run            │ configs/plays/{game_id}.txt                     │ per game                            │ one play_id per line                       │
  └─────────────────────────┴─────────────────────────────────────────────────┴─────────────────────────────────────┴────────────────────────────────────────────┘

  Everything else (outputs/, weights/, third_party/, logs/) is generated by the pipeline and gitignored.