#!/usr/bin/env bash
# Process a single play, activating the right conda env per stage so CUDA +
# torch versions stay compatible. Every stage reads/writes the self-contained
# play folder (data/{season}/week_NN/{matchup}/play_NNN/) via nfl_gsplat.paths.
#
# With --perception-only (used by the season perception array, S2), it stops
# after the ball track: avatars are built once per player in the avatar-build
# array (S3) and rendering runs in the render array (S4). Without it, the play
# is taken all the way to render.mp4 (single-play / debug use).
#
# Stage ordering rationale:
#   1. detect_track   → writes tracks.parquet (player masks needed by 02b)
#   2. 02b calibration→ writes cameras.npz (needs masks from step 1;
#                        read by field build and cross-cam re-ID in steps 3-4)
#   3. field recon    → reads cameras.npz for per-frame transforms
#   4. cross_cam_reid + jersey_ocr → read cameras.npz for 3D back-projection
#   5-9. identity → pose → ball → avatars → render
#
# Usage:  bash scripts/04_process_play.sh <play-dir> [--perception-only]
#   e.g.  bash scripts/04_process_play.sh data/2024/week_01/NO_at_ATL/play_001

set -euo pipefail

PLAY_DIR="${1:-}"; MODE="${2:-}"
if [[ -z "$PLAY_DIR" ]]; then
    echo "usage: $0 <play-dir> [--perception-only]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$REPO_ROOT"

CFG="--config configs/pipeline.yaml"

echo "=== [1/9] player detect + track → tracks.parquet  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.tracking.detect_track   --play-dir "$PLAY_DIR" $CFG
conda deactivate

# Runs after detect_track (player masks) but before field/re-ID (which read cameras.npz).
echo "=== [2/9] per-frame camera calibration → cameras.npz  (env: nfl_smplx) ==="
conda activate nfl_smplx
python scripts/02b_track_calibration.py --play-dir "$PLAY_DIR" $CFG
conda deactivate

echo "=== [3/9] static-field reconstruction → field.ply  (env: nfl_gsplat) ==="
conda activate nfl_gsplat
python -m nfl_gsplat.field.extract_static_frames \
    --play-dir "$PLAY_DIR" $CFG --config-override configs/field_recon.yaml
python -m nfl_gsplat.field.build_transforms \
    --play-dir "$PLAY_DIR" $CFG
python -m nfl_gsplat.field.train_field \
    --play-dir "$PLAY_DIR" --config configs/field_recon.yaml
conda deactivate

echo "=== [4/9] cross-cam re-ID + jersey OCR  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.tracking.cross_cam_reid --play-dir "$PLAY_DIR" $CFG
python -m nfl_gsplat.tracking.jersey_ocr     --play-dir "$PLAY_DIR" $CFG || true
conda deactivate

echo "=== [5/9] identity assignment → entities.json  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.identity.assign_stage   --play-dir "$PLAY_DIR" $CFG
conda deactivate

echo "=== [6/9] SMPLest-X → triangulate → fuse → smooth → FK (poses/{uid}.npz)  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.pose.run_pose           --play-dir "$PLAY_DIR" $CFG
conda deactivate

echo "=== [7/9] ball detect + 3D Kalman → ball.npz  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.ball.run_ball           --play-dir "$PLAY_DIR" $CFG
conda deactivate

if [[ "$MODE" == "--perception-only" ]]; then
    echo "perception complete: $PLAY_DIR (entities.json, poses/, ball.npz)"
    exit 0
fi

echo "=== [8/9] avatars for this play's players  (env: nfl_lhm) ==="
conda activate nfl_lhm
python -m nfl_gsplat.avatars.build_play      --play-dir "$PLAY_DIR" $CFG
conda deactivate

echo "=== [9/9] composite + novel-view render  (env: nfl_gsplat) ==="
conda activate nfl_gsplat
python scripts/05_render_novel_view.py --play-dir "$PLAY_DIR" \
    --trajectory configs/trajectories/fly_through.yaml
conda deactivate

echo "play render: $PLAY_DIR/render.mp4"
