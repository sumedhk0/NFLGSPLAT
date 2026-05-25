#!/usr/bin/env bash
# Process a single play, activating the right conda env per stage so CUDA +
# torch versions stay compatible. Every stage reads/writes outputs/{game}/{play}/
# via the shared config + path layer (nfl_gsplat.config / nfl_gsplat.paths).
#
# With --perception-only (used by the season perception array, S2), it stops
# after the ball track: avatars are built once per player in the avatar-build
# array (S3) and rendering runs in the render array (S4). Without it, the play
# is taken all the way to render.mp4 (single-play / debug use).
#
# Usage:  bash scripts/04_process_play.sh game_001 play_001 [--perception-only]

set -euo pipefail

GAME="${1:-}"; PLAY="${2:-}"; MODE="${3:-}"
if [[ -z "$GAME" || -z "$PLAY" ]]; then
    echo "usage: $0 <game_id> <play_id> [--perception-only]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$REPO_ROOT"

CFG="--config configs/pipeline.yaml"

echo "=== [1/6] tracking + cross-cam re-ID + jersey OCR  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.tracking.detect_track   --game "$GAME" --play "$PLAY" $CFG
python -m nfl_gsplat.tracking.cross_cam_reid --game "$GAME" --play "$PLAY" $CFG
python -m nfl_gsplat.tracking.jersey_ocr     --game "$GAME" --play "$PLAY" $CFG || true
conda deactivate

echo "=== [2/6] identity assignment → entities.json  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.identity.assign_stage   --game "$GAME" --play "$PLAY" $CFG
conda deactivate

echo "=== [3/6] SMPLest-X → triangulate → fuse → smooth → FK (poses/{uid}.npz)  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.pose.run_pose           --game "$GAME" --play "$PLAY" $CFG
conda deactivate

echo "=== [4/6] ball detect + 3D Kalman → ball.npz  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.ball.run_ball           --game "$GAME" --play "$PLAY" $CFG
conda deactivate

if [[ "$MODE" == "--perception-only" ]]; then
    echo "perception complete: outputs/$GAME/$PLAY/ (entities.json, poses/, ball.npz)"
    exit 0
fi

echo "=== [5/6] avatars for this play's players  (env: nfl_lhm) ==="
conda activate nfl_lhm
python -m nfl_gsplat.avatars.build_play      --game "$GAME" --play "$PLAY" $CFG
conda deactivate

echo "=== [6/6] composite + novel-view render  (env: nfl_gsplat) ==="
conda activate nfl_gsplat
python scripts/05_render_novel_view.py --game "$GAME" --play "$PLAY" \
    --trajectory configs/trajectories/fly_through.yaml
conda deactivate

echo "play render: outputs/$GAME/$PLAY/render.mp4"
