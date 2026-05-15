#!/usr/bin/env bash
# End-to-end processing for a single play.
#
# Activates the right conda env at each stage so CUDA + torch versions stay
# compatible. Every stage writes to outputs/{game}/{play}/ and is idempotent
# (content-hash manifest; rerun with --force to invalidate).
#
# Usage:  bash scripts/04_process_play.sh game_001 play_001

set -euo pipefail

GAME="${1:-}"; PLAY="${2:-}"
if [[ -z "$GAME" || -z "$PLAY" ]]; then
    echo "usage: $0 <game_id> <play_id>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$REPO_ROOT"

echo "=== [1/5] tracking + cross-cam re-ID  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.tracking.detect_track --game "$GAME" --play "$PLAY"
python -m nfl_gsplat.tracking.cross_cam_reid --game "$GAME" --play "$PLAY"
python -m nfl_gsplat.tracking.jersey_ocr --game "$GAME" --play "$PLAY" || true
conda deactivate

echo "=== [2/5] pose fusion  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.pose.smplestx_infer --game "$GAME" --play "$PLAY"
python -m nfl_gsplat.pose.triangulate --game "$GAME" --play "$PLAY"
python -m nfl_gsplat.pose.fuse_smplx --game "$GAME" --play "$PLAY"
python -m nfl_gsplat.pose.temporal_smooth --game "$GAME" --play "$PLAY"
conda deactivate

echo "=== [3/5] avatars  (env: nfl_lhm) ==="
conda activate nfl_lhm
python -m nfl_gsplat.avatars.lhm_wrapper --game "$GAME" --play "$PLAY"
conda deactivate

echo "=== [4/5] ball  (env: nfl_smplx) ==="
conda activate nfl_smplx
python -m nfl_gsplat.ball.detect_ball --game "$GAME" --play "$PLAY"
python -m nfl_gsplat.ball.kalman_3d --game "$GAME" --play "$PLAY"
conda deactivate

echo "=== [5/5] composite  (env: nfl_gsplat) ==="
conda activate nfl_gsplat
python scripts/05_render_novel_view.py --game "$GAME" --play "$PLAY" \
    --trajectory configs/trajectories/fly_through.yaml
conda deactivate

echo "play render: outputs/$GAME/$PLAY/render.mp4"
