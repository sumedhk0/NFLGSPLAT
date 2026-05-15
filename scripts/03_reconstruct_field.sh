#!/usr/bin/env bash
# One-time static-field reconstruction for a game.
#
# Steps:
#   1. Activate nfl_gsplat conda env.
#   2. Extract pre-snap empty-field frames from sideline + endzone videos.
#   3. Build nerfstudio transforms.json from calibrated poses.
#   4. Train splatfacto → export field.ply.
#
# Usage:  bash scripts/03_reconstruct_field.sh game_001

set -euo pipefail

GAME="${1:-}"
if [[ -z "$GAME" ]]; then
    echo "usage: $0 <game_id>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nfl_gsplat

cd "$REPO_ROOT"

python -m nfl_gsplat.field.extract_static_frames \
    --game "$GAME" \
    --config configs/pipeline.yaml --config-override configs/field_recon.yaml

python -m nfl_gsplat.field.build_transforms \
    --game "$GAME" \
    --config configs/pipeline.yaml

python -m nfl_gsplat.field.train_field \
    --game "$GAME" \
    --config configs/field_recon.yaml

echo "field.ply → outputs/$GAME/field/field.ply"
