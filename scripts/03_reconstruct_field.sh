#!/usr/bin/env bash
# Static-field reconstruction for one play folder.
#
# Steps:
#   1. Activate nfl_gsplat conda env.
#   2. Extract pre-snap empty-field frames from sideline + endzone videos.
#   3. Build nerfstudio transforms.json from calibrated poses.
#   4. Train splatfacto → export field.ply.
#
# Usage:  bash scripts/03_reconstruct_field.sh <play-dir>
#   e.g.  bash scripts/03_reconstruct_field.sh data/2024/week_01/NO_at_ATL/play_001

set -euo pipefail

PLAY_DIR="${1:-}"
if [[ -z "$PLAY_DIR" ]]; then
    echo "usage: $0 <play-dir>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nfl_gsplat

cd "$REPO_ROOT"

python -m nfl_gsplat.field.extract_static_frames \
    --play-dir "$PLAY_DIR" \
    --config configs/pipeline.yaml --config-override configs/field_recon.yaml

python -m nfl_gsplat.field.build_transforms \
    --play-dir "$PLAY_DIR" \
    --config configs/pipeline.yaml

python -m nfl_gsplat.field.train_field \
    --play-dir "$PLAY_DIR" \
    --config configs/field_recon.yaml

echo "field.ply → $PLAY_DIR/field.ply"
