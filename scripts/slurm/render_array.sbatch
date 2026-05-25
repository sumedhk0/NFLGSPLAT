#!/bin/bash
#SBATCH --job-name=nfl-render
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out

# S4: per-play render. Composes field + library avatars + referees + football
# along the virtual-camera trajectory → MP4. Depends (afterok) on S3 so every
# player's avatar is in the library.
#
# Submit:  sbatch --export=ALL,NFL_GAME=game_001 scripts/slurm/render_array.sbatch

set -euo pipefail
GAME="${NFL_GAME:-}"
[[ -z "$GAME" ]] && { echo "must pass NFL_GAME via --export" >&2; exit 2; }

cd "$SLURM_SUBMIT_DIR"
PLAYS_FILE="configs/plays/${GAME}.txt"
[[ -f "$PLAYS_FILE" ]] || { echo "missing plays list: $PLAYS_FILE" >&2; exit 2; }

PLAY="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PLAYS_FILE")"
[[ -z "$PLAY" ]] && { echo "task ${SLURM_ARRAY_TASK_ID}: no play — skipping"; exit 0; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nfl_gsplat
python scripts/05_render_novel_view.py --game "$GAME" --play "$PLAY" \
    --season "${NFL_SEASON:-0}" \
    --trajectory configs/trajectories/fly_through.yaml
