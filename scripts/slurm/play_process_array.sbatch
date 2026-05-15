#!/bin/bash
#SBATCH --job-name=nfl-plays
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/%x-%A_%a.out

# Array-job: process one play per array task. The play list for a game lives
# at configs/plays/{game_id}.txt (one play_id per line). Task i picks line i.
#
# Submit with:
#   sbatch --export=ALL,NFL_GAME=game_001 scripts/slurm/play_process_array.sbatch

set -euo pipefail

GAME="${NFL_GAME:-}"
if [[ -z "$GAME" ]]; then
    echo "must pass NFL_GAME via --export" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"
PLAYS_FILE="configs/plays/${GAME}.txt"
if [[ ! -f "$PLAYS_FILE" ]]; then
    echo "missing plays list: $PLAYS_FILE" >&2
    exit 2
fi

PLAY="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PLAYS_FILE")"
if [[ -z "$PLAY" ]]; then
    echo "task ${SLURM_ARRAY_TASK_ID} has no play in $PLAYS_FILE — skipping"
    exit 0
fi

bash scripts/04_process_play.sh "$GAME" "$PLAY"
