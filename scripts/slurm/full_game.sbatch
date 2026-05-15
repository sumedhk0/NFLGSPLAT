#!/bin/bash
#SBATCH --job-name=nfl-game
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Submit the full chain for one game: field reconstruction, then a play-array
# job depending on it. This driver itself is a CPU-only meta-job — the actual
# GPU work happens in the submitted children.
#
# Usage:  sbatch scripts/slurm/full_game.sbatch game_001

set -euo pipefail

GAME="${1:-}"
if [[ -z "$GAME" ]]; then
    echo "usage: sbatch $0 <game_id>" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PLAYS_FILE="configs/plays/${GAME}.txt"
if [[ ! -f "$PLAYS_FILE" ]]; then
    echo "missing plays list: $PLAYS_FILE (one play_id per line)" >&2
    exit 2
fi

NUM_PLAYS="$(wc -l < "$PLAYS_FILE")"

echo "=== submitting field_recon for $GAME ==="
FIELD_JOB="$(sbatch --parsable scripts/slurm/field_recon.sbatch "$GAME")"
echo "field_recon job id: $FIELD_JOB"

echo "=== submitting play array (1-$NUM_PLAYS) depending on $FIELD_JOB ==="
PLAY_JOB="$(sbatch --parsable \
    --dependency=afterok:"$FIELD_JOB" \
    --array=1-"$NUM_PLAYS" \
    --export=ALL,NFL_GAME="$GAME" \
    scripts/slurm/play_process_array.sbatch)"
echo "play_array job id: $PLAY_JOB"

echo
echo "submitted. Watch with:"
echo "  squeue -u \$USER"
