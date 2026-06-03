#!/bin/bash
#SBATCH --job-name=nfl-perception
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out

# S2: per-play perception (tracking → identity → SMPLest-X → triangulate → fuse
# → smooth → forward-kinematics → ball). NO avatar generation here — that is
# deduplicated into the avatar-build array (S3) so each player is built once.
#
# Partition + GPU are injected by run_season.py via sbatch CLI flags
# (--partition, --gres, -A), so they are not hard-coded here.
#
# Submit:  sbatch --export=ALL,NFL_GAME=game_001 scripts/slurm/perception_array.sbatch

set -euo pipefail
GAME="${NFL_GAME:-}"
[[ -z "$GAME" ]] && { echo "must pass NFL_GAME via --export" >&2; exit 2; }

cd "$SLURM_SUBMIT_DIR"
PLAYS_FILE="configs/plays/${GAME}.txt"
[[ -f "$PLAYS_FILE" ]] || { echo "missing plays list: $PLAYS_FILE" >&2; exit 2; }

PLAY="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PLAYS_FILE")"
[[ -z "$PLAY" ]] && { echo "task ${SLURM_ARRAY_TASK_ID}: no play — skipping"; exit 0; }

bash scripts/04_process_play.sh "$GAME" "$PLAY" --perception-only
