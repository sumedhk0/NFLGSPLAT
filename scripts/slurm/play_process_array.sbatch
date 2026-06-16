#!/bin/bash
#SBATCH --job-name=nfl-plays
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=1-22
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out

# Array-job: process one play per array task. Reads play folder paths from
# outputs/play_worklist.txt (one absolute or relative play-folder path per
# line). Task i picks line i and runs the full per-play driver.
#
# Submit with:
#   sbatch --array=1-$(wc -l < outputs/play_worklist.txt) \
#          scripts/slurm/play_process_array.sbatch

set -euo pipefail
WORKLIST="${NFL_WORKLIST:-outputs/play_worklist.txt}"
cd "$SLURM_SUBMIT_DIR"
[[ -f "$WORKLIST" ]] || { echo "missing worklist: $WORKLIST" >&2; exit 2; }
PLAY_DIR="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")"
[[ -z "$PLAY_DIR" ]] && { echo "task ${SLURM_ARRAY_TASK_ID}: no play — skipping"; exit 0; }
bash scripts/04_process_play.sh "$PLAY_DIR"
