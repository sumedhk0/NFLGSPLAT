#!/bin/bash
#SBATCH --job-name=nfl-perception
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out

# S2: per-play perception (tracking → identity → SMPLest-X → triangulate → fuse
# → smooth → forward-kinematics → ball), plus this play's own field. NO avatar
# generation here — that is deduplicated into the avatar-build array (S3).
#
# Reads the play folder for array task N from outputs/play_worklist.txt (written
# by run_season.py). Partition + GPU are injected by run_season.py via sbatch CLI
# flags (--partition, --gres, -A), so they are not hard-coded here.
#
# Submit:  sbatch --array=1-$(wc -l < outputs/play_worklist.txt) scripts/slurm/perception_array.sbatch

set -euo pipefail
WORKLIST="${NFL_WORKLIST:-outputs/play_worklist.txt}"

cd "$SLURM_SUBMIT_DIR"
[[ -f "$WORKLIST" ]] || { echo "missing worklist: $WORKLIST" >&2; exit 2; }

PLAY_DIR="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")"
[[ -z "$PLAY_DIR" ]] && { echo "task ${SLURM_ARRAY_TASK_ID}: no play — skipping"; exit 0; }

bash scripts/04_process_play.sh "$PLAY_DIR" --perception-only
