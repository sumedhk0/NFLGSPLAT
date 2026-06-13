#!/bin/bash
#SBATCH --job-name=nfl-avatars
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out

# S3: build one player's avatar per array task (one task per unique uid). This
# is the dedup that makes the shared library cache correct under parallelism —
# no two tasks ever write the same data/{season}/_library/{uid}. Heroes route to
# 3DGS-Avatar (nfl_avatar env), everyone else to LHM++ (nfl_lhm).
#
# Submit:  sbatch --array=1-$(wc -l < outputs/avatar_worklist.txt) \
#                 --export=ALL,NFL_SEASON=2024 scripts/slurm/avatar_build_array.sbatch

set -euo pipefail
SEASON="${NFL_SEASON:-0}"
WORKLIST="${NFL_WORKLIST:-outputs/avatar_worklist.txt}"

cd "$SLURM_SUBMIT_DIR"
[[ -f "$WORKLIST" ]] || { echo "missing worklist: $WORKLIST" >&2; exit 2; }

UID_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")"
[[ -z "$UID_LINE" ]] && { echo "task ${SLURM_ARRAY_TASK_ID}: no uid — skipping"; exit 0; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nfl_lhm
python -m nfl_gsplat.avatars.build_one --season "$SEASON" --uid "$UID_LINE"
