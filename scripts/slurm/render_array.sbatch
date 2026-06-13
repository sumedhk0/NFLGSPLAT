#!/bin/bash
#SBATCH --job-name=nfl-render
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out

# S4: per-play render. Composes field + library avatars + referees + football
# along the virtual-camera trajectory → MP4. Depends (afterok) on S3 so every
# player's avatar is in the library.
#
# Reads the play folder for array task N from outputs/play_worklist.txt.
#
# Submit:  sbatch --array=1-$(wc -l < outputs/play_worklist.txt) scripts/slurm/render_array.sbatch

set -euo pipefail
WORKLIST="${NFL_WORKLIST:-outputs/play_worklist.txt}"

cd "$SLURM_SUBMIT_DIR"
[[ -f "$WORKLIST" ]] || { echo "missing worklist: $WORKLIST" >&2; exit 2; }

PLAY_DIR="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WORKLIST")"
[[ -z "$PLAY_DIR" ]] && { echo "task ${SLURM_ARRAY_TASK_ID}: no play — skipping"; exit 0; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nfl_gsplat
python scripts/05_render_novel_view.py --play-dir "$PLAY_DIR" \
    --trajectory configs/trajectories/fly_through.yaml
