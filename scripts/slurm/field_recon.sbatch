#!/bin/bash
#SBATCH --job-name=nfl-field
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

# Reconstruct the static field for one PLAY.
#
# The argument is a play DIRECTORY, not a game id: it is forwarded verbatim to
# 03_reconstruct_field.sh, which takes a play dir. Submitting a bare game id
# fails there, not here.
#
# Submit with:
#   sbatch scripts/slurm/field_recon.sbatch data/2025/week_04/SEA_at_AZ/play_001

set -euo pipefail

PLAY_DIR="${1:-}"
if [[ -z "$PLAY_DIR" ]]; then
    echo "usage: sbatch $0 <play-dir>" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"
bash scripts/03_reconstruct_field.sh "$PLAY_DIR"
