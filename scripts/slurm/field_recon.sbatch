#!/bin/bash
#SBATCH --job-name=nfl-field
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Reconstruct the static field for one game.
# Submit with:  sbatch scripts/slurm/field_recon.sbatch game_001

set -euo pipefail

GAME="${1:-}"
if [[ -z "$GAME" ]]; then
    echo "usage: sbatch $0 <game_id>" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"
bash scripts/03_reconstruct_field.sh "$GAME"
