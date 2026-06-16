#!/bin/bash
#SBATCH --job-name=nfl-game
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

# DEPRECATED: full_game.sbatch does not map to the per-play tree layout.
# Use the season DAG runner instead — it discovers plays automatically and
# submits the staged SLURM array via play_process_array.sbatch.
#
# Supported path:
#   python scripts/run_season.py --config configs/season.yaml --submit

set -euo pipefail
echo "full_game.sbatch is deprecated under the per-play layout." >&2
echo "Use: python scripts/run_season.py --config configs/season.yaml --submit" >&2
echo "(discovers plays from data/{season}/week_NN/{matchup}/play_NNN and submits the staged DAG)." >&2
exit 1
