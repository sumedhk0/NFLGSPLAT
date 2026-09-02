#!/bin/bash
#SBATCH --job-name=nfl-render
#SBATCH --partition=embers
#SBATCH --account=paceship-pso
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
# Real gsplat render of a play's per-frame Gaussian scenes (05d --ply-dir).
# embers is preemptible: frames already written are skipped on requeue only
# if the renderer is re-run over the same out dir, so keep it short.
#
#   sbatch scripts/render_play.sbatch <ply-dir> <out-dir> [turns] [radius-m]
module load anaconda3
conda run -n nfl_smplx python scripts/05h_render_scene_plys.py \
  --ply-dir "$1" --out-dir "$2" --backend gsplat --device cuda:0 \
  --turns "${3:-0.5}" --radius-m "${4:-30}"
