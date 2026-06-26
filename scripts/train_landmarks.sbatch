#!/bin/bash
#SBATCH --job-name=nfl-landmarks
#SBATCH --partition=embers
#SBATCH --account=paceship-pso
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
# embers is preemptible: --requeue + per-epoch checkpoints make the job resume.
module load anaconda3
conda run -n nfl_smplx python -m nfl_gsplat.landmarks.train \
  --label "$1" --frames "$2" --out "$3" \
  --yard-min "${4:--25}" --yard-max "${5:-25}" \
  --epochs "${6:-200}" --batch-size 8 --lr 1e-3 --device cuda --resume
