#!/bin/bash
#SBATCH --job-name=nfl-field-train
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=embers
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

# Train the field splat from an EXISTING field/transforms.json.
#
# Use this, not field_recon.sbatch, whenever transforms.json was prepared
# elsewhere. field_recon.sbatch runs the full three-stage pipeline, and its
# middle stage (build_transforms) REGENERATES transforms.json from cameras.npz
# -- silently discarding any curation done beforehand. On play_001 that would
# replace a hand-verified 37-camera file with all 88 poses, including endzone
# frames measured 15-33 px off, and train on them.
#
# Requires in <play-dir>/field/ :
#   transforms.json      the poses to train against
#   frames/<cam>/*.png   every image transforms.json references (relative paths)
#
# Videos, cameras.npz and meta.yaml are NOT needed here -- nothing in this job
# re-derives a pose.
#
# Submit with:
#   sbatch scripts/slurm/field_train_only.sbatch data/2025/week_04/SEA_at_AZ/play_001

set -euo pipefail

PLAY_DIR="${1:-}"
if [[ -z "$PLAY_DIR" ]]; then
    echo "usage: sbatch $0 <play-dir>" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"

TJ="$PLAY_DIR/field/transforms.json"
if [[ ! -f "$TJ" ]]; then
    echo "missing $TJ -- copy the prepared field/ directory up first" >&2
    exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nfl_gsplat

python - "$TJ" <<'PY'
import json, sys, pathlib
tj = pathlib.Path(sys.argv[1])
d = json.load(open(tj))
missing = [f["file_path"] for f in d["frames"]
           if not (tj.parent / f["file_path"]).exists()]
if missing:
    raise SystemExit(f"{len(missing)} referenced image(s) not present, e.g. {missing[:3]}")
print(f"transforms.json: {len(d['frames'])} poses, all images present")
PY

python -m nfl_gsplat.field.train_field \
    --play-dir "$PLAY_DIR" \
    --config configs/field_recon.yaml

echo "field.ply -> $PLAY_DIR/field.ply"
