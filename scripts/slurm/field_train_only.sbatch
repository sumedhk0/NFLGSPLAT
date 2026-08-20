#!/bin/bash
#SBATCH --job-name=nfl-field-train
#SBATCH --account=paceship-pso
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
# middle stage (build_transforms) REGENERATES transforms.json from cameras.npz,
# which is only safe if cameras.npz was uploaded too -- and it is NOT needed
# here, so it usually has not been. Use this job whenever the poses were
# prepared elsewhere.
#
# Requires in <play-dir>/field/ :
#   transforms.json      the poses to train against
#   frames/<cam>/*.png   every image transforms.json references (relative paths)
#
# Videos, cameras.npz, tracks.parquet and meta.yaml are NOT needed here --
# nothing in this job re-derives a pose. play_001 is ~257 MB of frames plus a
# 92 KB transforms.json, against ~2 GB for the full play directory.
#
# --account defaults to paceship-pso; override per-submission with `sbatch -A`.
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

# Activate by PREFIX, not by name. These envs are built on scratch, outside
# conda's own envs dirs, and conda then lists them with an EMPTY name column --
# `conda activate nfl_gsplat` fails for an env that is sitting right there.
# $NFL_GSPLAT_PREFIX wins if set; otherwise find it by basename.
PREFIX="${NFL_GSPLAT_PREFIX:-}"
if [[ -z "$PREFIX" ]]; then
    while read -r p; do
        [[ "$(basename "$p")" == "nfl_gsplat" ]] && PREFIX="$p" && break
    done < <(conda env list | awk '!/^#/ && NF {print $NF}')
fi
if [[ -z "$PREFIX" || ! -d "$PREFIX" ]]; then
    echo "nfl_gsplat conda env not found." >&2
    echo "  build it:  bash scripts/00_setup_environments.sh --only nfl_gsplat" >&2
    echo "  or point at an existing one:  export NFL_GSPLAT_PREFIX=/path/to/env" >&2
    exit 2
fi
conda activate "$PREFIX"

for exe in ns-train ns-export; do
    command -v "$exe" >/dev/null 2>&1 || {
        echo "$exe not on PATH after activating $PREFIX -- the env exists but" >&2
        echo "nerfstudio is not installed in it. Re-run:" >&2
        echo "  bash scripts/00_setup_environments.sh --only nfl_gsplat" >&2
        exit 2
    }
done
echo "env: $PREFIX"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

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
