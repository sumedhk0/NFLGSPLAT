#!/bin/bash
# Build the nfl_gsplat environment with uv instead of conda.
#
# Covers ONLY nfl_gsplat -- the env the field-training job needs. The other
# three (nfl_smplx, nfl_lhm, nfl_avatar) stay on conda: nfl_smplx in particular
# needs chumpy built with --no-build-isolation and a PaddleOCR stack that conda
# already resolves, and there is no reason to churn them.
#
# Why: conda took 30-60 minutes on this env and failed twice in package
# extraction (CondaVerificationError: libabseil "appears to be corrupted"),
# which is a full or inode-capped filesystem rather than a bad package. uv
# resolves the identical pin set in ~23 s, installs only wheels -- no package in
# the lock needs a compiler at install time -- and puts everything in one venv
# directory that is cheap to delete and rebuild.
#
# Usage:
#   bash scripts/00_setup_uv_gsplat.sh [venv-path]
#
# Default venv path is $NFL_GSPLAT_PREFIX, else ./.venv-gsplat.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${1:-${NFL_GSPLAT_PREFIX:-$REPO_ROOT/.venv-gsplat}}"
LOCK="envs/requirements-gsplat-linux-py310.txt"

if [[ ! -f "$LOCK" ]]; then
    echo "missing $LOCK -- regenerate it with the command in envs/requirements-gsplat.in" >&2
    exit 2
fi

# uv installs to ~/.local/bin and needs no root, which is what makes this
# usable on a cluster login node.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
    echo "uv still not on PATH after install -- add ~/.local/bin to PATH" >&2
    exit 2
}
echo "uv: $(uv --version)"

# uv fetches a standalone CPython, so this does not depend on a python3.10
# module being available or on whatever the system python happens to be.
uv venv --python 3.10 "$VENV"

# --extra-index-url + unsafe-best-match are what pull torch 2.3.1+cu121 rather
# than the CPU build; the lock pins the +cu121 local version, so a plain PyPI
# install would simply fail to find it.
uv pip install --python "$VENV" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match \
    -r "$LOCK"

echo
echo "installed. verifying:"
"$VENV/bin/python" - <<'PY'
import torch, gsplat, nerfstudio
print("  torch      ", torch.__version__)
print("  gsplat     ", gsplat.__version__)
print("  nerfstudio ", nerfstudio.__version__)
print("  cuda avail ", torch.cuda.is_available(), "(False on a login node is expected)")
PY

for exe in ns-train ns-export; do
    [[ -x "$VENV/bin/$exe" ]] || { echo "MISSING $VENV/bin/$exe" >&2; exit 2; }
done
echo "  ns-train/ns-export present"
echo
echo "use it with:  export NFL_GSPLAT_PREFIX=$VENV"
echo
echo "NOTE: gsplat JIT-compiles its CUDA kernels on first use, so the JOB needs"
echo "nvcc on PATH (module load cuda/12.1.1 or similar). No wheel ships nvcc,"
echo "and neither did conda's pytorch-cuda -- this is not new with uv."
