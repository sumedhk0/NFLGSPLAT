#!/usr/bin/env bash
# Create the four conda environments used by the pipeline. Each env pins one
# CUDA / torch combo so incompatible stage dependencies can coexist.
#
# Usage: bash scripts/00_setup_environments.sh [--only nfl_smplx|nfl_gsplat|nfl_lhm|nfl_avatar]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$REPO_ROOT/envs"

ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v conda >/dev/null 2>&1 || {
    echo "conda not found on PATH. Install miniforge: https://conda-forge.org/" >&2
    exit 1
}

ENVS=(nfl_smplx nfl_gsplat nfl_lhm nfl_avatar)

for env in "${ENVS[@]}"; do
    if [[ -n "$ONLY" && "$env" != "$ONLY" ]]; then continue; fi
    yaml="$ENV_DIR/environment_${env#nfl_}.yml"
    if [[ ! -f "$yaml" ]]; then
        echo "missing env YAML: $yaml" >&2
        exit 1
    fi
    echo "=== building $env from $yaml ==="
    if conda env list | awk '{print $1}' | grep -Fxq "$env"; then
        conda env update -n "$env" -f "$yaml" --prune
    else
        conda env create -n "$env" -f "$yaml"
    fi

    # chumpy 0.70's setup.py imports `pip`, which is absent in pip's isolated
    # build env → "ModuleNotFoundError: No module named 'pip'". Install it (and
    # VPoser, which depends on it) against the env's real pip, after the wheel
    # for mmcv etc. is already in place.
    if [[ "$env" == "nfl_smplx" ]]; then
        echo "--- post-build: chumpy (no-build-isolation) + human-body-prior ---"
        conda run -n nfl_smplx python -m pip install -U pip setuptools wheel
        conda run -n nfl_smplx python -m pip install --no-build-isolation chumpy==0.70
        conda run -n nfl_smplx python -m pip install human-body-prior==2.2.2.0
    fi
done

echo
echo "done. Next: bash scripts/01_download_models.sh"
