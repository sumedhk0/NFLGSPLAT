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
    # Find the env by PREFIX PATH, not by name. When envs live outside conda's
    # default envs_dirs -- normal on a cluster, where they go on scratch --
    # `conda env list` prints them with an EMPTY name column, just the path. A
    # name match then finds nothing, the script takes the create branch, and
    # conda refuses with "prefix already exists" for an env that is right there.
    # Matching the directory's basename works in both layouts, and updating by
    # -p works even when the env was never registered under a name.
    prefix=""
    while read -r p; do
        [[ -n "$p" && "$(basename "$p")" == "$env" ]] && { prefix="$p"; break; }
    done < <(conda env list | awk '!/^#/ && NF {print $NF}')

    if [[ -n "$prefix" ]]; then
        echo "--- updating existing env at $prefix ---"
        conda env update -p "$prefix" -f "$yaml" --prune
    else
        conda env create -n "$env" -f "$yaml"
        prefix="$(conda env list | awk '!/^#/ && NF {print $NF}'                   | while read -r p; do                       [[ "$(basename "$p")" == "$env" ]] && echo "$p" && break;                     done)"
    fi

    # chumpy 0.70's setup.py imports `pip`, which is absent in pip's isolated
    # build env → "ModuleNotFoundError: No module named 'pip'". Install it
    # against the env's real pip, after the wheel for mmcv etc. is in place.
    if [[ "$env" == "nfl_smplx" ]]; then
        echo "--- post-build: chumpy (no-build-isolation) ---"
        # -p for the same reason as above: the name may not be registered.
        conda run -p "$prefix" python -m pip install -U pip setuptools wheel
        conda run -p "$prefix" python -m pip install --no-build-isolation chumpy==0.70
    fi
done

echo
echo "done. Next: bash scripts/01_download_models.sh"
