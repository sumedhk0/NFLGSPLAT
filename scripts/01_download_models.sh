#!/usr/bin/env bash
# Fetch non-gated pretrained weights + repo checkouts.
#
# Gated weights (SMPL-X .npz from MPI) are NOT downloaded here — the user
# must accept the license and place them under data/body_models/. See
# SETUP.md §3.
#
# Uses aria2c when available (parallel, resumable) and falls back to wget -c.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_ROOT/data/body_models"
REPOS_DIR="$REPO_ROOT/third_party"

mkdir -p "$MODELS_DIR" "$REPOS_DIR"

_fetch() {
    local url="$1" out="$2"
    if [[ -f "$out" ]]; then
        echo "  skip (exists): $out"
        return 0
    fi
    echo "  → $url"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 8 -s 8 --continue=true -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
    else
        wget --continue -O "$out" "$url"
    fi
}

_clone_or_pull() {
    local url="$1" dir="$2" rev="${3:-}"
    if [[ -d "$dir/.git" ]]; then
        git -C "$dir" fetch --tags
    else
        git clone "$url" "$dir"
    fi
    if [[ -n "$rev" ]]; then
        git -C "$dir" checkout "$rev"
    fi
}

echo "=== YOLOv8 person weights ==="
_fetch https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt "$MODELS_DIR/yolov8x.pt"

echo
echo "=== Football-YOLO (fine-tuned for the 'sports ball' class) ==="
# Placeholder URL — replace with the actual release once the user fine-tunes
# (or swap to a community-published weights file). See SETUP.md §3.
echo "  NOTE: football-specific YOLO weights are engagement-specific."
echo "  Place a fine-tuned checkpoint at $MODELS_DIR/ball_yolov8.pt."

echo
echo "=== SMPLest-X-H32 ==="
_clone_or_pull https://github.com/wqyin/SMPLest-X "$REPOS_DIR/SMPLest-X"
echo "  NOTE: place the pretrained model under"
echo "  third_party/SMPLest-X/pretrained_models/smplest_x_h/ as"
echo "  smplest_x_h.pth.tar + config_base.py (see SETUP.md §4)."

echo
echo "=== LHM++ ==="
_clone_or_pull https://github.com/aigc3d/LHM "$REPOS_DIR/LHM"
echo "  NOTE: LHM-1B and LHM-MINI weights live on Alibaba OSS; see the"
echo "  LHM repo README for bucket URLs. Expect a slow first fetch from US."

echo
echo "=== 3DGS-Avatar ==="
_clone_or_pull https://github.com/mikeqzy/3dgs-avatar-release "$REPOS_DIR/3dgs-avatar-release"

echo
echo "=== ViTPose (optional — only if LHM's own reference-pose step needs it) ==="
_clone_or_pull https://github.com/ViTAE-Transformer/ViTPose "$REPOS_DIR/ViTPose"
echo "  NOTE: our pipeline does not call ViTPose directly (reference selection"
echo "  uses SMPLest-X confidence). If LHM's avatar build asks for ViTPose"
echo "  weights at runtime, fetch them per the ViTPose README into $MODELS_DIR/."

echo
echo "done. Repos cloned + YOLOv8x fetched. The heavy model weights are manual:"
echo "  - data/body_models/smplx/*  + smpl/*   (gated; SETUP.md §2)"
echo "  - third_party/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar  (§4)"
echo "  - LHM-1B / LHM-MINI weights             (LHM repo / Alibaba OSS; §4)"
echo "  - data/body_models/ball_yolov8.pt       (engagement-specific; optional)"
