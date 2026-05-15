"""SMPLest-X-H32 per-camera per-frame inference.

Lazy torch / SMPLest-X import so this module is safe to import from CPU envs
(tests, tracking). Real inference requires the ``nfl_smplx`` conda env plus
the SMPLest-X repo checkout + pretrained weights.

Output cache schema (one NPZ per ``(cam, frame, global_player_id)``)::

    betas           [10]
    body_pose       [21, 3]       axis-angle per body joint
    global_orient   [3]
    transl          [3]
    joints3d_cam    [127, 3]      SMPL-X all-joints in camera coords
    joints2d        [127, 2]      pixel coords (re-projected)
    confidence      [127]

Only the first 22 joints are used by triangulation; the rest (hands/face)
are cached for future use.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class SMPLestXConfig:
    repo_dir: Path = Path("third_party/SMPLest-X")
    weights_path: Path = Path("data/body_models/smplest_x_h32.pth")
    device: str = "cuda:0"
    batch_size: int = 4
    input_size: int = 512


def _lazy_import():
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise SetupError(
            "torch not installed — activate the `nfl_smplx` conda env "
            "(`conda activate nfl_smplx`). See SETUP.md §1."
        ) from e
    return torch


def check_prerequisites(cfg: SMPLestXConfig) -> None:
    """Raise :class:`SetupError` if model code or weights are missing.

    Run this once at pipeline start so the failure mode is immediate and the
    error message names the exact missing path (per project error philosophy).
    """
    if not cfg.repo_dir.exists():
        raise SetupError(
            f"SMPLest-X repo not checked out at {cfg.repo_dir}. "
            "Run scripts/01_download_models.sh — see SETUP.md §4."
        )
    if not cfg.weights_path.exists():
        raise SetupError(
            f"SMPLest-X weights missing at {cfg.weights_path}. "
            "Run scripts/01_download_models.sh — see SETUP.md §4."
        )


def infer_crops(
    crops: np.ndarray,       # [N, H, W, 3] uint8 RGB
    bboxes: np.ndarray,      # [N, 4] image-space (x1, y1, x2, y2)
    cfg: SMPLestXConfig,
) -> dict[str, np.ndarray]:
    """Run SMPLest-X-H32 on a batch of player crops.

    Returns a dict of stacked per-sample outputs (shapes with leading ``N``).
    This is a thin wrapper; all heavy lifting is inside SMPLest-X. Kept here
    so the rest of the pose pipeline sees a stable dict schema.
    """
    check_prerequisites(cfg)
    _lazy_import()
    # Deliberately unimplemented beyond the stub; the real adapter lives inside
    # the nfl_smplx conda env and is wired up via scripts/04_process_play.sh.
    # Keeping the signature stable lets the orchestration script mock it during
    # CI without importing torch.
    raise NotImplementedError(
        "SMPLest-X adapter is env-gated; run inside the nfl_smplx conda env via "
        "scripts/04_process_play.sh. See SETUP.md §8 for the adapter wiring."
    )


def write_inference_cache(
    out_dir: Path | str,
    cam: str,
    frame_idx: int,
    global_player_id: int,
    result: dict[str, np.ndarray],
) -> Path:
    """Write one NPZ per ``(cam, frame, global_player_id)``. Thin helper so
    the main inference script calls this instead of open-coding atomic writes.
    """
    from nfl_gsplat.utils.io import write_npz

    out_dir = Path(out_dir)
    fname = f"{cam}__f{frame_idx:06d}__p{global_player_id:04d}.npz"
    path = out_dir / fname
    write_npz(path, **{k: np.asarray(v) for k, v in result.items()})
    return path
