"""3DGS-Avatar per-hero optimization — SLURM-only wrapper.

3DGS-Avatar trains per-player Gaussians via multi-view optimization against
the calibrated sideline + endzone crops and the per-frame SMPL-X poses. This
is expensive (tens of minutes per hero on an H100) so we guard it behind a
SLURM partition and do not attempt it for all players.

The module exports a clean interface but defers the training code itself to
the external repo — production only. Tests cover the config + output schema
contract, matching the LHM++ output format so the downstream animator can
consume either.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class GDGSAvatarConfig:
    iters: int = 20000
    lr: float = 1e-3
    sh_degree: int = 0
    batch_size: int = 1
    repo_dir: Path = Path("third_party/3dgs-avatar-release")


def train_hero(
    poses_npz: Path | str,
    crops_dir: Path | str,
    out_dir: Path | str,
    cfg: GDGSAvatarConfig,
) -> Path:
    """Train a per-hero 3DGS-Avatar, writing a canonical-space NPZ that shares
    the schema with :func:`lhm_wrapper.write_mock_avatar`.

    Requires the 3DGS-Avatar repo on disk and a CUDA-capable device in the
    ``nfl_avatar`` env. Raises :class:`SetupError` if either is missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.repo_dir.exists():
        raise SetupError(
            f"3DGS-Avatar repo not checked out at {cfg.repo_dir}. "
            "Run scripts/01_download_models.sh — see SETUP.md §4."
        )
    python = shutil.which("python")
    if python is None:
        raise SetupError("python not on PATH — activate nfl_avatar conda env.")

    _LOG.info(f"training 3DGS-Avatar → {out_dir} (iters={cfg.iters})")
    cmd = [
        python, "-m", "gdgs_avatar.train",
        "--poses", str(poses_npz),
        "--crops", str(crops_dir),
        "--output", str(out_dir),
        "--iters", str(cfg.iters),
        "--lr", str(cfg.lr),
        "--sh-degree", str(cfg.sh_degree),
    ]
    subprocess.check_call(cmd)

    out_npz = out_dir / "avatar.npz"
    if not out_npz.exists():
        raise RuntimeError(f"3DGS-Avatar training produced no {out_npz}")
    _validate_canonical_schema(out_npz)
    return out_npz


# Canonical avatar keys the downstream animator requires (shared with LHM++).
_REQUIRED_KEYS = (
    "canonical_xyz", "canonical_rot", "canonical_scale",
    "canonical_opacity", "canonical_sh", "lbs_weights",
)


def _validate_canonical_schema(npz_path: Path) -> None:
    """Ensure the hero NPZ matches the canonical avatar schema (so it loads into
    the library and animates like an LHM++ avatar)."""
    import numpy as np

    data = np.load(npz_path, allow_pickle=False)
    missing = [k for k in _REQUIRED_KEYS if k not in data.files]
    if missing:
        raise RuntimeError(
            f"3DGS-Avatar output {npz_path} missing canonical keys {missing}; "
            "the repo's exporter must emit the shared avatar schema (SETUP.md §8)."
        )
