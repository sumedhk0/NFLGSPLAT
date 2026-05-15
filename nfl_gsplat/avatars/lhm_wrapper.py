"""LHM++ feed-forward avatar generation with auto-VRAM tier selection.

The LHM family comes in two sizes:

- **LHM-1B**  full model, needs ≥ 16 GB free VRAM.
- **LHM-MINI** distilled, fits in ~8 GB.

Policy — strictly NO silent fallback:

- free ≥ 16 GB  → LHM-1B
- free in [8, 16) → LHM-MINI
- free < 8 GB  → :class:`LHMVRAMError`

Output schema (NPZ per player):

- canonical_xyz       [N, 3]
- canonical_rot       [N, 4]  wxyz quaternion
- canonical_scale     [N, 3]
- canonical_opacity   [N]
- canonical_sh        [N, 3, K]  SH coefficients (K depends on degree)
- lbs_weights         [N, J]    J = SMPL-X body joints (22 for our body-only fit)
- tier                str ("lhm_1b" | "lhm_mini" | "mock")

For CI / CPU-only envs :func:`write_mock_avatar` emits a canonical blob that
satisfies the smoke-test schema without any GPU work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from nfl_gsplat.errors import LHMVRAMError, SetupError
from nfl_gsplat.utils.io import write_npz
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

LHMTier = Literal["lhm_1b", "lhm_mini", "mock"]


@dataclass(frozen=True)
class LHMConfig:
    model_choice: str = "auto"       # auto | lhm_1b | lhm_mini
    vram_floor_gb: float = 8.0
    num_body_joints: int = 22        # matches our triangulation output
    sh_degree: int = 0


def _free_vram_gb() -> float:
    """Return free VRAM on device 0 in GB. Returns 0.0 if torch/CUDA unavailable.
    That triggers a clean failure path in :func:`pick_tier`.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total = torch.cuda.mem_get_info(0)
    return float(free_bytes) / (1024 ** 3)


def pick_tier(cfg: LHMConfig, free_gb: float | None = None) -> LHMTier:
    """Choose an LHM tier according to the policy above, or raise
    :class:`LHMVRAMError`. Explicit ``cfg.model_choice`` overrides auto-detect
    but still fails if VRAM is insufficient for the explicit choice."""
    free = _free_vram_gb() if free_gb is None else free_gb

    if cfg.model_choice == "lhm_1b":
        if free < 16.0:
            raise LHMVRAMError(
                f"lhm_1b requested but only {free:.1f} GB free (need >= 16 GB). "
                "Either run on a larger GPU or set avatars.lhm.model=auto."
            )
        return "lhm_1b"
    if cfg.model_choice == "lhm_mini":
        if free < cfg.vram_floor_gb:
            raise LHMVRAMError(
                f"lhm_mini requires >= {cfg.vram_floor_gb:.1f} GB, only {free:.1f} GB free."
            )
        return "lhm_mini"
    if cfg.model_choice != "auto":
        raise ValueError(f"unknown LHMConfig.model_choice: {cfg.model_choice}")

    if free < cfg.vram_floor_gb:
        raise LHMVRAMError(
            f"LHM requires >= {cfg.vram_floor_gb:.1f} GB free VRAM, only {free:.1f} GB available. "
            "See SETUP.md §4 for hardware minimums."
        )
    return "lhm_1b" if free >= 16.0 else "lhm_mini"


def generate_avatar(
    reference_crop: np.ndarray,    # [H, W, 3] uint8
    cfg: LHMConfig,
) -> dict[str, np.ndarray]:
    """Run LHM++ on a single reference crop. Requires the ``nfl_lhm`` conda
    env + pretrained weights. This is the production path — raises
    :class:`SetupError` if the model code isn't available.
    """
    tier = pick_tier(cfg)
    _LOG.info(f"LHM tier selected: {tier}")
    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError as e:
        raise SetupError(
            "torch not installed — activate the `nfl_lhm` conda env. See SETUP.md §1."
        ) from e
    # The actual LHM adapter lives in the external repo and is loaded by the
    # orchestration script inside the nfl_lhm env. We keep this function as a
    # stable seam so the rest of the pipeline's imports stay clean.
    raise NotImplementedError(
        "LHM adapter is env-gated; run inside nfl_lhm via scripts/04_process_play.sh. "
        "See SETUP.md §8."
    )


# --- Mock avatar (CPU smoke test) ------------------------------------------

def write_mock_avatar(
    out_path: Path | str,
    *,
    num_gaussians: int = 3000,
    num_joints: int = 22,
    sh_degree: int = 0,
    seed: int = 0,
) -> Path:
    """Write a canonical-space Gaussian blob + synthetic LBS weights.

    - Gaussians are scattered in a 1-m tall capsule centered on the pelvis.
    - LBS weights are hard-assigned to the single nearest body joint (one-hot).
      This makes the mock useful for testing :func:`animate_gaussians` because
      one-hot weights recover the underlying joint transform exactly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    xyz = np.column_stack([
        rng.normal(0.0, 0.18, num_gaussians),          # x: lateral
        rng.normal(0.0, 0.12, num_gaussians),          # y: front-back
        rng.uniform(-0.90, 0.70, num_gaussians),       # z: pelvis ± 1 m
    ]).astype(np.float32)

    rot = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (num_gaussians, 1))
    scale = np.full((num_gaussians, 3), np.log(0.04), dtype=np.float32)
    opacity = np.full((num_gaussians,), 2.0, dtype=np.float32)

    K_sh = (sh_degree + 1) ** 2
    sh = np.zeros((num_gaussians, 3, K_sh), dtype=np.float32)
    sh[:, 0, 0] = rng.uniform(0.30, 0.80, num_gaussians)     # R (SH DC)
    sh[:, 1, 0] = rng.uniform(0.20, 0.60, num_gaussians)     # G
    sh[:, 2, 0] = rng.uniform(0.10, 0.40, num_gaussians)     # B

    # One-hot LBS weights to the joint whose z-height is closest. Simplistic
    # but makes the mock behave deterministically under animate_gaussians.
    joint_z = np.linspace(-0.9, 0.9, num_joints)
    dist = np.abs(xyz[:, 2, None] - joint_z[None, :])        # [N, J]
    nearest = np.argmin(dist, axis=1)
    lbs = np.zeros((num_gaussians, num_joints), dtype=np.float32)
    lbs[np.arange(num_gaussians), nearest] = 1.0

    write_npz(
        out_path,
        canonical_xyz=xyz,
        canonical_rot=rot,
        canonical_scale=scale,
        canonical_opacity=opacity,
        canonical_sh=sh,
        lbs_weights=lbs,
        tier=np.array(["mock"]),
    )
    _LOG.info(f"wrote mock LHM avatar (N={num_gaussians}, J={num_joints}) → {out_path}")
    return out_path
