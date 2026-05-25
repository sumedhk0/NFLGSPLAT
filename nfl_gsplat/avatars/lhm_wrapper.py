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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Literal

import numpy as np

from nfl_gsplat.errors import LHMVRAMError, SetupError
from nfl_gsplat.identity.registry import REFEREE_UID, EntityType
from nfl_gsplat.utils.io import write_npz
from nfl_gsplat.utils.logging import get_logger

if TYPE_CHECKING:
    from nfl_gsplat.avatars.library import AvatarLibrary

_LOG = get_logger(__name__)

LHMTier = Literal["lhm_1b", "lhm_mini", "mock"]


@dataclass(frozen=True)
class LHMConfig:
    model_choice: str = "auto"       # auto | lhm_1b | lhm_mini
    vram_floor_gb: float = 8.0
    num_body_joints: int = 22        # matches our triangulation output
    sh_degree: int = 0
    repo_dir: Path = Path("third_party/LHM")
    weights_dir: Path = Path("data/body_models")


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
    model = _load_lhm_model(tier, cfg)
    raw = _forward_lhm(model, reference_crop, cfg)
    return _assemble_avatar(raw, tier, cfg)


_LHM_MODEL_CACHE: dict = {}


def _load_lhm_model(tier: LHMTier, cfg: LHMConfig):
    """Load + cache the LHM++ model for ``tier`` (env-gated; ``nfl_lhm``)."""
    if tier in _LHM_MODEL_CACHE:
        return _LHM_MODEL_CACHE[tier]
    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError as e:
        raise SetupError(
            "torch not installed — activate the `nfl_lhm` conda env. See SETUP.md §1."
        ) from e
    import sys

    sys.path.insert(0, str(cfg.repo_dir))
    try:
        from LHM.runners.infer import build_model  # type: ignore
    except Exception as e:  # noqa: BLE001 — setup-actionable
        raise SetupError(
            f"could not import LHM from {cfg.repo_dir} ({e}). Confirm the checkout + "
            "entrypoint — see SETUP.md §8."
        ) from e
    model = build_model(tier, weights_dir=str(cfg.weights_dir))
    _LHM_MODEL_CACHE[tier] = model
    return model


def _forward_lhm(model, reference_crop: np.ndarray, cfg: LHMConfig) -> dict[str, np.ndarray]:
    """Run LHM++ on one reference crop → raw canonical-Gaussian dict. Seam:
    monkeypatched in tests so the schema assembly runs without a GPU."""
    return model.reconstruct(reference_crop, sh_degree=cfg.sh_degree,
                             num_joints=cfg.num_body_joints)


def _assemble_avatar(raw: dict[str, np.ndarray], tier: LHMTier,
                     cfg: LHMConfig) -> dict[str, np.ndarray]:
    """Map raw LHM output to the canonical avatar NPZ schema. Pure."""
    n = np.asarray(raw["xyz"]).shape[0]
    K_sh = (cfg.sh_degree + 1) ** 2
    return {
        "canonical_xyz": np.asarray(raw["xyz"], dtype=np.float32).reshape(n, 3),
        "canonical_rot": np.asarray(raw["rot"], dtype=np.float32).reshape(n, 4),
        "canonical_scale": np.asarray(raw["scale"], dtype=np.float32).reshape(n, 3),
        "canonical_opacity": np.asarray(raw["opacity"], dtype=np.float32).reshape(n),
        "canonical_sh": np.asarray(raw["sh"], dtype=np.float32).reshape(n, 3, K_sh),
        "lbs_weights": np.asarray(raw["lbs_weights"], dtype=np.float32).reshape(n, cfg.num_body_joints),
        "tier": np.array([tier]),
    }


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


# --- Reference selection + library short-circuit ----------------------------

def select_reference_index(
    bbox_areas: np.ndarray,
    confidences: np.ndarray,
    *,
    min_conf: float = 0.4,
) -> int:
    """Pick the best reference detection for the one-time avatar build.

    Implements ``avatars.lhm.reference_selection`` = ``bbox_area_times_vitpose_conf``:
    among detections with ``conf >= min_conf``, choose the one maximizing
    ``bbox_area * conf``. Returns -1 if none clear the confidence gate.
    """
    areas = np.asarray(bbox_areas, dtype=np.float64)
    confs = np.asarray(confidences, dtype=np.float64)
    eligible = confs >= min_conf
    if not eligible.any():
        return -1
    score = np.where(eligible, areas * confs, -np.inf)
    return int(np.argmax(score))


@dataclass
class AvatarPlan:
    """Outcome of resolving a play's entities against the library."""

    avatars: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    cache_hits: list[str] = field(default_factory=list)
    generated: list[str] = field(default_factory=list)
    referees: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def resolve_avatars(
    entities: Iterable[tuple[str, str]],
    library: "AvatarLibrary",
    generate_fn: Callable[[str], dict[str, np.ndarray]],
    *,
    provenance: dict | None = None,
) -> AvatarPlan:
    """Decide, per entity, whether to load a cached avatar or build one.

    ``entities`` is an iterable of ``(player_uid, entity_type)`` for a play.
    Policy:

    - ``PLAYER`` with a library hit → load (skips LHM++); otherwise call
      ``generate_fn(uid)`` (the env-gated LHM++ build) and cache the result.
    - ``REFEREE`` → the single generic striped-shirt avatar (authored once).
    - ``OTHER`` / empty uid → dropped.

    Each unique uid is processed once, so a player appearing on many tracks
    costs at most one generation. ``generate_fn`` is injected so the cache
    logic is testable without a GPU.
    """
    plan = AvatarPlan()
    seen: set[str] = set()
    for uid, etype in entities:
        if etype == EntityType.OTHER.value or not uid:
            plan.dropped.append(uid)
            continue
        if uid in seen:
            continue
        seen.add(uid)

        if etype == EntityType.REFEREE.value or uid == REFEREE_UID:
            if not library.has_referee_avatar():
                raise SetupError(
                    "generic referee avatar missing from the library "
                    "(library/{season}/_assets/referee/avatar.npz). Author it "
                    "once via the referee asset step — see SETUP.md §8."
                )
            plan.avatars[REFEREE_UID] = library.get_referee_avatar()
            plan.referees.append(REFEREE_UID)
            continue

        if library.has_avatar(uid):
            plan.avatars[uid] = library.get_avatar(uid)
            plan.cache_hits.append(uid)
        else:
            avatar = generate_fn(uid)
            library.put_avatar(uid, avatar, provenance=provenance)
            plan.avatars[uid] = avatar
            plan.generated.append(uid)

    _LOG.info(
        f"avatar plan: {len(plan.cache_hits)} cache hits, "
        f"{len(plan.generated)} generated, {len(plan.referees)} referee, "
        f"{len(plan.dropped)} dropped"
    )
    return plan
