"""LHM++ feed-forward avatar generation with auto-VRAM tier selection.

The LHM family comes in two sizes:

- **LHM-1B**  full model, needs ≥ 16 GB free VRAM.
- **LHM-MINI** distilled, fits in ~8 GB.

Policy — strictly NO silent fallback:

- free ≥ 16 GB  → LHM-1B
- free in [8, 16) → LHM-MINI
- free < 8 GB  → :class:`LHMVRAMError`

INTEGRATION — *LHM-native* avatars (Option A).
LHM is an end-to-end animatable-avatar engine: its Gaussians carry no
per-Gaussian LBS weights, and it poses avatars internally via
``renderer.animate_gs_model(gs_attr, query_points, smplx_params)`` (SMPL-X
skinning applied to ``query_points`` at runtime). So an LHM avatar is stored as
its appearance Gaussians + the query points + the neutral-pose transform, and is
animated at render time by feeding our per-frame SMPL-X params to LHM's own
animator (see :mod:`nfl_gsplat.compositing.scene`). This is distinct from the
*canonical* format below, which our LBS renderer drives directly.

LHM-native schema (NPZ per LHM player), keys in :data:`LHM_NATIVE_KEYS`:

- app_xyz           [N, 3]      appearance Gaussian means (LHM/SMPL-X frame)
- app_rot           [N, 4]      wxyz quaternion
- app_scale         [N, 3]
- app_opacity       [N]
- app_sh            [N, K, 3]   SH coefficients (LHM layout)
- query_points      [L, 3]      SMPL-X query points the Gaussians attach to
- neutral_transform [*]         transform_mat_neutral_pose from infer_single_view
- tier              str ("lhm_1b" | "lhm_mini")

*Canonical* schema (mock / referee / 3DGS-hero), keys in
:data:`~nfl_gsplat.avatars.library.AVATAR_KEYS`, driven by ``animate_gaussians``:

- canonical_xyz/rot/scale/opacity/sh + lbs_weights [N, J] + tier

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

# LHM-native avatar NPZ keys (Option A): appearance Gaussians + the data LHM's
# own animator needs to pose them from per-frame SMPL-X params at render time.
LHM_NATIVE_KEYS = (
    "app_xyz", "app_rot", "app_scale", "app_opacity", "app_sh",
    "query_points", "neutral_transform",
)


def is_lhm_native(avatar: dict) -> bool:
    """True if ``avatar`` is the LHM-native format (vs the canonical LBS blob).

    Used by the library + render scene to route each avatar to the right
    animation path. Discriminates on keys, not just ``tier``, so a hand-built
    blob is classified correctly.
    """
    return all(k in avatar for k in LHM_NATIVE_KEYS)


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
    *,
    betas: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run LHM++ on a single reference crop → an LHM-native avatar (Option A).

    Requires the ``nfl_lhm`` conda env + pretrained weights. The result holds
    LHM's appearance Gaussians + query points + neutral transform; it is posed
    at render time by LHM's own animator (not our LBS renderer). ``betas`` is
    the frozen SMPL-X shape for this uid; if ``None``, LHM's own shape estimate
    is used. Raises :class:`SetupError` if the model code isn't available.
    """
    tier = pick_tier(cfg)
    _LOG.info(f"LHM tier selected: {tier}")
    model = _load_lhm_model(tier, cfg)
    raw = _forward_lhm(model, reference_crop, cfg, betas=betas)
    return _assemble_avatar(raw, tier)


_LHM_MODEL_CACHE: dict = {}


def _load_lhm_model(tier: LHMTier, cfg: LHMConfig):
    """Load + cache the LHM inferrer for ``tier`` (env-gated; ``nfl_lhm``).

    Mirrors ``third_party/LHM``: the entrypoint is ``HumanLRMInferrer`` (from
    ``LHM.runners.infer``), whose config/model selection is driven by the
    ``APP_MODEL_NAME`` env var (``LHM-1B`` / ``LHM-500M`` etc.). We set it from
    ``tier`` before constructing the inferrer, which loads the LHM model,
    SMPL-X pose estimator, and segmenter.

    NOTE (GPU bring-up): the exact env/cfg keys and the inferrer's preprocessing
    surface are finalized against the live repo at single-play bring-up; this
    wiring matches LHM.runners.infer.human_lrm as read from source.
    """
    if tier in _LHM_MODEL_CACHE:
        return _LHM_MODEL_CACHE[tier]
    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError as e:
        raise SetupError(
            "torch not installed — activate the `nfl_lhm` conda env. See SETUP.md §1."
        ) from e
    import os
    import sys

    sys.path.insert(0, str(cfg.repo_dir))
    # LHM selects its weights by model name via APP_MODEL_NAME (see parse_configs).
    os.environ.setdefault("APP_MODEL_NAME", _LHM_MODEL_NAME[tier])
    try:
        from LHM.runners.infer import HumanLRMInferrer  # type: ignore
    except Exception as e:  # noqa: BLE001 — setup-actionable
        raise SetupError(
            f"could not import LHM (LHM.runners.infer.HumanLRMInferrer) from "
            f"{cfg.repo_dir} ({e}). Confirm the checkout — see SETUP.md §8."
        ) from e
    model = HumanLRMInferrer()
    _LHM_MODEL_CACHE[tier] = model
    return model


# Maps our VRAM tier to LHM's published model names (parse_configs/AutoModelQuery).
_LHM_MODEL_NAME = {"lhm_1b": "LHM-1B", "lhm_mini": "LHM-500M"}


def _forward_lhm(model, reference_crop: np.ndarray, cfg: LHMConfig,
                 *, betas: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run the LHM inferrer on one reference crop → raw LHM-native dict.

    Mirrors ``HumanLRMInferrer.infer_mesh`` preprocessing, then calls
    ``model.infer_single_view(...)`` to obtain the appearance Gaussians,
    ``query_points`` and ``transform_mat_neutral_pose`` (the neutral build —
    no motion sequence). The crop is written to a temp image because LHM's
    preprocessing (segmentation, face crop, pose estimate) is path-driven.

    Seam: monkeypatched in tests so the schema assembly + library/render
    routing run without a GPU. GPU bring-up finalizes the exact attribute
    accessors on the returned GaussianModel against the live model.
    """
    import os
    import tempfile

    import cv2  # type: ignore
    import torch  # type: ignore

    with tempfile.TemporaryDirectory(prefix="lhm_ref_") as td:
        img_path = os.path.join(td, "ref.png")
        cv2.imwrite(img_path, reference_crop[:, :, ::-1])  # RGB -> BGR for cv2

        # Shape: frozen betas if given, else LHM's own pose-estimator beta.
        if betas is None:
            shape_param = model.pose_estimator(img_path).beta
        else:
            shape_param = np.asarray(betas, dtype=np.float32)

        parsing_mask = model.parsing(img_path)
        from LHM.runners.infer.human_lrm import infer_preprocess_image  # type: ignore

        image, _, _ = infer_preprocess_image(
            img_path, mask=parsing_mask, intr=None, pad_ratio=0, bg_color=1.0,
            max_tgt_size=896, aspect_standard=5.0 / 3, enlarge_ratio=[1.0, 1.0],
            render_tgt_size=model.cfg.source_size, multiply=14, need_mask=True,
        )
        try:
            src_head = cv2.resize(model.crop_face_image(img_path),
                                  (model.cfg.src_head_size, model.cfg.src_head_size))
        except Exception:  # noqa: BLE001 — head crop is optional in LHM
            src_head = np.zeros((model.cfg.src_head_size, model.cfg.src_head_size, 3),
                                dtype=np.uint8)
        src_head = torch.from_numpy(src_head / 255.0).float().permute(2, 0, 1).unsqueeze(0)

        device, dtype = "cuda", torch.float32
        smplx_params = _neutral_smplx_params(shape_param, device)
        model.model.to(dtype)
        gs_list, query_points, neutral_tf = model.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head.unsqueeze(0).to(device, dtype),
            None, None, None, None, None,
            smplx_params={k: v.to(device) for k, v in smplx_params.items()},
        )

    gs = gs_list[0]
    np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
    return {
        "app_xyz": np_(gs.xyz),
        "app_rot": np_(gs.rotation),
        "app_scale": np_(gs.scaling),
        "app_opacity": np_(gs.opacity),
        "app_sh": np_(gs.shs),
        "query_points": np_(query_points[0]),
        "neutral_transform": np_(neutral_tf),
    }


def _neutral_smplx_params(shape_param: np.ndarray, device: str) -> dict:
    """Neutral (rest-pose) SMPL-X params for the appearance build, matching the
    shapes LHM's ``infer_mesh`` uses. Torch import is lazy (caller is GPU-only)."""
    import torch  # type: ignore

    z = lambda *s: torch.zeros(*s)  # noqa: E731
    betas = torch.as_tensor(np.asarray(shape_param, dtype=np.float32)).reshape(1, -1)
    return {
        "betas": betas, "root_pose": z(1, 1, 3), "body_pose": z(1, 1, 21, 3),
        "jaw_pose": z(1, 1, 3), "leye_pose": z(1, 1, 3), "reye_pose": z(1, 1, 3),
        "lhand_pose": z(1, 1, 15, 3), "rhand_pose": z(1, 1, 15, 3),
        "expr": z(1, 1, 100), "trans": z(1, 1, 3),
    }


def _assemble_avatar(raw: dict[str, np.ndarray], tier: LHMTier) -> dict[str, np.ndarray]:
    """Map raw LHM output to the LHM-native avatar NPZ schema (Option A). Pure."""
    out = {k: np.asarray(raw[k], dtype=np.float32) for k in LHM_NATIVE_KEYS}
    out["tier"] = np.array([tier])
    return out


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
