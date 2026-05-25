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


_MODEL_CACHE: dict = {}

# Number of SMPL-X joints SMPLest-X-H32 regresses (body + hands + face).
NUM_SMPLX_JOINTS = 127
NUM_BODY_POSE_JOINTS = 21        # body_pose excludes the root (global_orient)


def _load_smplestx_model(cfg: SMPLestXConfig):
    """Load + cache the SMPLest-X model (env-gated; ``nfl_smplx``).

    The exact import path is the one documented in the SMPLer-X repo README; we
    add the checkout to ``sys.path`` and build its demo inferencer once. Raises
    a precise :class:`SetupError` if the expected entrypoint isn't found.
    """
    key = str(cfg.repo_dir)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    torch = _lazy_import()
    import sys

    sys.path.insert(0, str(cfg.repo_dir))
    try:
        # SMPLer-X exposes an Inferer/Demoer entrypoint; finalized per README.
        from main.inference import Inferer  # type: ignore
    except Exception as e:  # noqa: BLE001 — surface a setup-actionable message
        raise SetupError(
            f"could not import SMPLest-X inference from {cfg.repo_dir} ({e}). "
            "Confirm the checkout + entrypoint — see SETUP.md §8."
        ) from e
    model = Inferer(str(cfg.weights_path), device=cfg.device)
    if hasattr(model, "eval"):
        model.eval()
    _ = torch  # device/context already configured by the Inferer
    _MODEL_CACHE[key] = model
    return model


def _smplestx_forward(model, crops: np.ndarray, bboxes: np.ndarray,
                      cfg: SMPLestXConfig) -> list[dict[str, np.ndarray]]:
    """Run the model on crops → one raw param dict per sample. Seam: monkeypatched
    in tests so the schema assembly is exercised without torch/weights."""
    out: list[dict[str, np.ndarray]] = []
    for i in range(0, len(crops), cfg.batch_size):
        batch = crops[i:i + cfg.batch_size]
        boxes = bboxes[i:i + cfg.batch_size]
        out.extend(model.infer_batch(batch, boxes))   # repo returns per-sample dicts
    return out


def _assemble_smplestx_outputs(raw: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-sample raw dicts into the cache schema (leading ``N``). Pure."""
    n = len(raw)
    out = {
        "betas": np.zeros((n, 10), dtype=np.float32),
        "body_pose": np.zeros((n, NUM_BODY_POSE_JOINTS, 3), dtype=np.float32),
        "global_orient": np.zeros((n, 3), dtype=np.float32),
        "transl": np.zeros((n, 3), dtype=np.float32),
        "joints3d_cam": np.zeros((n, NUM_SMPLX_JOINTS, 3), dtype=np.float32),
        "joints2d": np.zeros((n, NUM_SMPLX_JOINTS, 2), dtype=np.float32),
        "confidence": np.zeros((n, NUM_SMPLX_JOINTS), dtype=np.float32),
    }
    for i, r in enumerate(raw):
        out["betas"][i] = np.asarray(r["betas"], dtype=np.float32).reshape(10)
        out["body_pose"][i] = np.asarray(r["body_pose"], dtype=np.float32).reshape(NUM_BODY_POSE_JOINTS, 3)
        out["global_orient"][i] = np.asarray(r["global_orient"], dtype=np.float32).reshape(3)
        out["transl"][i] = np.asarray(r["transl"], dtype=np.float32).reshape(3)
        out["joints3d_cam"][i] = np.asarray(r["joints3d_cam"], dtype=np.float32)
        out["joints2d"][i] = np.asarray(r["joints2d"], dtype=np.float32)
        out["confidence"][i] = np.asarray(r["confidence"], dtype=np.float32)
    return out


def infer_crops(
    crops: np.ndarray,       # [N, H, W, 3] uint8 RGB
    bboxes: np.ndarray,      # [N, 4] image-space (x1, y1, x2, y2)
    cfg: SMPLestXConfig,
) -> dict[str, np.ndarray]:
    """Run SMPLest-X-H32 on a batch of player crops.

    Returns a dict of stacked per-sample outputs (shapes with leading ``N``),
    matching the cache schema in this module's docstring. Heavy lifting is the
    external SMPLer-X model loaded inside the ``nfl_smplx`` env.
    """
    check_prerequisites(cfg)
    model = _load_smplestx_model(cfg)
    raw = _smplestx_forward(model, crops, bboxes, cfg)
    return _assemble_smplestx_outputs(raw)


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
