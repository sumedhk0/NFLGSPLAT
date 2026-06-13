"""SMPLest-X-H32 per-camera per-frame inference.

Lazy torch / SMPLest-X import so this module is safe to import from CPU envs
(tests, tracking). Real inference requires the ``nfl_smplx`` conda env plus
the SMPLest-X repo checkout + pretrained weights.

Output cache schema (one NPZ per ``(cam, frame, global_player_id)``)::

    betas           [10]
    body_pose       [21, 3]       axis-angle per body joint
    global_orient   [3]
    transl          [3]
    joints3d_cam    [J, 3]        SMPL-X all-joints in camera coords
    joints2d        [J, 2]        pixel coords in the *original frame*
    confidence      [J]           see note below

Only the first 22 joints are used by triangulation; the rest (hands/face)
are cached for future use. ``J`` is whatever the loaded SMPLest-X model
regresses (read from the model output, not hard-coded).

CONFIDENCE NOTE: SMPLest-X is a *regressor* — it emits SMPL-X parameters and
re-projected joints directly, with no per-joint heatmap confidence. We
therefore synthesize ``confidence = 1.0`` for every joint. Downstream
triangulation/fusion weights views by this confidence, so all views from
SMPLest-X are weighted equally (we cannot down-weight occluded joints from
SMPLest-X alone). If per-joint reliability is needed later, derive it from the
detector/track confidence and inject it here.

The real model glue (``_load_smplestx_model`` / ``_smplestx_forward``) mirrors
``third_party/SMPLest-X/main/inference.py`` (Config → Tester → model forward).
Both are monkeypatched in the CPU contract test so the schema assembly is
exercised without torch or weights.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class SMPLestXConfig:
    repo_dir: Path = Path("third_party/SMPLest-X")
    # The HF release lays weights + config out as
    #   pretrained_models/<ckpt_name>/<ckpt_name>.pth.tar
    #   pretrained_models/<ckpt_name>/config_base.py
    ckpt_name: str = "smplest_x_h"
    device: str = "cuda:0"
    batch_size: int = 4

    @property
    def model_dir(self) -> Path:
        return self.repo_dir / "pretrained_models" / self.ckpt_name

    @property
    def weights_path(self) -> Path:
        return self.model_dir / f"{self.ckpt_name}.pth.tar"

    @property
    def config_path(self) -> Path:
        return self.model_dir / "config_base.py"


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
    """Raise :class:`SetupError` if model code, weights, or config are missing.

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
            "Download the pretrained model into pretrained_models/"
            f"{cfg.ckpt_name}/ — see SETUP.md §4."
        )
    if not cfg.config_path.exists():
        raise SetupError(
            f"SMPLest-X config missing at {cfg.config_path}. It ships with the "
            "pretrained-model release alongside the .pth.tar — see SETUP.md §4."
        )


_MODEL_CACHE: dict = {}

# Default body-joint counts (the root pose is global_orient, excluded from
# body_pose). J (total regressed joints) is read from the model output.
NUM_SMPLX_JOINTS = 127           # documented default; assembly uses the real J
NUM_BODY_POSE_JOINTS = 21        # body_pose excludes the root (global_orient)


@contextlib.contextmanager
def _chdir(path: Path):
    """Temporarily chdir — SMPLest-X's config/human-model paths are repo-relative."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _load_smplestx_model(cfg: SMPLestXConfig):
    """Load + cache the SMPLest-X model (env-gated; ``nfl_smplx``).

    Mirrors ``third_party/SMPLest-X/main/inference.py``: load the bundled
    ``config_base.py``, point it at the checkpoint, build a ``Tester`` and call
    ``_make_model`` (which loads weights + ``.eval()``). The repo resolves
    ``./pretrained_models`` and human-model files relative to its own root, so
    the build runs with the cwd set to ``repo_dir``.
    """
    key = str(cfg.repo_dir)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    _lazy_import()
    import sys
    import tempfile

    repo = cfg.repo_dir.resolve()
    sys.path.insert(0, str(repo))
    try:
        from main.base import Tester          # type: ignore
        from main.config import Config        # type: ignore
    except Exception as e:  # noqa: BLE001 — surface a setup-actionable message
        raise SetupError(
            f"could not import SMPLest-X (main.config / main.base) from {repo} "
            f"({e}). Confirm the checkout — see SETUP.md §8."
        ) from e

    log_dir = Path(tempfile.mkdtemp(prefix="smplestx_log_"))
    with _chdir(repo):
        smplestx_cfg = Config.load_config(str(cfg.config_path))
        smplestx_cfg.update_config({
            "model": {"pretrained_model_path": str(cfg.weights_path.resolve())},
            "log": {"exp_name": "nfl_infer", "log_dir": str(log_dir)},
        })
        smplestx_cfg.prepare_log()
        tester = Tester(smplestx_cfg)
        tester._make_model()      # builds DataParallel model, loads ckpt, .eval()

    model = _SMPLestXRunner(tester, smplestx_cfg)
    _MODEL_CACHE[key] = model
    return model


class _SMPLestXRunner:
    """Thin holder bundling the Tester, its cfg, and the input patch shape.

    Keeps everything ``_smplestx_forward`` needs in one place and matches the
    object the contract test monkeypatches in for ``_load_smplestx_model``.
    """

    def __init__(self, tester, smplestx_cfg):
        self.tester = tester
        self.cfg = smplestx_cfg
        # input_img_shape is (H, W) of the network patch, e.g. (512, 384).
        self.input_img_shape = tuple(smplestx_cfg.model.input_img_shape)
        self.bbox_ratio = getattr(getattr(smplestx_cfg, "data", None), "bbox_ratio", 1.25)


def _apply_affine(pts_xy: np.ndarray, trans_2x3: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to [N, 2] points. Used to map patch-space joint
    projections back to the source-crop pixel frame via ``inv_trans``."""
    pts = np.asarray(pts_xy, dtype=np.float64)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)   # [N, 3]
    return (homog @ np.asarray(trans_2x3, dtype=np.float64).T).astype(np.float32)


def _smplestx_forward(model, crops: np.ndarray, bboxes: np.ndarray,
                      cfg: SMPLestXConfig) -> list[dict[str, np.ndarray]]:
    """Run the model on crops → one raw param dict per sample.

    Each crop is treated as its own image: we re-derive the network patch with
    the repo's ``process_bbox`` + ``generate_patch_image`` (so inputs match the
    training distribution), run the forward in ``'test'`` mode, then map the
    re-projected joints from patch space back to the *original frame* using the
    patch ``inv_trans`` plus the crop's offset in ``bboxes`` (x1, y1).

    Seam: monkeypatched in tests so the schema assembly runs without torch.
    """
    torch = _lazy_import()
    import sys
    import torchvision.transforms as T  # type: ignore

    sys.path.insert(0, str(cfg.repo_dir.resolve()))
    from utils.data_utils import generate_patch_image, process_bbox  # type: ignore

    runner: _SMPLestXRunner = model
    net = runner.tester.model
    in_h, in_w = runner.input_img_shape
    to_tensor = T.ToTensor()

    results: list[dict[str, np.ndarray]] = []
    for start in range(0, len(crops), cfg.batch_size):
        batch_crops = crops[start:start + cfg.batch_size]
        batch_boxes = bboxes[start:start + cfg.batch_size]

        patches = []
        inv_transes = []
        for crop in batch_crops:
            h, w = crop.shape[:2]
            full_box = np.array([0.0, 0.0, w, h], dtype=np.float32)  # xywh of whole crop
            pbox = process_bbox(full_box, w, h, runner.input_img_shape,
                                ratio=runner.bbox_ratio)
            patch, _trans, inv_trans = generate_patch_image(
                cvimg=crop, bbox=pbox, scale=1.0, rot=0.0, do_flip=False,
                out_shape=runner.input_img_shape,
            )
            patches.append(to_tensor(patch.astype(np.float32)) / 255.0)
            inv_transes.append(inv_trans)

        img = torch.stack(patches, dim=0).to(cfg.device)
        with torch.no_grad():
            out = net({"img": img}, {}, {}, "test")

        root_pose = out["smplx_root_pose"].detach().cpu().numpy()
        body_pose = out["smplx_body_pose"].detach().cpu().numpy()
        shape = out["smplx_shape"].detach().cpu().numpy()
        cam_trans = out["cam_trans"].detach().cpu().numpy()
        joint_proj = out["smplx_joint_proj"].detach().cpu().numpy()   # patch space
        joint_cam = out["smplx_joint_cam"].detach().cpu().numpy()

        for i in range(len(batch_crops)):
            n_joints = joint_cam[i].shape[0]
            # joint_proj is in network-patch pixels; inv_trans -> crop pixels,
            # then add the crop's top-left in the full frame.
            proj_patch = joint_proj[i][:, :2] * np.array(
                [in_w, in_h], dtype=np.float32) if joint_proj[i].max() <= 1.5 else joint_proj[i][:, :2]
            crop_xy = _apply_affine(proj_patch, inv_transes[i])
            x1, y1 = float(batch_boxes[i][0]), float(batch_boxes[i][1])
            joints2d = crop_xy + np.array([x1, y1], dtype=np.float32)
            results.append({
                "betas": shape[i][:10],
                "body_pose": body_pose[i].reshape(NUM_BODY_POSE_JOINTS, 3),
                "global_orient": root_pose[i].reshape(3),
                "transl": cam_trans[i].reshape(3),
                "joints3d_cam": joint_cam[i],
                "joints2d": joints2d,
                # Regressor has no per-joint confidence — see module docstring.
                "confidence": np.ones(n_joints, dtype=np.float32),
            })
    return results


def _assemble_smplestx_outputs(raw: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-sample raw dicts into the cache schema (leading ``N``). Pure.

    The total joint count ``J`` is read from the samples rather than hard-coded,
    so this works for whichever SMPLest-X variant produced ``raw``.
    """
    n = len(raw)
    j = int(np.asarray(raw[0]["joints3d_cam"]).shape[0]) if n else NUM_SMPLX_JOINTS
    out = {
        "betas": np.zeros((n, 10), dtype=np.float32),
        "body_pose": np.zeros((n, NUM_BODY_POSE_JOINTS, 3), dtype=np.float32),
        "global_orient": np.zeros((n, 3), dtype=np.float32),
        "transl": np.zeros((n, 3), dtype=np.float32),
        "joints3d_cam": np.zeros((n, j, 3), dtype=np.float32),
        "joints2d": np.zeros((n, j, 2), dtype=np.float32),
        "confidence": np.zeros((n, j), dtype=np.float32),
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

    ``bboxes`` carry each crop's location in the original frame; the re-projected
    joints (``joints2d``) are returned in that original-frame pixel space.

    Returns a dict of stacked per-sample outputs (shapes with leading ``N``),
    matching the cache schema in this module's docstring. Heavy lifting is the
    external SMPLest-X model loaded inside the ``nfl_smplx`` env.
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
