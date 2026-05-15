"""Train a static 3DGS field via nerfstudio's ``splatfacto``, then export PLY.

Nerfstudio is heavyweight and only lives in the ``nfl_gsplat`` conda env. We
shell out to ``ns-train`` and ``ns-export`` rather than importing nerfstudio
in-process so the rest of the pipeline (tracking, pose fusion) can run in the
``smplx`` or ``avatar`` envs without nerfstudio installed.

For CPU-only CI, :func:`write_mock_field_ply` emits a synthetic field-shaped
Gaussian blob that satisfies the smoke-test assertions (>50 k primitives,
non-empty PLY, non-black render) without any GPU work.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.field_landmarks import FIELD_LENGTH_M, FIELD_WIDTH_M
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class FieldTrainConfig:
    splatfacto_iters: int = 30000
    sh_degree: int = 3
    cull_alpha_thresh: float = 0.1
    background_color: str = "black"


def train_field(
    transforms_json: Path | str,
    out_dir: Path | str,
    cfg: FieldTrainConfig,
    *,
    experiment_name: str = "field",
) -> Path:
    """Run ``ns-train splatfacto`` and export a ``field.ply``.

    Requires ``ns-train`` and ``ns-export`` on PATH (nerfstudio env). Raises
    :class:`SetupError` if either binary is missing.

    Returns the written ``field.ply`` path.
    """
    if shutil.which("ns-train") is None:
        raise SetupError(
            "ns-train not found — activate the `nfl_gsplat` conda env "
            "(`conda activate nfl_gsplat`). See SETUP.md §1."
        )
    if shutil.which("ns-export") is None:
        raise SetupError(
            "ns-export not found — activate the `nfl_gsplat` conda env. See SETUP.md §1."
        )

    transforms_json = Path(transforms_json)
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "nerfstudio_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir = transforms_json.parent
    cmd_train = [
        "ns-train", "splatfacto",
        "--data", str(data_dir),
        "--max-num-iterations", str(cfg.splatfacto_iters),
        "--output-dir", str(ckpt_dir),
        "--experiment-name", experiment_name,
        "--pipeline.model.sh-degree", str(cfg.sh_degree),
        "--pipeline.model.cull-alpha-thresh", str(cfg.cull_alpha_thresh),
        "--pipeline.model.background-color", cfg.background_color,
        "--viewer.quit-on-train-completion", "True",
        # Our poses are already metric + scene-centered, so don't let nerfstudio
        # rescale or re-center.
        "nerfstudio-data",
        "--auto-scale-poses", "False",
        "--center-method", "none",
        "--orientation-method", "none",
    ]
    _LOG.info("launching ns-train splatfacto  ({} iters)".format(cfg.splatfacto_iters))
    subprocess.check_call(cmd_train)

    # ns-train writes configs under {ckpt_dir}/{experiment_name}/splatfacto/<timestamp>/config.yml
    config_candidates = sorted(
        (ckpt_dir / experiment_name / "splatfacto").glob("*/config.yml"),
        key=lambda p: p.stat().st_mtime,
    )
    if not config_candidates:
        raise RuntimeError(
            f"ns-train produced no config.yml under {ckpt_dir} — training likely failed"
        )
    config = config_candidates[-1]

    ply_out = out_dir / "field.ply"
    cmd_export = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config),
        "--output-dir", str(out_dir),
    ]
    _LOG.info("exporting Gaussian PLY → {}".format(ply_out))
    subprocess.check_call(cmd_export)
    # ns-export writes "splat.ply" by default; rename if needed.
    default_ply = out_dir / "splat.ply"
    if default_ply.exists() and not ply_out.exists():
        default_ply.rename(ply_out)
    if not ply_out.exists():
        raise RuntimeError(f"expected field.ply at {ply_out}, not found")
    return ply_out


# --- Mock PLY (CPU-only smoke test) ----------------------------------------

def write_mock_field_ply(
    out_path: Path | str,
    *,
    num_gaussians: int = 60_000,
    seed: int = 0,
) -> Path:
    """Write a minimal 3DGS PLY shaped like the NFL field for CI smoke tests.

    Gaussians are scattered on the ``Z=0`` plane within the field rectangle,
    with near-white DC color and small isotropic scales. Enough structure to
    satisfy the smoke-test assertions (> 50 k primitives, render non-black).

    Only SH degree 0 is written (3 DC coefficients). Loaders that expect
    higher-degree SH will see ``sh_degree == 0`` and fall back cleanly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    half_L = FIELD_LENGTH_M / 2.0
    half_W = FIELD_WIDTH_M / 2.0
    xyz = np.column_stack([
        rng.uniform(-half_L, +half_L, num_gaussians),
        rng.uniform(-half_W, +half_W, num_gaussians),
        rng.normal(0.0, 0.02, num_gaussians),       # hug the ground plane
    ]).astype(np.float32)

    # Near-white turf with mild variation; convert sRGB [0,1] to SH DC via
    # (C - 0.5) / 0.28209479.
    base_rgb = np.stack([
        rng.uniform(0.05, 0.20, num_gaussians),     # R
        rng.uniform(0.40, 0.65, num_gaussians),     # G
        rng.uniform(0.05, 0.20, num_gaussians),     # B
    ], axis=1).astype(np.float32)
    f_dc = ((base_rgb - 0.5) / 0.28209479).astype(np.float32)

    opacity_logit = np.full((num_gaussians,), 2.0, dtype=np.float32)       # sigmoid(2)=0.88
    scale_log = np.full((num_gaussians, 3), np.log(0.05), dtype=np.float32)  # 5 cm
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_gaussians, 1))
    normals = np.zeros((num_gaussians, 3), dtype=np.float32)

    header_fields = [
        ("x", "f"), ("y", "f"), ("z", "f"),
        ("nx", "f"), ("ny", "f"), ("nz", "f"),
        ("f_dc_0", "f"), ("f_dc_1", "f"), ("f_dc_2", "f"),
        ("opacity", "f"),
        ("scale_0", "f"), ("scale_1", "f"), ("scale_2", "f"),
        ("rot_0", "f"), ("rot_1", "f"), ("rot_2", "f"), ("rot_3", "f"),
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {num_gaussians}\n"
        + "".join(f"property float {name}\n" for name, _ in header_fields)
        + "end_header\n"
    ).encode("ascii")

    rows = np.empty((num_gaussians, len(header_fields)), dtype=np.float32)
    rows[:, 0:3] = xyz
    rows[:, 3:6] = normals
    rows[:, 6:9] = f_dc
    rows[:, 9] = opacity_logit
    rows[:, 10:13] = scale_log
    rows[:, 13:17] = quat

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(rows.tobytes(order="C"))
    _LOG.info(f"wrote mock field PLY with {num_gaussians} Gaussians → {out_path}")
    return out_path


def read_ply_gaussian_count(path: Path | str) -> int:
    """Parse a PLY header and return the vertex/Gaussian count.

    Just enough parser to sanity-check smoke-test outputs — does not load
    the vertex payload.
    """
    with open(path, "rb") as f:
        line = f.readline().strip()
        if line != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        count: int | None = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            line = line.strip()
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            if line == b"end_header":
                break
        if count is None:
            raise ValueError(f"{path}: no 'element vertex' line in header")
        return count
