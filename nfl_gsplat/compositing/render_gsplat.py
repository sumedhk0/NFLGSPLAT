"""Drive ``gsplat.rasterization`` over a virtual camera trajectory.

This module is the only place ``gsplat`` is imported in the pipeline; all
other modules operate on the :class:`GaussianBatch` numpy dataclass. Torch
and gsplat are lazy-imported so CI (CPU-only) never touches them.

Output contract: ``render_trajectory`` writes ``{out_dir}/{frame:06d}.png``
and returns the list of written paths. The compositor script then calls
``nfl_gsplat.utils.video.encode_mp4`` to produce the final MP4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from nfl_gsplat.compositing.merge_ply import GaussianBatch
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class RenderConfig:
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane_m: float = 0.1
    far_plane_m: float = 200.0


def _lazy_imports():
    try:
        import torch  # type: ignore
        from gsplat import rasterization  # type: ignore
    except ImportError as e:
        raise SetupError(
            "gsplat / torch not installed — activate the `nfl_gsplat` conda env. "
            "See SETUP.md §1."
        ) from e
    return torch, rasterization


def _batch_to_torch(batch: GaussianBatch, device: str):
    import torch  # type: ignore
    xyz = torch.from_numpy(batch.xyz).to(device).float()
    rot = torch.from_numpy(batch.rot).to(device).float()
    # gsplat wants quaternion normalized.
    rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-12)
    scale = torch.from_numpy(np.exp(batch.scale)).to(device).float()
    opacity = torch.from_numpy(1.0 / (1.0 + np.exp(-batch.opacity))).to(device).float()
    # gsplat expects SH as [N, K, 3] (row-major in K, then channels).
    sh = torch.from_numpy(np.transpose(batch.sh, (0, 2, 1)).copy()).to(device).float()
    return xyz, rot, scale, opacity, sh


def render_trajectory(
    batch: GaussianBatch,
    intrinsics: CameraIntrinsics,
    poses: Sequence[CameraPose],
    out_dir: Path | str,
    cfg: RenderConfig,
    *,
    device: str = "cuda:0",
) -> list[Path]:
    """Render a virtual-camera trajectory and save PNGs.

    ``intrinsics`` is shared across all frames — the virtual camera uses a
    fixed intrinsic. ``poses`` is one :class:`CameraPose` per output frame.
    """
    torch, rasterization = _lazy_imports()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    xyz, rot, scale, opacity, sh = _batch_to_torch(batch, device)
    K_np = intrinsics.K()
    K_t = torch.from_numpy(K_np).to(device).float()

    written: list[Path] = []
    for i, pose in enumerate(poses):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = pose.R
        w2c[:3, 3] = pose.t
        viewmat = torch.from_numpy(w2c).to(device).float().unsqueeze(0)  # [1, 4, 4]
        Ks = K_t.unsqueeze(0)                                            # [1, 3, 3]

        render, _alpha, _info = rasterization(
            means=xyz, quats=rot, scales=scale, opacities=opacity,
            colors=sh, viewmats=viewmat, Ks=Ks,
            width=intrinsics.width, height=intrinsics.height,
            near_plane=cfg.near_plane_m, far_plane=cfg.far_plane_m,
            sh_degree=batch.sh_degree,
        )
        img = (render[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        out = out_dir / f"{i:06d}.png"
        _save_png(out, img)
        written.append(out)
    _LOG.info(f"rendered {len(written)} frames to {out_dir}")
    return written


def _save_png(path: Path, img: np.ndarray) -> None:
    import imageio.v3 as iio
    iio.imwrite(str(path), img)
