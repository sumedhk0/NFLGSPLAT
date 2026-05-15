"""Render a novel-view MP4 for a processed play.

Reads the per-play artifacts, merges them into a single Gaussian batch,
drives ``gsplat.rasterization`` over a virtual-camera trajectory, and
encodes the resulting PNG sequence to MP4.

Usage::

    python scripts/05_render_novel_view.py --game game_001 --play play_001 \
        --trajectory configs/trajectories/fly_through.yaml
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from nfl_gsplat.avatars.lbs_animate import animate_gaussians
from nfl_gsplat.compositing.merge_ply import (
    GaussianBatch,
    load_gaussian_ply,
    merge_batches,
)
from nfl_gsplat.compositing.render_gsplat import RenderConfig, render_trajectory
from nfl_gsplat.compositing.trajectory import sample_trajectory
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.io import read_npz
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.video import encode_mp4

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _animate_player_batch(avatar_npz: Path, joint_tfms: np.ndarray) -> GaussianBatch:
    """Apply LBS to a canonical avatar for one frame.

    ``joint_tfms [J, 4, 4]`` is the per-joint canonical→world transform."""
    d = read_npz(avatar_npz)
    xyz_w, rot_w = animate_gaussians(
        d["canonical_xyz"], d["canonical_rot"], d["lbs_weights"], joint_tfms
    )
    return GaussianBatch(
        xyz=xyz_w.astype(np.float32),
        rot=rot_w.astype(np.float32),
        scale=d["canonical_scale"].astype(np.float32),
        opacity=d["canonical_opacity"].astype(np.float32),
        sh=d["canonical_sh"].astype(np.float32),
        sh_degree=int(round(np.sqrt(d["canonical_sh"].shape[-1])) - 1),
    )


@app.command()
def main(
    game: str = typer.Option(...),
    play: str = typer.Option(...),
    trajectory: Path = typer.Option(...),
    out_root: Path = typer.Option(Path("outputs")),
    device: str = typer.Option("cuda:0"),
) -> None:
    play_dir = out_root / game / play
    field_ply = out_root / game / "field" / "field.ply"
    if not field_ply.exists():
        raise SetupError(f"field.ply missing at {field_ply}; run 03_reconstruct_field.sh first.")

    field_batch = load_gaussian_ply(field_ply)
    intr, poses = sample_trajectory(trajectory)
    num_frames = len(poses)

    avatar_dir = play_dir / "avatars"
    avatars = sorted(avatar_dir.glob("*.npz"))
    if not avatars:
        _LOG.warning(f"no avatars under {avatar_dir}; rendering field only")

    # Per-frame joint transforms from fused SMPL-X poses. Contract is one NPZ
    # per global_player_id under play_dir/poses/, keys {joint_tfms [T, J, 4, 4]}.
    pose_dir = play_dir / "poses"
    pose_tfms: dict[str, np.ndarray] = {}
    for avatar in avatars:
        gpid = avatar.stem
        p = pose_dir / f"{gpid}.npz"
        if not p.exists():
            _LOG.warning(f"no poses for player {gpid}; skipping avatar")
            continue
        pose_tfms[gpid] = read_npz(p)["joint_tfms"]

    frames_dir = play_dir / "render" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for t in range(num_frames):
        animated = []
        for gpid, tfms in pose_tfms.items():
            animated.append(_animate_player_batch(avatar_dir / f"{gpid}.npz", tfms[t]))
        merged = merge_batches([field_batch, *animated])
        render_trajectory(
            merged, intr, poses=[poses[t]],
            out_dir=frames_dir / f"t{t:06d}",
            cfg=RenderConfig(), device=device,
        )

    final_mp4 = play_dir / "render.mp4"
    encode_mp4(frames_dir, final_mp4, fps=30.0)
    _LOG.info(f"wrote {final_mp4}")


if __name__ == "__main__":
    app()
