"""Render a novel-view MP4 for a processed play.

Reads the per-play artifacts, merges them into a single Gaussian batch,
drives ``gsplat.rasterization`` over a virtual-camera trajectory, and
encodes the resulting PNG sequence to MP4.

Entities are resolved through the season avatar/shape library: players load
their cached canonical avatar by ``player_uid``, referees share the generic
striped-shirt asset, and the football is the canonical asset oriented along the
Kalman velocity each frame. When ``play_dir/entities.json`` is absent we fall
back to the legacy layout (one avatar NPZ per player under ``play_dir/avatars``).

Usage::

    python scripts/05_render_novel_view.py \
        --play-dir data/2024/week_01/NO_at_ATL/play_001 \
        --trajectory configs/trajectories/fly_through.yaml
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.compositing.merge_ply import GaussianBatch, load_gaussian_ply
from nfl_gsplat.compositing.render_gsplat import RenderConfig, render_trajectory
from nfl_gsplat.compositing.scene import (
    compose_frame,
    football_batch,
    posed_avatar_batch,
)
from nfl_gsplat.compositing.trajectory import sample_trajectory
from nfl_gsplat.errors import SetupError
from nfl_gsplat.identity.registry import REFEREE_UID, EntityType
from nfl_gsplat.utils.io import read_json, read_npz
from nfl_gsplat.utils.logging import get_logger
from nfl_gsplat.utils.video import encode_mp4

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _load_entities(play_dir: Path, library: AvatarLibrary) -> list[tuple[dict, np.ndarray]]:
    """Return ``[(avatar_dict, joint_tfms [T, J, 4, 4]), ...]`` for the play.

    Avatar source by entity type: player → library by ``player_uid``; referee →
    the generic library asset. Poses live in ``play_dir/poses/{key}.npz`` keyed
    by the entity's uid (or, in legacy mode, the global_player_id).
    """
    pose_dir = play_dir / "poses"
    entities_json = play_dir / "entities.json"
    out: list[tuple[dict, np.ndarray]] = []

    if entities_json.exists():
        ref_avatar = library.get_referee_avatar() if library.has_referee_avatar() else None
        for ent in read_json(entities_json):
            uid, etype = ent["player_uid"], ent["entity_type"]
            # instance_id keys the per-instance pose; player_uid keys the avatar
            # (multiple referees share __referee__ but pose independently).
            instance_id = ent.get("instance_id", uid)
            pose_path = pose_dir / f"{instance_id}.npz"
            if etype == EntityType.OTHER.value or not uid or not pose_path.exists():
                continue
            if etype == EntityType.REFEREE.value or uid == REFEREE_UID:
                if ref_avatar is None:
                    _LOG.warning("referee entity but no generic referee avatar; skipping")
                    continue
                avatar = ref_avatar
            else:
                if not library.has_avatar(uid):
                    _LOG.warning(f"no cached avatar for player {uid}; skipping")
                    continue
                avatar = library.get_avatar(uid)
            out.append((avatar, read_npz(pose_path)["joint_tfms"]))
        return out

    # Legacy fallback: avatar NPZ per player under play_dir/avatars/.
    for avatar_npz in sorted((play_dir / "avatars").glob("*.npz")):
        key = avatar_npz.stem
        pose_path = pose_dir / f"{key}.npz"
        if not pose_path.exists():
            _LOG.warning(f"no poses for {key}; skipping avatar")
            continue
        out.append((read_npz(avatar_npz), read_npz(pose_path)["joint_tfms"]))
    return out


def _load_ball(play_dir: Path, library: AvatarLibrary) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Return ``(football_asset, xyz [T, 3], vel [T, 3])`` if a ball track and a
    canonical football asset are both available, else None."""
    ball_npz = play_dir / "ball.npz"
    if not ball_npz.exists() or not library.has_football_asset():
        return None
    d = read_npz(ball_npz)
    if "xyz" not in d or "vel" not in d:
        return None
    return library.get_football_asset(), d["xyz"], d["vel"]


@app.command()
def main(
    play_dir: Path = typer.Option(..., "--play-dir"),
    trajectory: Path = typer.Option(...),
    spin_rate: float = typer.Option(6.0),
    device: str = typer.Option("cuda:0"),
) -> None:
    pd = PlayDir.from_dir(play_dir)
    field_ply = pd.field_ply
    if not field_ply.exists():
        raise SetupError(f"field.ply missing at {field_ply}; run 03_reconstruct_field.sh first.")

    field_batch: GaussianBatch = load_gaussian_ply(field_ply)
    intr, poses = sample_trajectory(trajectory)
    num_frames = len(poses)

    library = AvatarLibrary(root=pd.library_root, season="")
    entities = _load_entities(pd.dir, library)
    if not entities:
        _LOG.warning(f"no posed entities for {pd.dir}; rendering field only")
    ball = _load_ball(pd.dir, library)

    frames_dir = pd.dir / "render" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for t in range(num_frames):
        posed = [posed_avatar_batch(avatar, tfms[t]) for avatar, tfms in entities]
        ball_batch = None
        if ball is not None:
            asset, xyz, vel = ball
            if t < len(xyz) and np.isfinite(xyz[t]).all():
                ball_batch = football_batch(asset, xyz[t], vel[t], t=t / 30.0, spin_rate=spin_rate)
        merged = compose_frame(field_batch, posed, ball_batch)
        render_trajectory(
            merged, intr, poses=[poses[t]],
            out_dir=frames_dir / f"t{t:06d}",
            cfg=RenderConfig(), device=device,
        )

    encode_mp4(frames_dir, pd.render_mp4, fps=30.0)
    _LOG.info(f"wrote {pd.render_mp4}")


if __name__ == "__main__":
    app()
