"""Single source of truth for every on-disk artifact path.

Stages never hand-build paths; they call :func:`resolve_paths` and read the
attributes. This keeps the per-game / per-play layout consistent across the
whole pipeline and makes the season DAG (which scans many plays) trivial.

Layout::

    data/raw/{game}/{sideline,endzone}.mp4         raw_video(cam)
    data/raw/{game}/plays.yaml                      plays_yaml
    data/annotations/{game}/{cam}_landmarks.json    annotations(cam)
    outputs/{game}/calib/cameras.json               calib_json
    outputs/{game}/field/field.ply                  field_ply
    outputs/{game}/{play}/tracks.parquet            tracks
    outputs/{game}/{play}/entities.json             entities
    outputs/{game}/{play}/smplestx/                 smplestx_dir
    outputs/{game}/{play}/poses/{uid}.npz           pose(uid)
    outputs/{game}/{play}/ball.npz                  ball
    outputs/{game}/{play}/render.mp4                render_mp4
    library/{season}/                               library_root
    data/rosters/{season}/                          rosters_dir
    outputs/{season}/identity/registry.json         (via identity.registry)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _select(cfg: DictConfig, key: str, default):
    val = OmegaConf.select(cfg, key)
    return default if val is None else val


@dataclass(frozen=True)
class GamePaths:
    raw_root: Path
    outputs_root: Path
    library_root: Path
    rosters_root: Path
    cameras: tuple[str, ...]
    game_id: str
    season: str

    @property
    def raw_dir(self) -> Path:
        return self.raw_root / self.game_id

    def raw_video(self, cam: str) -> Path:
        return self.raw_dir / f"{cam}.mp4"

    @property
    def plays_yaml(self) -> Path:
        return self.raw_dir / "plays.yaml"

    def annotations(self, cam: str) -> Path:
        return Path("data/annotations") / self.game_id / f"{cam}_landmarks.json"

    @property
    def game_out(self) -> Path:
        return self.outputs_root / self.game_id

    @property
    def calib_json(self) -> Path:
        return self.game_out / "calib" / "cameras.json"

    @property
    def field_ply(self) -> Path:
        return self.game_out / "field" / "field.ply"

    @property
    def library_dir(self) -> Path:
        return self.library_root / self.season

    @property
    def rosters_dir(self) -> Path:
        return self.rosters_root / self.season


@dataclass(frozen=True)
class PlayPaths:
    game: GamePaths
    play_id: str

    @property
    def dir(self) -> Path:
        return self.game.game_out / self.play_id

    @property
    def tracks(self) -> Path:
        return self.dir / "tracks.parquet"

    @property
    def entities(self) -> Path:
        return self.dir / "entities.json"

    @property
    def smplestx_dir(self) -> Path:
        return self.dir / "smplestx"

    @property
    def joints3d(self) -> Path:
        return self.dir / "joints3d.npz"

    @property
    def poses_dir(self) -> Path:
        return self.dir / "poses"

    def pose(self, uid: str) -> Path:
        return self.poses_dir / f"{uid}.npz"

    @property
    def ball(self) -> Path:
        return self.dir / "ball.npz"

    @property
    def render_mp4(self) -> Path:
        return self.dir / "render.mp4"


def game_paths(cfg: DictConfig, game_id: str) -> GamePaths:
    cams = tuple(str(c) for c in _select(cfg, "cameras", ["sideline", "endzone"]))
    return GamePaths(
        raw_root=Path(str(_select(cfg, "paths.raw_video", "data/raw"))),
        outputs_root=Path(str(_select(cfg, "paths.outputs", "outputs"))),
        library_root=Path(str(_select(cfg, "avatars.library.path", "library"))),
        rosters_root=Path("data/rosters"),
        cameras=cams,
        game_id=game_id,
        season=str(_select(cfg, "identity.season", "0")),
    )


def play_paths(cfg: DictConfig, game_id: str, play_id: str) -> PlayPaths:
    return PlayPaths(game=game_paths(cfg, game_id), play_id=play_id)
