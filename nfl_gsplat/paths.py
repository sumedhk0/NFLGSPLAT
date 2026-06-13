"""Single source of truth for every on-disk artifact path.

Stages never hand-build paths; they construct a :class:`PlayDir` (from an
explicit play directory or from config) and read its attributes. Every play is
self-contained under ``data/{season}/week_NN/{away}_at_{home}/play_NNN/``; the
only cross-play state lives in three season-shared roots prefixed ``_``.

Layout::

    data/{season}/week_NN/{matchup}/play_NNN/sideline.mp4   video(cam)
    data/{season}/week_NN/{matchup}/play_NNN/endzone.mp4
    data/{season}/week_NN/{matchup}/play_NNN/cameras.json   cameras_json (per-play calib)
    data/{season}/week_NN/{matchup}/play_NNN/field.ply      field_ply    (per-play field)
    data/{season}/week_NN/{matchup}/play_NNN/tracks.parquet tracks
    data/{season}/week_NN/{matchup}/play_NNN/entities.json  entities
    data/{season}/week_NN/{matchup}/play_NNN/smplestx/      smplestx_dir
    data/{season}/week_NN/{matchup}/play_NNN/poses/{uid}.npz pose(uid)
    data/{season}/week_NN/{matchup}/play_NNN/ball.npz       ball
    data/{season}/week_NN/{matchup}/play_NNN/render.mp4     render_mp4
    data/{season}/week_NN/{matchup}/play_NNN/meta.yaml      meta_yaml
    data/{season}/_library/                                 library_root  (cross-play)
    data/{season}/_rosters/                                 rosters_root
    data/{season}/_registry.json                            registry_path

``matchup`` is ``"{away}_at_{home}"`` (NFL-standard). :attr:`PlayDir.teams`
returns ``(home, away)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _select(cfg: DictConfig, key: str, default):
    val = OmegaConf.select(cfg, key)
    return default if val is None else val


@dataclass(frozen=True)
class PlayDir:
    """Resolver for every artifact of one play, plus the season-shared roots."""

    season: str
    week: int
    matchup: str                 # "{away}_at_{home}"
    play_id: str                 # "play_001"
    data_root: Path = Path("data")
    cameras: tuple[str, ...] = ("sideline", "endzone")

    # --- the play folder + its artifacts -----------------------------------

    @property
    def season_root(self) -> Path:
        return self.data_root / str(self.season)

    @property
    def week_dir(self) -> Path:
        return self.season_root / f"week_{int(self.week):02d}"

    @property
    def matchup_dir(self) -> Path:
        return self.week_dir / self.matchup

    @property
    def dir(self) -> Path:
        return self.matchup_dir / self.play_id

    def video(self, cam: str) -> Path:
        return self.dir / f"{cam}.mp4"

    @property
    def cameras_json(self) -> Path:
        return self.dir / "cameras.json"

    @property
    def field_ply(self) -> Path:
        return self.dir / "field.ply"

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

    @property
    def meta_yaml(self) -> Path:
        return self.dir / "meta.yaml"

    # --- season-shared (cross-play) roots ----------------------------------

    @property
    def library_root(self) -> Path:
        return self.season_root / "_library"

    @property
    def rosters_root(self) -> Path:
        return self.season_root / "_rosters"

    @property
    def registry_path(self) -> Path:
        return self.season_root / "_registry.json"

    # --- derived metadata ---------------------------------------------------

    @property
    def teams(self) -> tuple[str, str]:
        """``(home, away)`` parsed from the matchup ``{away}_at_{home}``."""
        away, home = self.matchup.split("_at_")
        return home, away

    # --- constructors -------------------------------------------------------

    @classmethod
    def from_dir(cls, path, *, cameras: tuple[str, ...] = ("sideline", "endzone")) -> "PlayDir":
        """Build a :class:`PlayDir` from an existing play directory path.

        Expects ``.../{data_root}/{season}/week_NN/{matchup}/play_NNN``.
        """
        p = Path(path)
        play_id = p.name
        matchup = p.parent.name
        week_name = p.parent.parent.name
        if not week_name.startswith("week_"):
            raise ValueError(f"{path}: expected a week_NN folder, got {week_name!r}")
        week = int(week_name[len("week_"):])
        season = p.parent.parent.parent.name
        data_root = p.parent.parent.parent.parent
        return cls(season=season, week=week, matchup=matchup, play_id=play_id,
                   data_root=data_root, cameras=tuple(cameras))


def play_dir(cfg: DictConfig, season, week: int, matchup: str, play_id: str) -> PlayDir:
    """Construct a :class:`PlayDir` from config defaults (data root + cameras)."""
    cams = tuple(str(c) for c in _select(cfg, "cameras", ["sideline", "endzone"]))
    return PlayDir(
        season=str(season),
        week=int(week),
        matchup=str(matchup),
        play_id=str(play_id),
        data_root=Path(str(_select(cfg, "paths.data_root", "data"))),
        cameras=cams,
    )
