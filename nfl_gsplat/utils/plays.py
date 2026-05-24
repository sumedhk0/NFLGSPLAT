"""Per-game play manifest: frame windows + game metadata.

One ``plays.yaml`` lives beside each game's video at ``data/raw/{game}/plays.yaml``.
A game is one continuous sideline + endzone MP4 pair; plays are *time windows*
into them. This manifest is what slices the per-game video into per-play frame
ranges and supplies the season / home / away metadata the identity layer needs.

Schema::

    meta:
      season: 2024
      home_team: ATL
      away_team: "NO"      # quote abbreviations: YAML reads bare NO as boolean false
      fps: 30.0
    plays:
      play_001: {start_frame: 1200, end_frame: 1380, gsis_play_id: 36}
      play_002: {start_frame: 1500, end_frame: 1700}

``gsis_play_id`` is optional and only used for nflverse participation alignment
(Tier 2); without it the per-game roster is the candidate set.
"""
from __future__ import annotations

from dataclasses import dataclass

from omegaconf import OmegaConf

from nfl_gsplat.errors import SetupError


@dataclass(frozen=True)
class PlayWindow:
    play_id: str
    start_frame: int
    end_frame: int          # inclusive
    gsis_play_id: str | None = None

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class PlaysManifest:
    season: str
    home_team: str
    away_team: str
    fps: float
    plays: dict[str, PlayWindow]

    @property
    def game_teams(self) -> tuple[str, str]:
        return (self.home_team, self.away_team)

    def play_ids(self) -> list[str]:
        return list(self.plays.keys())

    def window(self, play_id: str) -> PlayWindow:
        if play_id not in self.plays:
            raise KeyError(f"play {play_id!r} not in manifest (have {self.play_ids()})")
        return self.plays[play_id]


def load_plays(path) -> PlaysManifest:
    """Load + validate a ``plays.yaml``. Raises :class:`SetupError` on missing
    file or malformed schema (per the project's fail-loud philosophy)."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"plays manifest missing at {path}. Create it with per-play frame "
            "windows (meta + plays). See SETUP.md §5."
        )
    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(raw, dict) or "meta" not in raw or "plays" not in raw:
        raise SetupError(f"{path}: expected top-level 'meta' and 'plays' keys.")

    meta = raw["meta"]
    for key in ("season", "home_team", "away_team"):
        if key not in meta:
            raise SetupError(f"{path}: meta.{key} is required.")
    # YAML 1.1 coerces NO / NA / ON / yes / off to booleans — a footgun for team
    # abbreviations like "NO" (New Orleans). Fail loud and tell the user to quote.
    for key in ("home_team", "away_team"):
        if isinstance(meta[key], bool):
            raise SetupError(
                f"{path}: meta.{key} parsed as a boolean — quote the abbreviation "
                f'(e.g. {key}: "NO") so YAML keeps it a string.'
            )

    plays: dict[str, PlayWindow] = {}
    for pid, spec in raw["plays"].items():
        if "start_frame" not in spec or "end_frame" not in spec:
            raise SetupError(f"{path}: play {pid} needs start_frame and end_frame.")
        start, end = int(spec["start_frame"]), int(spec["end_frame"])
        if end < start:
            raise SetupError(f"{path}: play {pid} end_frame {end} < start_frame {start}.")
        gsis = spec.get("gsis_play_id")
        plays[str(pid)] = PlayWindow(
            play_id=str(pid), start_frame=start, end_frame=end,
            gsis_play_id=str(gsis) if gsis is not None else None,
        )

    return PlaysManifest(
        season=str(meta["season"]),
        home_team=str(meta["home_team"]),
        away_team=str(meta["away_team"]),
        fps=float(meta.get("fps", 30.0)),
        plays=plays,
    )
