"""Per-play metadata (``meta.yaml``) — fps + teams + optional gsis play id.

One ``meta.yaml`` lives in each play folder
(``data/{season}/week_NN/{matchup}/play_NNN/meta.yaml``). Season/week/teams are
also encoded in the path, but this file is the authoritative record and carries
``fps`` and ``gsis_play_id``, which the path does not. Replaces the old
``plays.yaml`` frame-window manifest (plays are now standalone clips).

Schema::

    season: 2024
    week: 1
    home_team: ATL
    away_team: "NO"      # quote abbreviations: bare NO/ON/NA parse as booleans
    fps: 30.0
    gsis_play_id: 36     # optional; nflverse participation alignment only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from nfl_gsplat.errors import SetupError


@dataclass(frozen=True)
class CalibHint:
    ref_frame: int
    ref_x: float
    yard: int
    side: str          # home | away | mid
    increasing: str    # left | right (image direction yards increase)


@dataclass(frozen=True)
class PlayMeta:
    season: str
    week: int
    home_team: str
    away_team: str
    fps: float
    gsis_play_id: str | None = None
    calib_hints: dict[str, CalibHint] = field(default_factory=dict)

    @property
    def game_teams(self) -> tuple[str, str]:
        return (self.home_team, self.away_team)


def load_meta(path) -> PlayMeta:
    """Load + validate a play's ``meta.yaml`` (fail-loud per project philosophy)."""
    path = Path(path)
    if not path.exists():
        raise SetupError(
            f"play meta.yaml missing at {path}. Create it (season/week/home_team/"
            "away_team/fps) — see SETUP.md §5. Use scripts/new_play.py to scaffold one."
        )
    raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(raw, dict):
        raise SetupError(f"{path}: expected a mapping of meta fields.")
    for key in ("season", "week", "home_team", "away_team"):
        if key not in raw:
            raise SetupError(f"{path}: meta.{key} is required.")
    # YAML 1.1 coerces NO / NA / ON / yes / off to booleans — a footgun for team
    # abbreviations like "NO" (New Orleans). Fail loud and tell the user to quote.
    for key in ("home_team", "away_team"):
        if isinstance(raw[key], bool):
            raise SetupError(
                f"{path}: meta.{key} parsed as a boolean — quote the abbreviation "
                f'(e.g. {key}: "NO") so YAML keeps it a string.'
            )
    gsis = raw.get("gsis_play_id")
    hints: dict[str, CalibHint] = {}
    raw_hints = raw.get("calib_hints") or {}
    for cam, h in raw_hints.items():
        side = str(h["side"])
        inc = str(h["increasing"])
        yard = int(h["yard"])
        if side not in ("home", "away", "mid"):
            raise SetupError(f"{path}: calib_hints.{cam}.side must be home/away/mid, got {side!r}.")
        if inc not in ("left", "right"):
            raise SetupError(f"{path}: calib_hints.{cam}.increasing must be left/right, got {inc!r}.")
        if side == "mid":
            yard = 50
        elif yard < 5 or yard > 45 or yard % 5 != 0:
            raise SetupError(f"{path}: calib_hints.{cam}.yard {yard} invalid (5..45 step 5, or mid=50).")
        hints[str(cam)] = CalibHint(
            ref_frame=int(h["ref_frame"]), ref_x=float(h["ref_x"]),
            yard=yard, side=side, increasing=inc,
        )
    return PlayMeta(
        season=str(raw["season"]),
        week=int(raw["week"]),
        home_team=str(raw["home_team"]),
        away_team=str(raw["away_team"]),
        fps=float(raw.get("fps", 30.0)),
        gsis_play_id=str(gsis) if gsis is not None else None,
        calib_hints=hints,
    )
