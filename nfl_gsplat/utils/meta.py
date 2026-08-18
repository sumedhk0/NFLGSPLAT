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
    endzone_prior:        # optional; required for --mode identity-endzone
      x_range: [-150, -60]
      y_range: [-15, 15]
      z_range: [10, 60]
      focal_range: [1500, 3500]
    endzone_anchor:        # optional; read off the mosaic diag PNG (--mode mosaic-endzone)
      lines:
        - {point_px: [120, 340], world_x_m: -18.288}
        - {point_px: [520, 340], world_x_m: 0.0}
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
    endzone_prior: dict | None = None
    endzone_anchor: dict | None = None

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
    endzone_prior: dict | None = None
    raw_ep = raw.get("endzone_prior")
    if raw_ep is not None:
        endzone_prior = {}
        for key in ("x_range", "y_range", "z_range", "focal_range"):
            if key not in raw_ep:
                raise SetupError(f"{path}: endzone_prior.{key} is required.")
            vals = list(raw_ep[key])
            if len(vals) != 2:
                raise SetupError(
                    f"{path}: endzone_prior.{key} must be a 2-element [min, max] "
                    f"list, got {vals!r}."
                )
            endzone_prior[key] = [float(v) for v in vals]
    endzone_anchor: dict | None = None
    raw_ea = raw.get("endzone_anchor")
    if raw_ea is not None:
        lines = raw_ea.get("lines") if isinstance(raw_ea, dict) else None
        if lines is None:
            raise SetupError(
                f"{path}: endzone_anchor.lines is required (a list of exactly "
                "two {point_px, world_x_m} entries naming the outermost yard lines)."
            )
        if len(lines) != 2:
            raise SetupError(
                f"{path}: endzone_anchor.lines must have exactly TWO entries "
                f"(the outermost detected yard lines), got {len(lines)}."
            )
        parsed_lines = []
        for entry in lines:
            if "point_px" not in entry or "world_x_m" not in entry:
                raise SetupError(
                    f"{path}: each endzone_anchor.lines entry needs point_px "
                    "and world_x_m."
                )
            pt = list(entry["point_px"])
            if len(pt) != 2:
                raise SetupError(
                    f"{path}: endzone_anchor.lines[].point_px must be [x, y], "
                    f"got {pt!r}."
                )
            parsed_lines.append({
                "point_px": [float(pt[0]), float(pt[1])],
                "world_x_m": float(entry["world_x_m"]),
            })
        endzone_anchor = {"lines": parsed_lines}
    return PlayMeta(
        season=str(raw["season"]),
        week=int(raw["week"]),
        home_team=str(raw["home_team"]),
        away_team=str(raw["away_team"]),
        fps=float(raw.get("fps", 30.0)),
        gsis_play_id=str(gsis) if gsis is not None else None,
        calib_hints=hints,
        endzone_prior=endzone_prior,
        endzone_anchor=endzone_anchor,
    )
