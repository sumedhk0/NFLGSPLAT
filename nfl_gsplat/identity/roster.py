"""Roster / participation prior as a *pluggable* identity source.

The hard part of season-scale reuse is recognition: deciding that the #12 in
play 4 of game 1 is the same person as the #12 in play 30 of game 9. An
external roster turns this from open-set re-identification into constrained
classification — per play we know the small candidate set of (jersey, team,
position) tuples actually on the field.

Two sources implement the :class:`IdentitySource` protocol:

- :class:`RosterSource` — backed by nflverse / ``nfl_data_py`` exports
  (rosters + optional participation), loaded from ``data/rosters/{season}/``.
- :class:`OcrOnlySource` — no roster available; the candidate set is empty and
  the registry synthesizes a uid from the observed (team, jersey). Coarser, but
  keeps the pipeline working without external data.

``player_uid`` is the stable key for the avatar/shape library. When the roster
carries an nflverse ``gsis_id`` we use it (survives mid-season jersey changes
and trades); otherwise we fall back to ``f"{season}_{team}_{jersey}"``.

CPU-only. ``pandas`` is required; ``nfl_data_py`` is imported lazily only by the
fetch helper, never here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def synthetic_uid(season: int | str, team: str, jersey: int) -> str:
    """Deterministic fallback uid when no nflverse id is available."""
    return f"{season}_{team}_{int(jersey)}"


@dataclass(frozen=True)
class RosterEntry:
    """One rostered player. ``player_uid`` is the library key."""

    player_uid: str
    team: str
    jersey: int
    position: str = ""
    name: str | None = None


@runtime_checkable
class IdentitySource(Protocol):
    """Returns the candidate players plausibly on the field for a play.

    An empty list is a valid answer (see :class:`OcrOnlySource`) and signals
    the resolver to synthesize identities from observed (team, jersey).
    """

    def candidates_for_play(self, game_id: str, play_id: str) -> list[RosterEntry]: ...


class OcrOnlySource:
    """No roster: candidates are always empty; identities come from OCR+color."""

    def candidates_for_play(self, game_id: str, play_id: str) -> list[RosterEntry]:
        return []


# Columns we expect in a roster parquet (nflverse-style; extra columns ignored).
_ROSTER_COLUMNS = ("team", "jersey_number", "position")


class RosterSource:
    """Roster-backed identity source.

    Built from a roster DataFrame (one row per rostered player for the season /
    game) and an optional participation map ``(game_id, play_id) -> [player_uid]``
    that narrows candidates to the personnel actually on the field. When no
    participation row exists for a play, we fall back to the full per-game
    roster — coarser, but still a tight candidate set (~100 players, two teams).
    """

    def __init__(
        self,
        roster_by_uid: dict[str, RosterEntry],
        *,
        game_teams: dict[str, tuple[str, str]] | None = None,
        participation: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        self._roster = roster_by_uid
        self._game_teams = game_teams or {}
        self._participation = participation or {}

    # --- construction -------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        roster_df: pd.DataFrame,
        season: int | str,
        *,
        game_teams: dict[str, tuple[str, str]] | None = None,
        participation: dict[tuple[str, str], list[str]] | None = None,
    ) -> "RosterSource":
        """Build from a roster DataFrame. Requires columns ``team``,
        ``jersey_number``, ``position``; uses ``gsis_id`` for the uid when
        present, otherwise a synthetic ``{season}_{team}_{jersey}`` key.
        """
        missing = [c for c in _ROSTER_COLUMNS if c not in roster_df.columns]
        if missing:
            raise SetupError(
                f"roster DataFrame missing columns {missing}; expected "
                f"{list(_ROSTER_COLUMNS)} (nflverse export). See SETUP.md §9."
            )
        has_gsis = "gsis_id" in roster_df.columns
        has_name = "player_name" in roster_df.columns
        roster: dict[str, RosterEntry] = {}
        for _, row in roster_df.iterrows():
            jersey_raw = row["jersey_number"]
            if pd.isna(jersey_raw):
                continue
            team = str(row["team"])
            jersey = int(jersey_raw)
            uid = (
                str(row["gsis_id"])
                if has_gsis and not pd.isna(row["gsis_id"])
                else synthetic_uid(season, team, jersey)
            )
            roster[uid] = RosterEntry(
                player_uid=uid,
                team=team,
                jersey=jersey,
                position=str(row["position"]) if not pd.isna(row["position"]) else "",
                name=str(row["player_name"]) if has_name and not pd.isna(row["player_name"]) else None,
            )
        _LOG.info(f"RosterSource: {len(roster)} rostered players for season {season}")
        return cls(roster, game_teams=game_teams, participation=participation)

    @classmethod
    def from_parquet(
        cls,
        season: int | str,
        roster_dir: Path | str,
        *,
        game_teams: dict[str, tuple[str, str]] | None = None,
    ) -> "RosterSource":
        """Load ``rosters.parquet`` (and optional ``participation.parquet``)
        produced by ``scripts/fetch_roster.py``. Fails loudly if absent."""
        roster_dir = Path(roster_dir)
        roster_path = roster_dir / "rosters.parquet"
        if not roster_path.exists():
            raise SetupError(
                f"roster parquet missing at {roster_path}. Run "
                "`python scripts/fetch_roster.py --season "
                f"{season}` — see SETUP.md §9."
            )
        roster_df = pd.read_parquet(roster_path)
        participation = None
        part_path = roster_dir / "participation.parquet"
        if part_path.exists():
            participation = _participation_from_df(pd.read_parquet(part_path))
        return cls.from_dataframe(
            roster_df, season, game_teams=game_teams, participation=participation
        )

    # --- IdentitySource -----------------------------------------------------

    def candidates_for_play(self, game_id: str, play_id: str) -> list[RosterEntry]:
        uids = self._participation.get((game_id, play_id))
        if uids:
            return [self._roster[u] for u in uids if u in self._roster]
        # Fall back to the full per-game roster (both teams) when participation
        # for this play is unavailable.
        teams = self._game_teams.get(game_id)
        if teams:
            return [e for e in self._roster.values() if e.team in teams]
        return list(self._roster.values())


def _participation_from_df(df: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Build ``(game_id, play_id) -> [player_uid]`` from a participation table.

    Expects columns ``game_id``, ``play_id``, ``player_uid`` (one row per
    on-field player per play). Unknown schemas are ignored (returns {}).
    """
    needed = {"game_id", "play_id", "player_uid"}
    if not needed.issubset(df.columns):
        _LOG.warning(
            f"participation parquet missing {needed - set(df.columns)}; "
            "ignoring participation, falling back to per-game roster."
        )
        return {}
    out: dict[tuple[str, str], list[str]] = {}
    for (g, p), grp in df.groupby(["game_id", "play_id"]):
        out[(str(g), str(p))] = [str(u) for u in grp["player_uid"].tolist()]
    return out
