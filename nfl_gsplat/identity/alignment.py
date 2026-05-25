"""Map nflverse participation to our (game, play) keys for candidate narrowing.

Participation data is keyed by nflverse game/play ids; our pipeline is keyed by
``(game_id, play_id)`` with frame windows in ``plays.yaml``. Each play window
optionally carries ``gsis_play_id`` linking the two. This builds the
``participation`` dict :class:`~nfl_gsplat.identity.roster.RosterSource` accepts
(``{(game_id, play_id): [player_uid]}``), narrowing the candidate set per play
to the ~22 players actually on the field.

Plays without a ``gsis_play_id`` are omitted, so the RosterSource falls back to
the full per-game roster for them.
"""
from __future__ import annotations

import pandas as pd

from nfl_gsplat.utils.plays import PlaysManifest


def align_participation(
    participation_long: pd.DataFrame,
    manifest: PlaysManifest,
    game_id: str,
    gsis_game_id: str | None = None,
) -> dict[tuple[str, str], list[str]]:
    """Return ``{(game_id, play_id): [player_uid]}`` from a long participation
    table (columns ``game_id, play_id, player_uid`` in nflverse gsis ids)."""
    out: dict[tuple[str, str], list[str]] = {}
    pp = participation_long
    have = {"play_id", "player_uid"}.issubset(pp.columns)
    if not have:
        return out
    for window in manifest.plays.values():
        if window.gsis_play_id is None:
            continue
        rows = pp[pp["play_id"].astype(str) == window.gsis_play_id]
        if gsis_game_id is not None and "game_id" in pp.columns:
            rows = rows[rows["game_id"].astype(str) == str(gsis_game_id)]
        uids = [str(u) for u in rows["player_uid"].tolist()]
        if uids:
            out[(game_id, window.play_id)] = uids
    return out
