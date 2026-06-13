"""Map nflverse participation to our (game, play) keys for candidate narrowing.

Participation data is keyed by nflverse game/play ids; our pipeline is keyed by
``(game_id, play_id)`` with per-play meta.yaml files. Each play's meta.yaml
optionally carries ``gsis_play_id`` linking the two. This builds the
``participation`` dict :class:`~nfl_gsplat.identity.roster.RosterSource` accepts
(``{(game_id, play_id): [player_uid]}``), narrowing the candidate set per play
to the ~22 players actually on the field.

Plays without a ``gsis_play_id`` are omitted, so the RosterSource falls back to
the full per-game roster for them.
"""
from __future__ import annotations

import pandas as pd

from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.meta import load_meta


def align_participation(
    participation_long: pd.DataFrame,
    play_dirs: list[PlayDir],
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
    for play_dir in play_dirs:
        meta = load_meta(play_dir.meta_yaml)
        if meta.gsis_play_id is None:
            continue
        rows = pp[pp["play_id"].astype(str) == meta.gsis_play_id]
        if gsis_game_id is not None and "game_id" in pp.columns:
            rows = rows[rows["game_id"].astype(str) == str(gsis_game_id)]
        uids = [str(u) for u in rows["player_uid"].tolist()]
        if uids:
            out[(game_id, play_dir.play_id)] = uids
    return out
