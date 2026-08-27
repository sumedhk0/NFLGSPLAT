"""Tests for the nflverse participation lookup.

The failure mode that matters is a WRONG match rather than no match: attaching
the wrong 22 players to a play makes every downstream identity confidently
wrong, and nothing later in the pipeline could detect it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.identity.participation import (INCH_TO_M, find_play, game_id,
                                               players_on_play)


def _pbp():
    return pd.DataFrame({
        "game_id": ["2025_04_SEA_ARI"] * 3,
        "play_id": [63, 88, 120],
        "time": ["14:54", "13:02", "13:02"],
        "desc": [
            "(14:54) (Shotgun) 1-K.Murray pass short left to 18-M.Harrison pushed ob at ARZ 34",
            "(13:02) 33-T.Benson right guard to ARZ 40 for 6 yards",
            "(13:02) 33-T.Benson right guard to ARZ 40 for 6 yards",
        ],
    })


def _participation():
    return pd.DataFrame({
        "nflverse_game_id": ["2025_04_SEA_ARI"],
        "play_id": [63],
        "offense_players": ["A;B"],
        "defense_players": ["C"],
    })


def _roster():
    return pd.DataFrame({
        "week": [4, 4, 4, 3],
        "gsis_id": ["A", "B", "C", "A"],
        "team": ["ARI", "ARI", "SEA", "ARI"],
        "jersey_number": [1.0, 18.0, 21.0, 1.0],
        "full_name": ["Kyler Murray", "Marvin Harrison Jr.", "Devon Witherspoon", "Kyler Murray"],
        "position": ["QB", "WR", "DB", "QB"],
        "height": [70.0, 75.0, 72.0, 70.0],
        "weight": [207.0, 205.0, 180.0, 207.0],
    })


def test_game_id_maps_local_team_codes_to_nflverse():
    """This project calls Arizona AZ; nflverse calls it ARI. A silent mismatch
    yields an empty play table and looks like missing data."""
    assert game_id(2025, 4, "SEA", "AZ") == "2025_04_SEA_ARI"


def test_find_play_matches_a_clip_title():
    got = find_play(_pbp(), "2025_04_SEA_ARI",
                    "(14:54) (Shotgun) 1-K.Murray pass short left to 18-M.Harrison pushed ob at ARZ 3")
    assert int(got["play_id"]) == 63


def test_ambiguous_title_refuses_rather_than_guessing():
    """Two plays share a description AND a clock. Picking one would attach the
    wrong 22 players, and nothing downstream could tell."""
    with pytest.raises(SetupError, match="confidently wrong"):
        find_play(_pbp(), "2025_04_SEA_ARI",
                  "(13:02) 33-T.Benson right guard to ARZ 40 for 6 yards")


def test_unknown_game_fails_with_the_team_code_hint():
    with pytest.raises(SetupError, match="ARI, not AZ"):
        find_play(_pbp(), "2025_04_SEA_AZ", "(14:54) whatever")


def test_players_on_play_returns_the_field_with_metric_height():
    got = players_on_play(_participation(), _roster(), "2025_04_SEA_ARI", 63, 4)
    assert len(got) == 3
    assert set(got["jersey_number"]) == {1.0, 18.0, 21.0}
    murray = got[got["full_name"] == "Kyler Murray"].iloc[0]
    assert murray["height_m"] == pytest.approx(70.0 * INCH_TO_M, abs=1e-9)


def test_partial_roster_join_fails_loud():
    """A player who never joins can never be identified. Better to learn that
    here than to wonder later why one track never resolves."""
    roster = _roster()
    roster = roster[roster["gsis_id"] != "C"]
    with pytest.raises(SetupError, match="matched 2 of 3"):
        players_on_play(_participation(), roster, "2025_04_SEA_ARI", 63, 4)


def test_missing_participation_record_fails_loud():
    with pytest.raises(SetupError, match="participation"):
        players_on_play(_participation(), _roster(), "2025_04_SEA_ARI", 999, 4)


def test_roster_week_is_respected():
    """Weekly rosters change; using the wrong week can attach a jersey number
    the player did not wear in this game."""
    got = players_on_play(_participation(), _roster(), "2025_04_SEA_ARI", 63, 4)
    assert (got["gsis_id"] == "A").sum() == 1, "week 3 duplicate leaked in"
