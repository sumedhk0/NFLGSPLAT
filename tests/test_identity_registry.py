"""Identity resolution: roster-constrained assignment + persistent registry.

Synthetic tracks (no GPU / OCR) carry pre-attached ``team`` / ``is_referee``
columns as the real pipeline would after team_color classification.
"""
from __future__ import annotations

import pandas as pd

from nfl_gsplat.identity.registry import (
    EntityType,
    IdentityMatchConfig,
    REFEREE_UID,
    assign_identities,
    consensus_jersey,
    load_registry,
    register_play,
    resolve_tracks,
)
from nfl_gsplat.identity.roster import OcrOnlySource, RosterEntry, RosterSource
from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS


def _track(track_id: int, jersey: int, team: str | None, is_ref: bool = False,
           n_frames: int = 5) -> pd.DataFrame:
    rows = []
    for f in range(n_frames):
        row = {c: 0 for c in TRACK_COLUMNS}
        row.update({
            "frame": f, "cam": "sideline", "track_id": track_id,
            "global_player_id": track_id, "conf": 0.9,
            "jersey_number_ocr": jersey,
        })
        row["team"] = team
        row["is_referee"] = is_ref
        rows.append(row)
    return pd.DataFrame(rows)


def _tracks(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _cfg(**kw) -> IdentityMatchConfig:
    return IdentityMatchConfig(season=2024, **kw)


# --- misread jersey snaps to the nearest valid candidate -------------------

def test_misread_jersey_snaps_to_nearest_candidate():
    candidates = [
        RosterEntry("h12", "HOME", 12, "QB"),
        RosterEntry("h80", "HOME", 80, "WR"),
    ]
    # Track voted "13" (a misread of 12) on HOME → should snap to #12, not #80.
    df = _track(track_id=1, jersey=13, team="HOME")
    out = assign_identities(df, candidates, _cfg())
    assert len(out) == 1
    assert out[0].player_uid == "h12"
    assert out[0].entity_type == EntityType.PLAYER.value


# --- colliding home/away numbers disambiguated by team ---------------------

def test_colliding_jerseys_disambiguated_by_team():
    candidates = [
        RosterEntry("h12", "HOME", 12, "QB"),
        RosterEntry("a12", "AWAY", 12, "CB"),
    ]
    df = _tracks(
        _track(track_id=1, jersey=12, team="HOME"),
        _track(track_id=2, jersey=12, team="AWAY"),
    )
    out = {a.track_id: a.player_uid for a in assign_identities(df, candidates, _cfg())}
    assert out[1] == "h12"
    assert out[2] == "a12"


# --- non-roster tracks → referee (striped) or other (dropped) --------------

def test_referee_track_routes_to_referee_uid():
    candidates = [RosterEntry("h12", "HOME", 12, "QB")]
    df = _track(track_id=9, jersey=-1, team=None, is_ref=True)
    out = assign_identities(df, candidates, _cfg())
    assert out[0].player_uid == REFEREE_UID
    assert out[0].entity_type == EntityType.REFEREE.value


def test_unmatched_non_referee_is_other():
    candidates = [RosterEntry("h12", "HOME", 12, "QB")]
    # Team "ZZZ" matches no candidate → gated out → OTHER (not a referee).
    df = _track(track_id=7, jersey=99, team="ZZZ", is_ref=False)
    out = assign_identities(df, candidates, _cfg())
    assert out[0].entity_type == EntityType.OTHER.value
    assert out[0].player_uid == ""


# --- resolve_tracks attaches columns ---------------------------------------

def test_resolve_tracks_adds_columns():
    candidates = [RosterEntry("h12", "HOME", 12, "QB")]
    df = _track(track_id=1, jersey=12, team="HOME")
    out = resolve_tracks(df, candidates, _cfg())
    assert "player_uid" in out.columns and "entity_type" in out.columns
    assert (out["player_uid"] == "h12").all()


# --- OCR-only synthesizes a uid from (team, jersey) ------------------------

def test_ocr_only_synthesizes_uid():
    src = OcrOnlySource()
    candidates = src.candidates_for_play("game_001", "play_001")
    assert candidates == []
    df = _track(track_id=1, jersey=12, team="HOME")
    out = assign_identities(df, candidates, _cfg())
    assert out[0].player_uid == "2024_HOME_12"
    assert out[0].entity_type == EntityType.PLAYER.value


# --- RosterSource candidate lookup -----------------------------------------

def test_roster_source_from_dataframe_and_per_game_fallback():
    roster_df = pd.DataFrame({
        "gsis_id": ["00-1", "00-2", "00-3"],
        "team": ["HOME", "HOME", "AWAY"],
        "jersey_number": [12, 80, 12],
        "position": ["QB", "WR", "CB"],
        "player_name": ["A", "B", "C"],
    })
    src = RosterSource.from_dataframe(
        roster_df, season=2024, game_teams={"game_001": ("HOME", "AWAY")}
    )
    cands = src.candidates_for_play("game_001", "play_001")
    assert {c.player_uid for c in cands} == {"00-1", "00-2", "00-3"}


# --- persistence: same uid across two plays --------------------------------

def test_registry_persists_uid_across_plays(tmp_path):
    candidates = [RosterEntry("h12", "HOME", 12, "QB")]
    cfg = _cfg()
    root = tmp_path / "outputs"

    for play in ("play_001", "play_030"):
        df = _track(track_id=1, jersey=12, team="HOME")
        asg = assign_identities(df, candidates, cfg)
        register_play(2024, "game_001", play, asg, outputs_root=root)

    reg = load_registry(2024, outputs_root=root)
    assert "game_001/play_001" in reg["plays"]
    assert "game_001/play_030" in reg["plays"]
    assert "h12" in reg["uids"]
    assert consensus_jersey(2024, "h12", outputs_root=root) == 12
