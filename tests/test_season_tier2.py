"""Tier-2 CPU logic: roster alignment, consensus, collect_uids, betas refine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.identity.alignment import align_participation
from nfl_gsplat.identity.consensus import apply_remap, build_consensus_remap
from nfl_gsplat.season.collect_uids import collect_player_uids, uids_to_build
from nfl_gsplat.season.refine_betas import BetasAppearance, select_best_betas
from nfl_gsplat.utils.io import write_json
from nfl_gsplat.season.scaffold import scaffold_play


# --- T2.2 alignment ---------------------------------------------------------

def test_align_participation_maps_to_our_keys(tmp_path):
    part = pd.DataFrame({
        "game_id": ["G1", "G1", "G1"],
        "play_id": ["100", "100", "200"],
        "player_uid": ["00-A", "00-B", "00-C"],
    })
    play_dirs = [
        scaffold_play(tmp_path, season=2024, week=1, away="AWAY", home="HOME",
                      play="play_001", gsis_play_id="100"),
        scaffold_play(tmp_path, season=2024, week=1, away="AWAY", home="HOME",
                      play="play_002", gsis_play_id="200"),
        scaffold_play(tmp_path, season=2024, week=1, away="AWAY", home="HOME",
                      play="play_003"),
    ]
    out = align_participation(part, play_dirs, "game_001", gsis_game_id="G1")
    assert out[("game_001", "play_001")] == ["00-A", "00-B"]
    assert out[("game_001", "play_002")] == ["00-C"]
    # play_003 has no gsis id → omitted (falls back to per-game roster).
    assert ("game_001", "play_003") not in out


# --- T2.3 consensus ---------------------------------------------------------

def test_consensus_remaps_rare_misread_uid():
    registry = {
        "season": "2024",
        "plays": {
            **{f"g/p{i}": [{"entity_type": "player", "player_uid": "2024_HOME_12"}] for i in range(5)},
            "g/p99": [{"entity_type": "player", "player_uid": "2024_HOME_13"}],   # one misread
        },
        "uids": {},
    }
    remap = build_consensus_remap(registry)
    assert remap == {"2024_HOME_13": "2024_HOME_12"}

    ents = [{"instance_id": "9", "player_uid": "2024_HOME_13", "entity_type": "player"}]
    assert apply_remap(ents, remap)[0]["player_uid"] == "2024_HOME_12"


def test_consensus_leaves_roster_uids_untouched():
    registry = {
        "plays": {
            "g/p1": [{"entity_type": "player", "player_uid": "00-9999"}],
            "g/p2": [{"entity_type": "player", "player_uid": "00-0001"}],
        },
        "uids": {},
    }
    # gsis ids don't parse as season_team_jersey → no remap.
    assert build_consensus_remap(registry) == {}


# --- T2.1 collect_uids ------------------------------------------------------

def test_collect_and_filter_uids(tmp_path):
    e1 = tmp_path / "p1_entities.json"
    e2 = tmp_path / "p2_entities.json"
    write_json(e1, [
        {"instance_id": "1", "player_uid": "qb_12", "entity_type": "player"},
        {"instance_id": "9", "player_uid": "__referee__", "entity_type": "referee"},
    ])
    write_json(e2, [
        {"instance_id": "1", "player_uid": "qb_12", "entity_type": "player"},   # recurs
        {"instance_id": "2", "player_uid": "wr_81", "entity_type": "player"},
    ])
    uids = collect_player_uids([e1, e2])
    assert uids == {"qb_12", "wr_81"}        # referee excluded; player dedup'd

    lib = AvatarLibrary(tmp_path / "library", season=2024)
    # Pretend qb_12 is already cached → only wr_81 remains to build.
    from tests.test_avatar_library import _avatar
    lib.put_avatar("qb_12", _avatar())
    assert uids_to_build([e1, e2], lib) == ["wr_81"]


# --- T2.5 betas refine ------------------------------------------------------

def test_select_best_betas_picks_highest_score():
    apps = [
        BetasAppearance(bbox_area=100.0, conf=0.9, betas=np.full(10, 0.1)),
        BetasAppearance(bbox_area=400.0, conf=0.8, betas=np.full(10, 0.7)),   # 320, best
        BetasAppearance(bbox_area=50.0, conf=0.95, betas=np.full(10, 0.3)),
    ]
    best = select_best_betas(apps)
    assert np.allclose(best, 0.7)


def test_select_best_betas_empty_is_none():
    assert select_best_betas([]) is None
