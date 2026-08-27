"""Merging per-camera identities, and the correspondence that falls out of it."""
import pandas as pd
import pytest

from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.identity.jersey_vote import TrackIdentity
from nfl_gsplat.identity.merge_cameras import (PlayerIdentity, correspondence,
                                               coverage, merge)


def ident(track_id, jersey, player="Someone", team="ARI", height=1.9, votes=10):
    return TrackIdentity(track_id=track_id, jersey=jersey, player=player,
                         team=team, height_m=height, votes=votes, margin=3.0)


def test_union_is_larger_than_either_camera():
    """The point of two cameras: each covers what the other misses."""
    merged = merge({
        "sideline": [ident(1, 85, "McBride"), ident(2, 18, "Harrison")],
        "endzone": [ident(7, 85, "McBride"), ident(9, 99, "Williams")],
    })
    assert set(merged) == {85, 18, 99}


def test_corroborated_only_when_two_cameras_agree():
    merged = merge({
        "sideline": [ident(1, 85, "McBride"), ident(2, 18, "Harrison")],
        "endzone": [ident(7, 85, "McBride")],
    })
    assert merged[85].corroborated
    assert not merged[18].corroborated
    assert merged[85].cameras == ("endzone", "sideline")


def test_votes_accumulate_across_cameras():
    merged = merge({
        "sideline": [ident(1, 85, "McBride", votes=4)],
        "endzone": [ident(7, 85, "McBride", votes=44)],
    })
    assert merged[85].votes == {"sideline": 4, "endzone": 44}
    assert merged[85].total_votes == 48


def test_correspondence_pairs_tracks_without_geometry():
    """The whole reason this module exists."""
    merged = merge({
        "sideline": [ident(1, 85, "McBride"), ident(2, 18, "Harrison")],
        "endzone": [ident(7, 85, "McBride"), ident(3, 18, "Harrison")],
    })
    assert correspondence(merged, "sideline", "endzone") == {1: 7, 2: 3}


def test_correspondence_skips_players_only_one_camera_saw():
    merged = merge({
        "sideline": [ident(1, 85, "McBride"), ident(2, 18, "Harrison")],
        "endzone": [ident(7, 85, "McBride")],
    })
    assert correspondence(merged, "sideline", "endzone") == {1: 7}


def test_conflicting_names_for_one_jersey_raise():
    """Different rosters would poison every identity downstream."""
    with pytest.raises(CalibrationError, match="different rosters"):
        merge({
            "sideline": [ident(1, 85, "McBride")],
            "endzone": [ident(7, 85, "Somebody Else")],
        })


def test_coverage_names_who_is_missing():
    on_field = pd.DataFrame([
        {"jersey_number": 85, "full_name": "McBride", "team": "ARI", "position": "TE"},
        {"jersey_number": 18, "full_name": "Harrison", "team": "ARI", "position": "WR"},
        {"jersey_number": 1, "full_name": "Murray", "team": "ARI", "position": "QB"},
    ])
    merged = merge({"sideline": [ident(1, 85, "McBride")]})
    n, missing = coverage(merged, on_field)
    assert n == 1
    assert sorted(j for j, *_ in missing) == [1, 18]


def test_empty_merge_is_empty_not_an_error():
    assert merge({"sideline": []}) == {}
