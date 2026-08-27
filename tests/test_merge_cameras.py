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


# --- geometry as a falsifier -------------------------------------------------
# Agreement on a jersey number is not proof the two tracks are one player.
# These cover the case that actually bit: 11 m and 16 m apart on play_001.

from nfl_gsplat.identity.merge_cameras import (check_separation,
                                               drop_contradicted)


def walk(track, x0, y0, n=40, dx=0.0):
    return {track: {f: (x0 + dx * f, y0) for f in range(n)}}


def two_cams(sx, sy, ex, ey, n=40):
    pos = {"sideline": {}, "endzone": {}}
    pos["sideline"].update(walk(1, sx, sy, n))
    pos["endzone"].update(walk(7, ex, ey, n))
    return pos


def merged_pair(sv=10, ev=10):
    return merge({
        "sideline": [ident(1, 85, "McBride", votes=sv)],
        "endzone": [ident(7, 85, "McBride", votes=ev)],
    })


def test_agreeing_tracks_pass():
    checks = check_separation(merged_pair(), two_cams(0, 0, 0.5, 0.5),
                              "sideline", "endzone")
    assert checks[85].testable
    assert not checks[85].contradicted
    assert checks[85].separation_m == pytest.approx(0.7071, abs=1e-3)


def test_distant_tracks_are_contradicted():
    """The real failure: same jersey, 16 m apart, different human beings."""
    checks = check_separation(merged_pair(), two_cams(0, 0, 16.0, 0),
                              "sideline", "endzone")
    assert checks[85].contradicted


def test_too_few_shared_frames_is_untestable_not_a_pass():
    """Silence must not read as agreement."""
    checks = check_separation(merged_pair(), two_cams(0, 0, 99.0, 0, n=3),
                              "sideline", "endzone")
    assert not checks[85].testable
    assert not checks[85].contradicted     # untestable, so not an accusation
    assert checks[85].n_frames == 3


def test_players_only_one_camera_saw_are_not_checked():
    merged = merge({"sideline": [ident(1, 85, "McBride")]})
    assert check_separation(merged, two_cams(0, 0, 0, 0),
                            "sideline", "endzone") == {}


def test_contradiction_keeps_the_better_evidenced_camera():
    """Kyler Murray: 0 votes on the sideline, 73 on the endzone."""
    merged = merged_pair(sv=0, ev=73)
    checks = check_separation(merged, two_cams(0, 0, 11.0, 0),
                              "sideline", "endzone")
    kept = drop_contradicted(merged, checks)
    assert kept[85].cameras == ("endzone",)
    assert kept[85].tracks == {"endzone": 7}


def test_contradiction_with_no_clear_winner_drops_the_player():
    """A wrong identity is worse than a missing one."""
    merged = merged_pair(sv=10, ev=11)
    checks = check_separation(merged, two_cams(0, 0, 11.0, 0),
                              "sideline", "endzone")
    assert drop_contradicted(merged, checks) == {}


def test_uncontradicted_players_survive_untouched():
    merged = merged_pair(sv=4, ev=44)
    checks = check_separation(merged, two_cams(0, 0, 0.5, 0),
                              "sideline", "endzone")
    kept = drop_contradicted(merged, checks)
    assert kept[85].cameras == ("endzone", "sideline")
