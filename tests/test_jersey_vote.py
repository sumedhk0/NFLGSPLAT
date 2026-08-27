"""Tests for constrained jersey assignment.

The failure that matters is a CONFIDENT WRONG identity: it silently attaches the
wrong body shape, the wrong avatar and the wrong cross-camera correspondence,
and nothing downstream can detect it. So these lean on the refusal paths.
"""
from __future__ import annotations

import collections

import pandas as pd

from nfl_gsplat.identity.jersey_vote import (assign, restrict_to_known, tally)


def _on_field():
    return pd.DataFrame({
        "team": ["ARI", "ARI", "SEA"],
        "jersey_number": [1.0, 18.0, 21.0],
        "full_name": ["Kyler Murray", "Marvin Harrison Jr.", "Devon Witherspoon"],
        "position": ["QB", "WR", "DB"],
        "height_m": [1.778, 1.905, 1.829],
    })


def test_reads_outside_the_known_set_are_dropped():
    """The participation data's whole value: a read of 48 when nobody wears 48
    is evidence of a misread, removable with no confidence threshold."""
    votes = tally([(7, 18), (7, 18), (7, 48), (7, 99)])
    kept = restrict_to_known(votes, [1, 18, 21])
    assert dict(kept[7]) == {18: 2}


def test_confident_track_is_assigned_with_roster_facts():
    votes = {5: collections.Counter({18: 6, 1: 1})}
    got = assign(votes, _on_field())
    assert len(got) == 1
    assert got[0].jersey == 18
    assert got[0].player == "Marvin Harrison Jr."
    assert got[0].height_m == 1.905     # the number SMPLest-X could not supply


def test_two_tracks_cannot_claim_the_same_player():
    """Taking each track's best guess independently would give both tracks #18
    and leave #1 unclaimed -- wrong, and invisible without a global assignment."""
    votes = {5: collections.Counter({18: 9}), 6: collections.Counter({18: 4, 1: 3})}
    got = assign(votes, _on_field())
    assert len({t.jersey for t in got}) == len(got), "a player was assigned twice"
    assert {t.track_id for t in got} <= {5, 6}
    strong = [t for t in got if t.track_id == 5]
    assert strong and strong[0].jersey == 18, "the better-read track lost its player"


def test_a_tie_is_left_unassigned():
    """18 vs 85 tied on real footage. A coin flip here is a wrong identity."""
    votes = {5: collections.Counter({18: 3, 1: 3})}
    assert assign(votes, _on_field()) == []


def test_too_few_votes_is_left_unassigned():
    votes = {5: collections.Counter({18: 1})}
    assert assign(votes, _on_field(), min_votes=2) == []


def test_no_usable_reads_returns_empty_not_an_error():
    assert assign({}, _on_field()) == []


def test_more_tracks_than_players_does_not_crash():
    """Detections include officials and sideline staff, so tracks outnumber the
    22 routinely."""
    votes = {i: collections.Counter({18: 5}) for i in range(6)}
    got = assign(votes, _on_field())
    assert len(got) <= 3
    assert len({t.jersey for t in got}) == len(got)


def test_strong_reads_override_a_colour_mistake():
    """Colour is evidence, not ground truth -- measured 8 of 9 correct on real
    footage. A hard constraint turned that 11% error rate straight into lost
    identities (8 identified against 9 with no constraint), so a clear read must
    win over the colour label."""
    votes = {5: collections.Counter({21: 9})}          # 21 is SEA
    got = assign(votes, _on_field(), team_of_track={5: "ARI"})
    assert len(got) == 1 and got[0].jersey == 21


def test_colour_still_decides_a_weak_tie():
    """The case colour was added for: votes cannot separate 18 from 21."""
    votes = {5: collections.Counter({18: 4, 21: 4})}
    got = assign(votes, _on_field(), team_of_track={5: "ARI"})
    assert len(got) == 1 and got[0].jersey == 18, "colour failed to break a tie"


def test_team_constraint_breaks_a_tie_votes_cannot():
    """18 (ARI) vs 21 (SEA) tied on votes; colour decides."""
    votes = {5: collections.Counter({18: 4, 21: 4})}
    assert assign(votes, _on_field()) == [], "tie should be refused without team"
    got = assign(votes, _on_field(), team_of_track={5: "ARI"})
    assert len(got) == 1 and got[0].jersey == 18


def test_unknown_team_for_a_track_is_not_constrained():
    """Colour clustering does not always label every track; those must still be
    assignable rather than silently dropped."""
    votes = {5: collections.Counter({18: 6})}
    got = assign(votes, _on_field(), team_of_track={})
    assert len(got) == 1 and got[0].jersey == 18


def test_alignment_alone_identifies_a_sole_candidate():
    """A track the formation calls QB, with NO jersey reads at all, must still
    be identified when only one player on the field plays that spot.

    This is the case alignment exists for -- the quarterback on play_001 was
    never read once -- and it was silently broken by a stale loop variable: the
    sole-candidate test read the position of whichever track the cost loop
    happened to finish on, not the track being decided.
    """
    votes = {5: collections.Counter(), 6: collections.Counter({18: 9})}
    got = assign(votes, _on_field(), position_of_track={5: "QB", 6: "WR"})
    by_track = {t.track_id: t.jersey for t in got}
    assert by_track.get(5) == 1, "the quarterback was not identified by alignment"
    assert by_track.get(6) == 18


def test_a_shared_position_group_still_needs_votes():
    """With TWO receivers on the field, alignment narrows but cannot decide, so
    the vote thresholds must still apply. Otherwise the solver would pick one of
    them arbitrarily and report it with full confidence."""
    on = pd.DataFrame({
        "team": ["ARI", "ARI", "SEA"],
        "jersey_number": [18.0, 4.0, 21.0],
        "full_name": ["Marvin Harrison Jr.", "Greg Dortch", "Devon Witherspoon"],
        "position": ["WR", "WR", "DB"],
        "height_m": [1.905, 1.702, 1.829],
    })
    votes = {5: collections.Counter()}
    got = assign(votes, on, position_of_track={5: "WR"})
    assert got == [], "a track with no votes was assigned to one of several WRs"
