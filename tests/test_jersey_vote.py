"""Tests for constrained jersey assignment.

The failure that matters is a CONFIDENT WRONG identity: it silently attaches the
wrong body shape, the wrong avatar and the wrong cross-camera correspondence,
and nothing downstream can detect it. So these lean on the refusal paths.
"""
from __future__ import annotations

import collections

import pandas as pd

from nfl_gsplat.identity.jersey_vote import (assign, credit_truncations,
                                            is_truncation_of,
                                            restrict_to_known, tally)


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


# --- the sole-candidate rule must be injective --------------------------------
# Alignment alone is DECISIVE only when exactly one player on the field plays a
# spot AND exactly one track claims it. Measured on play_001's sideline, tracks
# 15 and 5 BOTH voted tight end (37/45 and 26/35 frames) while Arizona fielded
# one; the rule fired anyway and let the assignment solver break the tie with no
# evidence, on zero jersey votes.

def test_two_tracks_claiming_one_sole_position_get_no_free_identity():
    """Ambiguous alignment must fall back to the vote thresholds, not guess."""
    votes = {10: collections.Counter(), 11: collections.Counter()}
    got = assign(votes, _on_field(),
                 position_of_track={10: "QB", 11: "QB"})
    assert got == []


def test_one_track_claiming_a_sole_position_still_wins_on_alignment():
    """The rule's real case is preserved: one quarterback, one claimant."""
    votes = {10: collections.Counter()}
    got = assign(votes, _on_field(), position_of_track={10: "QB"})
    assert [(t.track_id, t.jersey) for t in got] == [(10, 1)]


def test_ambiguous_alignment_still_lets_a_read_track_win():
    """Losing the free identity must not lose an EARNED one."""
    votes = {10: collections.Counter({1: 9}), 11: collections.Counter()}
    got = assign(votes, _on_field(), position_of_track={10: "QB", 11: "QB"})
    assert [(t.track_id, t.jersey) for t in got] == [(10, 1)]


# --- OCR drops digits, it rarely invents them ---------------------------------
# Measured on play_001: among tracks with a contested read, the top two
# candidates are digit-related 24% (sideline) and 36% (endzone) of the time,
# against 6% for random pairs of the jerseys on the field. The confusions are
# what truncation predicts -- 14 vs 4, 85 vs 8, 13 vs 1, 70 vs 0.

def test_truncation_is_recognised_at_both_ends():
    assert is_truncation_of(4, 14)      # dropped LEADING digit
    assert is_truncation_of(8, 85)      # dropped TRAILING digit
    assert is_truncation_of(0, 70)
    assert is_truncation_of(1, 13)


def test_a_number_is_not_a_truncation_of_itself_or_of_a_shorter_one():
    assert not is_truncation_of(14, 14)
    assert not is_truncation_of(14, 4)      # inventing a digit, not dropping
    assert not is_truncation_of(5, 14)      # unrelated digits


def test_short_reads_credit_the_longer_jersey():
    """The contest that motivated this: #14 read 53 times, #4 read 36."""
    votes = collections.Counter({14: 53, 4: 36})
    out = credit_truncations(votes, [14, 4, 20], weight=1.0)
    assert out[14] == 53 + 36
    assert out[4] == 36          # the short read stays, it may simply be right


def test_credit_never_goes_to_a_jersey_nobody_wears():
    votes = collections.Counter({4: 30})
    out = credit_truncations(votes, [4, 20, 21], weight=1.0)
    assert set(out) == {4}       # 14 is not on the field, so it gets nothing


def test_zero_weight_changes_nothing():
    votes = collections.Counter({14: 53, 4: 36})
    assert credit_truncations(votes, [14, 4], weight=0.0) == votes


def test_credited_votes_win_a_contest_the_margin_rule_would_have_lost():
    """#14 took 53 against 36 -- a ratio of 1.47, just under MIN_MARGIN."""
    on_field = pd.DataFrame({
        "team": ["ARI", "ARI"],
        "jersey_number": [14.0, 4.0],
        "full_name": ["Michael Wilson", "Greg Dortch"],
        "position": ["WR", "WR"],
        "height_m": [1.91, 1.70],
        "weight": [200.0, 173.0],
    })
    raw = {7: collections.Counter({14: 53, 4: 36})}
    assert assign(raw, on_field) == []          # margin 1.47 < 1.5, refused
    credited = {7: credit_truncations(raw[7], [14, 4], weight=1.0)}
    got = assign(credited, on_field)
    assert [(t.track_id, t.jersey) for t in got] == [(7, 14)]
