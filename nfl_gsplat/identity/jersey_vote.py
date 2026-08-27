"""Assign tracks to the known players on a play, from jersey votes.

The hard version of this problem -- read a number 0-99 off a 60-180 px crop and
trust it -- yielded 5 usable identities across a whole play. Two things make the
easy version possible:

* :mod:`nfl_gsplat.identity.participation` supplies the 22 players actually on
  the field, so a read of "48" when no 48 is playing is *evidence of a misread*
  rather than a new identity;
* tracks give ~1000 frames per player instead of one, so a single confident
  frame is not needed -- only a majority.

Assignment is global, not per track. Taking each track's best guess
independently lets two tracks claim the same player and leaves others
unclaimed, which is both wrong and undetectable; a one-to-one assignment makes
the competition explicit and lets a weakly-read track lose to a strongly-read
one rather than stealing from it.

Tracks with too little evidence are left UNASSIGNED. A wrong identity is worse
than a missing one: it silently attaches the wrong body shape, the wrong avatar
and the wrong cross-camera correspondence, and nothing downstream can detect it.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# A track must clear both to be assigned at all.
MIN_VOTES: int = 2
MIN_MARGIN: float = 1.5      # winner must beat the runner-up by this factor

# Cross-team pairings are PENALISED, not forbidden, in units of votes. Colour is
# evidence, not ground truth: measured against the roster on play_001 it was
# right for 8 of 9 identified tracks. Forbidding outright turned that 11% error
# rate straight into lost identities -- the hard constraint scored 8 identities
# against 9 with no constraint at all, and prevented no wrong assignment.
#
# At this value a track with a clear read (8+ votes) overrides a colour mistake,
# while a 4-4 tie is still decided by colour, which is the case colour was added
# for.
CROSS_TEAM_PENALTY: float = 5.0

# Formation position -> the coarse group a roster lists. The roster says "OL",
# alignment says "left guard"; they have to meet somewhere.
POSITION_GROUP = {
    "C": "OL", "G": "OL", "T": "OL", "OL": "OL",
    "QB": "QB", "RB": "RB", "FB": "RB", "TE": "TE", "WR": "WR",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "ILB": "LB", "OLB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "FS": "DB", "SS": "DB", "S": "DB", "DB": "DB",
}

# Weight of an alignment agreement, in votes. Deliberately larger than
# CROSS_TEAM_PENALTY: where a jersey read is a tenth-of-the-time guess at a
# number, alignment is a geometric fact about where a body stood, and on
# play_001 it labelled every one of the 22 while OCR reached 9.
POSITION_BONUS: float = 6.0


@dataclass(frozen=True)
class TrackIdentity:
    track_id: int
    jersey: int
    player: str
    team: str
    height_m: float
    votes: int
    margin: float


def tally(reads) -> dict[int, collections.Counter]:
    """``{track_id: Counter(jersey -> count)}`` from ``(track_id, jersey)`` pairs."""
    out: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for track_id, jersey in reads:
        out[int(track_id)][int(jersey)] += 1
    return dict(out)


def restrict_to_known(votes, known_jerseys) -> dict[int, collections.Counter]:
    """Drop reads for numbers nobody on the field is wearing.

    This is where the participation data earns its place: a misread is usually
    a number that is not in play, so most OCR noise is removable without any
    confidence threshold at all.
    """
    known = {int(j) for j in known_jerseys}
    out = {}
    for track_id, counter in votes.items():
        kept = collections.Counter({j: c for j, c in counter.items() if j in known})
        if kept:
            out[int(track_id)] = kept
    return out


def row_team(on_field, jersey):
    """Team of the player wearing jersey."""
    return on_field[on_field["jersey_number"] == jersey].iloc[0]["team"]


def assign(votes, on_field, *, min_votes: int = MIN_VOTES,
           min_margin: float = MIN_MARGIN,
           team_of_track=None, position_of_track=None,
           cross_team_penalty: float = CROSS_TEAM_PENALTY,
           position_bonus: float = POSITION_BONUS) -> list[TrackIdentity]:
    """One-to-one assignment of tracks to players, by vote weight.

    ``on_field`` is the frame returned by
    :func:`~nfl_gsplat.identity.participation.players_on_play`.

    ``team_of_track`` optionally maps track -> team. Cross-team pairings are
    PENALISED by ``cross_team_penalty`` votes, not forbidden -- see that
    constant for the measurement that settled it.

    ``position_of_track`` optionally maps track -> a formation position ("C",
    "CB", ...). Agreeing with the player's roster position is rewarded by
    ``position_bonus``. This is the stronger of the two signals: alignment is a
    geometric fact about where a body stood and covers every player on the
    field, whereas a jersey read is a low-probability guess that covered 9 of
    22. It is still a bonus rather than a veto -- a back can motion out and a
    lineman can report eligible, and a hard rule would turn those into losses
    the way the team veto did.
    """
    from scipy.optimize import linear_sum_assignment

    jerseys = [int(j) for j in on_field["jersey_number"]]
    tracks = sorted(votes)
    if not tracks:
        _LOG.warning("jersey assignment: no track had a usable read")
        return []

    # Cost is negative vote count, so the solver maximises total agreement.
    # No-vote pairs cost ZERO, not a large penalty. A penalty looks like it
    # discourages bad matches, but linear_sum_assignment must match
    # min(tracks, jerseys) pairs regardless -- with 24 tracks against 22
    # jerseys almost everything gets paired -- so a large constant makes the
    # solver minimise the NUMBER of no-vote pairs instead of maximising votes.
    # Measured: it handed a track that read #70 eighteen times the jersey #0
    # on two votes, to spare another track from going unmatched.
    teams = [str(t) for t in on_field["team"]]
    groups = [POSITION_GROUP.get(str(p).upper(), str(p).upper())
              for p in on_field["position"]]
    cost = np.zeros((len(tracks), len(jerseys)))
    blocked = 0
    for i, track_id in enumerate(tracks):
        track_team = None if team_of_track is None else team_of_track.get(track_id)
        track_pos = (None if position_of_track is None
                     else position_of_track.get(track_id))
        for j, jersey in enumerate(jerseys):
            count = votes[track_id].get(jersey, 0)
            penalty = (cross_team_penalty
                       if track_team is not None and teams[j] != track_team
                       else 0.0)
            if penalty:
                blocked += 1
            bonus = 0.0
            if track_pos is not None:
                want = POSITION_GROUP.get(str(track_pos).upper())
                if want is not None and groups[j] == want:
                    bonus = position_bonus
            cost[i, j] = -float(count) + penalty - bonus
    if blocked:
        _LOG.info("team colour penalised %d of %d track/player pairings",
                  blocked, len(tracks) * len(jerseys))

    rows, cols = linear_sum_assignment(cost)
    out = []
    for i, j in zip(rows, cols):
        track_id, jersey = tracks[i], jerseys[j]
        counter = votes[track_id]
        got = counter.get(jersey, 0)

        # Alignment alone is DECISIVE when only one player on the field plays
        # that spot. Arizona fields one quarterback, one back and one tight end,
        # so a track the formation calls "QB" can only be Kyler Murray -- no
        # jersey read required, and demanding one would discard the strongest
        # evidence available. Where several players share a group (five linemen,
        # five defensive backs) alignment narrows but cannot decide, and the
        # vote thresholds still apply.
        # Re-fetch, do NOT reuse the loop variable from the cost matrix above:
        # that one holds whatever the LAST track happened to have, so every
        # decision here was made against another player's alignment. It read
        # correctly and silently did the wrong thing.
        this_pos = (None if position_of_track is None
                    else position_of_track.get(track_id))
        sole_candidate = False
        if this_pos is not None:
            want = POSITION_GROUP.get(str(this_pos).upper())
            if want is not None and groups[j] == want:
                sole_candidate = sum(1 for g in groups if g == want) == 1
        if sole_candidate:
            out.append(TrackIdentity(
                track_id=int(track_id), jersey=int(jersey),
                player=str(on_field.iloc[j]["full_name"]),
                team=str(on_field.iloc[j]["team"]),
                height_m=float(on_field.iloc[j]["height_m"]),
                votes=int(got), margin=float("inf")))
            continue

        if got < min_votes:
            continue
        # The runner-up competes on the same terms the solver used: a read for
        # the other team is discounted by the colour penalty rather than
        # ignored, so a colour mistake cannot silently manufacture a tie.
        others = []
        for other, count in counter.items():
            if other == jersey or other not in set(jerseys):
                continue
            adj = count
            if team_of_track is not None and team_of_track.get(track_id) is not None:
                if str(row_team(on_field, other)) != team_of_track[track_id]:
                    adj = max(0.0, count - cross_team_penalty)
            others.append(adj)
        runner_up = max(others) if others else 0
        margin = got / runner_up if runner_up else float("inf")
        if margin < min_margin:
            _LOG.info("track %d: %d votes for %d but runner-up has %d "
                      "(margin %.2f) -- left unassigned", track_id, got,
                      jersey, runner_up, margin)
            continue
        row = on_field[on_field["jersey_number"] == jersey].iloc[0]
        out.append(TrackIdentity(
            track_id=int(track_id), jersey=int(jersey),
            player=str(row["full_name"]), team=str(row["team"]),
            height_m=float(row["height_m"]), votes=int(got),
            margin=float(margin)))

    _LOG.info("jersey assignment: %d/%d tracks identified against %d players "
              "on the field", len(out), len(tracks), len(jerseys))
    return sorted(out, key=lambda t: (t.team, t.jersey))
