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


def assign(votes, on_field, *, min_votes: int = MIN_VOTES,
           min_margin: float = MIN_MARGIN) -> list[TrackIdentity]:
    """One-to-one assignment of tracks to players, by vote weight.

    ``on_field`` is the frame returned by
    :func:`~nfl_gsplat.identity.participation.players_on_play`.
    """
    from scipy.optimize import linear_sum_assignment

    jerseys = [int(j) for j in on_field["jersey_number"]]
    tracks = sorted(votes)
    if not tracks:
        _LOG.warning("jersey assignment: no track had a usable read")
        return []

    # Cost is negative vote count, so the solver maximises agreement. Pairs with
    # no votes at all get a large finite cost rather than inf: the solver must
    # stay feasible, and those pairings are rejected afterwards by the
    # thresholds instead of by making the problem unsolvable.
    cost = np.full((len(tracks), len(jerseys)), 1e3)
    for i, track_id in enumerate(tracks):
        for j, jersey in enumerate(jerseys):
            count = votes[track_id].get(jersey, 0)
            if count:
                cost[i, j] = -float(count)

    rows, cols = linear_sum_assignment(cost)
    out = []
    for i, j in zip(rows, cols):
        track_id, jersey = tracks[i], jerseys[j]
        counter = votes[track_id]
        got = counter.get(jersey, 0)
        if got < min_votes:
            continue
        others = [c for k, c in counter.items() if k != jersey]
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
