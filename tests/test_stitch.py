"""Tests for joining track fragments into players.

The dangerous failure is a WRONG join: welding one player's jersey votes onto
another's alignment produces a confident identity for a person who was never
there, and nothing downstream can detect it. So the refusals matter more than
the joins.
"""
from __future__ import annotations

from nfl_gsplat.tracking.stitch import MAX_SPEED_M_S, stitch, track_endpoints

FPS = 60.0


def _walk(track_id, f0, f1, x0, y0, x1=None, y1=None):
    n = f1 - f0 + 1
    x1 = x0 if x1 is None else x1
    y1 = y0 if y1 is None else y1
    return [(f0 + i, x0 + (x1 - x0) * i / max(1, n - 1),
             y0 + (y1 - y0) * i / max(1, n - 1)) for i in range(n)]


def test_endpoints_are_first_and_last_by_frame():
    pos = {7: list(reversed(_walk(7, 10, 20, 1.0, 2.0, 3.0, 4.0)))}
    ends = track_endpoints(pos)
    first, last, p0, p1 = ends[7]
    assert (first, last) == (10, 20)
    assert tuple(p0) == (1.0, 2.0) and tuple(p1) == (3.0, 4.0)


def test_a_short_close_gap_is_joined():
    """A player occluded for a few frames re-emerges nearby -- the case this
    exists for, since occlusion is routine when 22 bodies converge."""
    pos = {1: _walk(1, 0, 50, 0.0, 0.0, 5.0, 0.0),
           2: _walk(2, 56, 100, 5.5, 0.2, 9.0, 0.2)}
    got = stitch(pos, fps=FPS)
    assert got[1] == got[2], "an obvious continuation was not joined"


def test_a_distant_fragment_is_not_joined():
    """Two different players. Welding them would give one person's votes to the
    other, and nothing downstream could tell."""
    pos = {1: _walk(1, 0, 50, 0.0, 0.0),
           2: _walk(2, 56, 100, 40.0, 20.0)}
    got = stitch(pos, fps=FPS)
    assert got[1] != got[2]


def test_the_reach_scales_with_the_gap():
    """Distance alone is the wrong test: 8 m is impossible in a tenth of a
    second and easy in a second. A fixed radius is too tight for a receiver and
    too loose for a lineman."""
    near_in_time = {1: _walk(1, 0, 50, 0.0, 0.0), 2: _walk(2, 52, 90, 8.0, 0.0)}
    far_in_time = {1: _walk(1, 0, 50, 0.0, 0.0), 2: _walk(2, 100, 140, 8.0, 0.0)}
    assert stitch(near_in_time, fps=FPS)[1] != stitch(near_in_time, fps=FPS)[2]
    joined = stitch(far_in_time, fps=FPS)
    assert joined[1] == joined[2], "a reachable gap was refused"


def test_a_fragment_cannot_continue_into_two_players():
    """One body has one future. Letting two fragments claim the same
    continuation would merge three people into two."""
    pos = {1: _walk(1, 0, 50, 0.0, 0.0),
           2: _walk(2, 0, 50, 0.5, 0.0),
           3: _walk(3, 56, 100, 0.2, 0.0)}
    got = stitch(pos, fps=FPS)
    assert len(set(got.values())) == 2, "a continuation was claimed twice"


def test_joins_only_go_forward_in_time():
    pos = {1: _walk(1, 60, 100, 0.0, 0.0), 2: _walk(2, 0, 50, 0.2, 0.0)}
    got = stitch(pos, fps=FPS)
    assert got[2] == 2, "the earlier fragment should own the chain"


def test_a_chain_of_three_collapses_to_one_player():
    pos = {1: _walk(1, 0, 40, 0.0, 0.0, 2.0, 0.0),
           2: _walk(2, 46, 80, 2.3, 0.0, 4.0, 0.0),
           3: _walk(3, 86, 120, 4.3, 0.0, 6.0, 0.0)}
    got = stitch(pos, fps=FPS)
    assert len(set(got.values())) == 1


def test_empty_input_is_not_an_error():
    assert stitch({}) == {}
