"""Tests for inferring position from alignment.

Formation carries far more identity information than jersey pixels do, but it
fails in ways that look plausible -- a receiver labelled "running back" is still
a player standing on a field. These pin the rules that were wrong first.
"""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.identity.formation import (assign_offense_roles, parse_personnel,
                                           line_of_scrimmage_x, resolve_trench,
                                           split_sides)

OFFENSE = parse_personnel("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR")


def test_personnel_parses_to_eleven():
    assert sum(OFFENSE.values()) == 11
    assert OFFENSE["G"] == 2 and OFFENSE["QB"] == 1


def test_yard_line_on_own_side():
    """ARI 26 with ARI in possession is 26 yards from ARI's own goal line."""
    x = line_of_scrimmage_x("ARI 26", "ARI", own_goal_x_m=-45.720)
    assert x == pytest.approx(-45.720 + 26 * 0.9144, abs=1e-6)


def test_yard_line_on_the_opponents_side():
    x = line_of_scrimmage_x("SEA 30", "ARI", own_goal_x_m=-45.720)
    assert x == pytest.approx(45.720 - 30 * 0.9144, abs=1e-6)


def test_unparseable_yard_line_fails_loud():
    """Falling back to player density biased the line 1.1 m on real footage."""
    with pytest.raises(SetupError, match="ARI 26"):
        line_of_scrimmage_x("midfield", "ARI")


def _formation():
    """A shotgun set: 5 linemen on the ball, QB 5 m back behind the centre,
    a back beside him, three receivers wide, a tight end just outside the line.
    """
    los = 0.0
    pts = [(-0.3, -2.0), (-0.3, -1.0), (-0.2, 0.0), (-0.3, 1.0), (-0.3, 2.0),  # OL
           (-5.0, 0.0),                                                        # QB
           (-4.6, 1.6),                                                        # RB
           (-0.4, -9.0), (-0.4, 9.0), (-1.0, -6.0),                            # WR
           (-0.4, 3.5)]                                                        # TE
    return np.array(pts), los


def test_line_is_the_contiguous_run_not_the_closest_to_the_ball():
    """Split receivers line up ON the line too. Picking the five nearest the
    ball put a "tackle" nine metres wide on real footage and threaded the tight
    end through the middle of the line."""
    xy, los = _formation()
    roles = assign_offense_roles(xy, OFFENSE, los)
    line = [i for i, r in roles.items() if r in ("C", "G", "T")]
    assert len(line) == 5
    assert max(abs(xy[i, 1]) for i in line) <= 2.5, "a wide player joined the line"


def test_centre_sits_in_the_middle_of_the_line():
    xy, los = _formation()
    roles = assign_offense_roles(xy, OFFENSE, los)
    centre = [i for i, r in roles.items() if r == "C"]
    assert len(centre) == 1
    assert abs(xy[centre[0], 1]) < 0.5


def test_quarterback_is_deep_and_central():
    xy, los = _formation()
    roles = assign_offense_roles(xy, OFFENSE, los)
    qb = [i for i, r in roles.items() if r == "QB"][0]
    assert xy[qb, 0] < -3.0, "quarterback is not in the backfield"
    assert abs(xy[qb, 1]) < 2.0, "quarterback is not behind the centre"


def test_back_is_beside_the_quarterback():
    """Assigning the deepest remaining player put the back 14 m wide -- a
    receiver who happened to be deep."""
    xy, los = _formation()
    roles = assign_offense_roles(xy, OFFENSE, los)
    qb = [i for i, r in roles.items() if r == "QB"][0]
    rb = [i for i, r in roles.items() if r == "RB"]
    assert rb, "no back assigned in a formation that has one beside the QB"
    assert np.hypot(*(xy[rb[0]] - xy[qb])) < 4.0


def test_an_empty_backfield_leaves_the_back_unassigned():
    """Better a missing label than a receiver wearing the back's identity."""
    xy, los = _formation()
    xy[6] = (-0.4, -12.0)                     # motion the back out wide
    roles = assign_offense_roles(xy, OFFENSE, los)
    assert "RB" not in roles.values()


def test_trench_goes_to_whichever_side_is_short():
    """Both lines stand on the ball; three linemen fell inside the tolerance on
    real footage while the defence already had its eleven."""
    xy = np.zeros((22, 2))
    xy[:8, 0] = -3.0            # clearly offence
    xy[8:19, 0] = +3.0          # clearly defence (11)
    xy[19:, 0] = -0.1           # in the trench
    off, dfn, tren = split_sides(xy, 0.0)
    assert len(tren) == 3
    off, dfn = resolve_trench(off, dfn, tren, xy, 0.0)
    assert len(off) == 11 and len(dfn) == 11
