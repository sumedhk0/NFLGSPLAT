"""08u: an interval where one camera tracks a different man is unpaired; jitter and coin flips are not."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "unpair", Path(__file__).resolve().parents[1] / "scripts" / "08u_unpair_bad_runs.py")
unpair = importlib.util.module_from_spec(_spec)
sys.modules["unpair"] = unpair
_spec.loader.exec_module(unpair)


def _still(frames, xy):
    return {f: np.array(xy, float) for f in frames}


def test_a_sideline_excursion_is_the_intruder():
    # play 1's id 17: endzone steady, four sideline frames of a man 6 m away
    E = {17: _still(range(440, 480), (-23.0, -1.4))}
    S = {17: {**_still(range(440, 457), (-23.1, -1.3)), **_still(range(457, 461), (-26.2, +3.9)),
              **_still(range(461, 480), (-23.0, -1.4))}}
    plan = unpair.plan_unpairings(S, E)
    assert len(plan) == 1
    pid, cam, lo, hi, _why = plan[0]
    assert (pid, cam, lo, hi) == (17, "sideline", 457, 460)


def test_an_endzone_excursion_is_the_intruder():
    S = {5: _still(range(380, 420), (-20.0, 2.0))}
    E = {5: {**_still(range(380, 389), (-20.2, 2.1)), **_still(range(389, 404), (-14.0, 4.0)),
             **_still(range(404, 420), (-20.1, 2.0))}}
    plan = unpair.plan_unpairings(S, E)
    assert [(p, c, lo, hi) for p, c, lo, hi, _ in plan] == [(5, "endzone", 389, 403)]


def test_a_coin_flip_does_not_fire():
    # both cameras drift 1.1 m from context in the run, neither is clearly the intruder: leave it alone
    S = {40: {**_still(range(400, 410), (0.0, 0.0)), **_still(range(410, 420), (0.8, 0.8)),
              **_still(range(420, 430), (0.0, 0.0))}}
    E = {40: {**_still(range(400, 410), (0.1, 0.0)), **_still(range(410, 420), (-0.8, -0.8)),
              **_still(range(420, 430), (0.1, 0.0))}}
    assert unpair.plan_unpairings(S, E) == []


def test_short_disagreement_is_jitter():
    S = {9: _still(range(300, 330), (0.0, 0.0))}
    E = {9: {**_still(range(300, 313), (0.0, 0.1)), **_still(range(313, 315), (5.0, 5.0)),
             **_still(range(315, 330), (0.0, 0.1))}}
    assert unpair.plan_unpairings(S, E, min_run=4) == []


def test_a_handover_metres_apart_unpairs_the_later_span():
    # play 1's id 82: sideline ends, endzone begins 8 frames later 5.9 m away
    S = {82: _still(range(280, 308), (-23.1, -1.0))}
    E = {82: _still(range(315, 340), (-28.9, 0.0))}
    plan = unpair.plan_unpairings(S, E)
    assert [(p, c, lo, hi) for p, c, lo, hi, _ in plan] == [(82, "endzone", 315, 339)]


def test_a_continuous_handover_is_left_alone():
    S = {21: _still(range(200, 244), (-27.1, -4.0))}
    E = {21: _still(range(246, 300), (-27.3, -4.1))}
    assert unpair.plan_unpairings(S, E) == []


def test_runs_of():
    flags = [False, True, True, True, True, False, True, True, False]
    frames = list(range(10, 19))
    assert unpair.runs_of(flags, frames, 4) == [(11, 14)]
    assert unpair.runs_of(flags, frames, 2) == [(11, 14), (16, 17)]
