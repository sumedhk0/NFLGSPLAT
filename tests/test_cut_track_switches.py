"""08t: a track that switches men is cut at the switch; continuous motion and long gaps are left alone."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "cuts", Path(__file__).resolve().parents[1] / "scripts" / "08t_cut_track_switches.py")
cuts = importlib.util.module_from_spec(_spec)
sys.modules["cuts"] = cuts
_spec.loader.exec_module(cuts)


def _still(frames, xy):
    return {f: np.array(xy, float) for f in frames}


def test_a_same_camera_switch_across_a_short_gap_is_cut_before_the_gap():
    # play 1's id 82: sideline at x=-23.1 to 307, resumes at 315 at x=-28.9 -- 5.9 m in 8 frames
    S = {82: {**_still(range(217, 308), (-23.1, -1.0)), **_still(range(315, 400), (-28.9, 0.0))}}
    plan = cuts.plan_cuts(S, {})
    assert [(p, f, k) for p, f, k, _ in plan] == [(82, 307, "switch")]


def test_a_sprinter_is_not_a_switch():
    # 0.18 m/frame straight-line run, under the 0.20 bound, over a 60-frame stretch
    S = {5: {f: np.array([-30.0 + 0.18 * (f - 300), 0.0]) for f in range(300, 360)}}
    assert cuts.plan_cuts(S, {}) == []


def test_a_long_gap_is_not_a_switch_because_it_is_never_drawn():
    # 8 m apart but 60 frames of nothing in between: fill_gaps will not bridge it, so nothing teleports
    S = {9: {**_still(range(100, 120), (0.0, 0.0)), **_still(range(180, 200), (8.0, 0.0))}}
    assert cuts.plan_cuts(S, {}, max_gap=30) == []


def test_one_frame_of_box_jitter_is_not_a_switch():
    S = {3: {**_still(range(100, 120), (0.0, 0.0)), 120: np.array([0.6, 0.0]),
             **_still(range(121, 140), (0.0, 0.0))}}
    assert cuts.plan_cuts(S, {}) == []          # 0.6 m is under MIN_JUMP_M; never a real man swap


def test_a_welded_tail_is_a_candidate_cut_at_the_sideline_end():
    # play 1's id 19: sideline 14-430, endzone 483-639, no overlap
    S = {19: _still(range(14, 431), (-24.4, -2.8))}
    E = {19: _still(range(483, 640), (-33.0, 3.4))}
    plan = cuts.plan_cuts(S, E)
    assert [(p, f, k) for p, f, k, _ in plan] == [(19, 430, "tail")]


def test_overlapping_cameras_are_not_a_tail():
    S = {21: _still(range(26, 244), (-27.1, -4.0))}
    E = {21: _still(range(140, 648), (-27.3, -4.1))}
    assert cuts.plan_cuts(S, E) == []


def test_only_the_earliest_cut_per_id_is_kept():
    S = {7: {**_still(range(0, 50), (0.0, 0.0)), **_still(range(55, 100), (5.0, 0.0)),
             **_still(range(105, 150), (10.0, 0.0))}}
    plan = cuts.plan_cuts(S, {})
    assert [(p, f) for p, f, _k, _d in plan] == [(7, 49)]
