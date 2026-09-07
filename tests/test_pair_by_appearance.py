"""pair_by_appearance: number and kit decide before position; a conflict is a veto."""
import numpy as np

from nfl_gsplat.tracking.pair_by_appearance import CamTrack, global_ids, pair_by_appearance


def _track(cam, tid, x0, y0, kit, number, n=60, drift=(0.0, 0.0)):
    fr = np.arange(n)
    xy = np.column_stack([x0 + 0.03 * fr + drift[0], y0 + 0.0 * fr + drift[1]])
    return CamTrack(cam, tid, fr, xy, kit, number)


def test_number_match_pairs_at_three_metres_and_kit_only_needs_two():
    side = [_track("sideline", 0, 0.0, 0.0, 1, 15), _track("sideline", 1, 10.0, 0.0, 0, -1)]
    end = [_track("endzone", 0, 3.0, 0.0, 1, 15),        # same number, 3 m off: paired on the number
           _track("endzone", 1, 11.5, 0.0, 0, -1)]       # no number, same kit, 1.5 m: paired on kit + position
    pairs = pair_by_appearance(side, end)
    assert {(p.s, p.e, p.evidence) for p in pairs} == {(0, 0, "number"), (1, 1, "kit")}
    gs, ge = global_ids(2, 2, pairs)
    assert gs[0] == ge[0] and gs[1] == ge[1] and gs[0] != gs[1]


def test_kit_conflict_and_number_conflict_are_vetoes_even_when_nearest():
    side = [_track("sideline", 0, 0.0, 0.0, 1, 15)]
    end = [_track("endzone", 0, 0.2, 0.0, 0, -1),        # 0.2 m away but the other kit
           _track("endzone", 1, 0.3, 0.0, 1, 87)]        # same kit, a different number read
    assert pair_by_appearance(side, end) == []


def test_position_alone_never_pairs_and_time_disjoint_fragments_share_a_partner():
    side = [_track("sideline", 0, 0.0, 0.0, -1, -1)]
    end = [_track("endzone", 0, 0.1, 0.0, -1, -1)]
    assert pair_by_appearance(side, end) == []           # neither kit nor number known: no pair
    # two sideline fragments of one player (frames 0-59 and 60-119) both pair with one endzone track
    a = _track("sideline", 0, 0.0, 0.0, 1, -1)
    b = CamTrack("sideline", 1, np.arange(60, 120), np.column_stack([1.8 + 0.03 * np.arange(60, 120), np.zeros(60)]), 1, -1)
    e = CamTrack("endzone", 0, np.arange(120), np.column_stack([0.5 + 0.03 * np.arange(120), np.zeros(120)]), 1, -1)
    pairs = pair_by_appearance([a, b], [e])
    assert {(p.s, p.e) for p in pairs} == {(0, 0), (1, 0)}
    gs, ge = global_ids(2, 1, pairs)
    assert gs[0] == gs[1] == ge[0]


def test_stitch_joins_fragments_of_one_player_and_refuses_the_other_kit():
    from nfl_gsplat.tracking.pair_by_appearance import chains_from_joins, stitch_by_appearance

    a = CamTrack("sideline", 0, np.arange(0, 60), np.column_stack([0.03 * np.arange(60), np.zeros(60)]), 1, -1)
    b = CamTrack("sideline", 1, np.arange(75, 135), np.column_stack([1.8 + 0.03 * np.arange(60), np.zeros(60)]), 1, -1)   # 15 frames later, 0.03 m off the last point
    c = CamTrack("sideline", 2, np.arange(75, 135), np.column_stack([1.8 + 0.03 * np.arange(60), 0.2 + np.zeros(60)]), 0, -1)  # nearer still, other kit
    d = CamTrack("sideline", 3, np.arange(200, 260), np.column_stack([6.0 + np.zeros(60), np.zeros(60)]), 1, -1)             # 65 frames later: over the gap
    joins = stitch_by_appearance([a, b, c, d])
    assert [(i, j) for i, j, *_ in joins] == [(0, 1)]
    g = chains_from_joins(4, joins)
    assert g[0] == g[1] and len(set(g)) == 3
