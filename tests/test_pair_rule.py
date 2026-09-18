"""pair_rule.mispaired_ids: a two-camera id whose cameras disagree at the median is not one player."""
import numpy as np

from nfl_gsplat.render.pair_rule import mispaired_ids


def test_a_crossing_pair_is_flagged_and_an_honest_one_is_not():
    side = {f: {1: np.array([0.0, -10.0 + 0.5 * f]), 2: np.array([5.0, 5.0 + 0.02 * f])} for f in range(40)}
    end = {f: {1: np.array([0.0, 0.0]), 2: np.array([5.8, 5.4 + 0.02 * f])} for f in range(40)}
    bad = mispaired_ids(side, end)
    assert set(bad) == {1} and bad[1] > 4.0
    assert mispaired_ids(side, end, min_overlap=41) == {}


def test_mispaired_frames_removes_the_frames_common_mode_and_flags_the_odd_row():
    import numpy as np

    from nfl_gsplat.render.pair_rule import mispaired_frames

    side = {}; end = {}
    for f in range(500, 510):
        side[f] = {p: np.array([float(p), 0.0]) for p in (1, 2, 3, 4, 5)}
        # the endzone camera's depth wander: every body 1.2 m off in x on these frames
        end[f] = {p: np.array([float(p) + 1.2, 0.05]) for p in (1, 2, 3, 4, 5)}
        end[f][5] = np.array([5.0 + 1.2 + 2.5, 0.0])                 # id 5's endzone row is another man's
    got = mispaired_frames(side, end, max_m=1.0)
    assert set(got) == {(f, 5) for f in range(500, 510)}
    assert all(2.3 < r < 2.7 for r in got.values())
    # with too few pairs for a common mode the raw distance counts
    side2 = {600: {1: np.array([0.0, 0.0]), 2: np.array([3.0, 0.0])}}
    end2 = {600: {1: np.array([1.2, 0.0]), 2: np.array([4.2, 0.0])}}
    assert set(mispaired_frames(side2, end2, max_m=1.0)) == {(600, 1), (600, 2)}
    assert mispaired_frames(side2, end2, max_m=1.5) == {}


def test_place_from_refit_skips_vetoed_frames():
    import numpy as np

    from nfl_gsplat.render.play_timeline import place_from_refit

    ground = {10: {7: np.array([0.0, 0.0]), 8: np.array([5.0, 0.0])}}
    refit = {10: {7: {"transl": [0.5, 0.0, 0.0]}, 8: {"transl": [5.5, 0.0, 0.0]}}}
    out, shifts = place_from_refit(ground, refit, skip={(10, 7)})
    assert np.allclose(out[10][7], [0.0, 0.0]) and np.allclose(out[10][8], [5.5, 0.0]) and len(shifts) == 1
