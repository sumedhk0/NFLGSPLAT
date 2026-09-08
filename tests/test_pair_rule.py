"""pair_rule.mispaired_ids: a two-camera id whose cameras disagree at the median is not one player."""
import numpy as np

from nfl_gsplat.render.pair_rule import mispaired_ids


def test_a_crossing_pair_is_flagged_and_an_honest_one_is_not():
    side = {f: {1: np.array([0.0, -10.0 + 0.5 * f]), 2: np.array([5.0, 5.0 + 0.02 * f])} for f in range(40)}
    end = {f: {1: np.array([0.0, 0.0]), 2: np.array([5.8, 5.4 + 0.02 * f])} for f in range(40)}
    bad = mispaired_ids(side, end)
    assert set(bad) == {1} and bad[1] > 4.0
    assert mispaired_ids(side, end, min_overlap=41) == {}
