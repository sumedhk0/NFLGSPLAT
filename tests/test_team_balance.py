"""Forcing the two-team split toward the 11-v-11 the rules guarantee."""
import numpy as np

from nfl_gsplat.identity.team_color import (
    split_two_teams,
    split_two_teams_balanced,
)


def two_teams(n_a=11, n_b=11, seed=0, spread=6.0):
    rng = np.random.default_rng(seed)
    a = rng.normal([10, 200, 200], spread, size=(n_a, 3))
    b = rng.normal([110, 200, 200], spread, size=(n_b, 3))
    return np.vstack([a, b]), np.r_[np.zeros(n_a, int), np.ones(n_b, int)]


def accuracy(got, truth):
    agree = float((got == truth).mean())
    return max(agree, 1.0 - agree)      # cluster labels are arbitrary


def test_a_clean_split_is_left_alone():
    colors, truth = two_teams()
    got = split_two_teams_balanced(colors)
    assert accuracy(got, truth) == 1.0


def test_a_lopsided_split_is_rebalanced():
    """One team drifting toward the other must not swallow it."""
    rng = np.random.default_rng(3)
    a = rng.normal([10, 200, 200], 6.0, size=(11, 3))
    b = rng.normal([40, 200, 200], 30.0, size=(11, 3))   # smeared, overlapping
    colors = np.vstack([a, b])
    truth = np.r_[np.zeros(11, int), np.ones(11, int)]

    plain = split_two_teams(colors)
    balanced = split_two_teams_balanced(colors)
    plain_gap = abs(int((plain == 0).sum()) - int((plain == 1).sum()))
    bal_gap = abs(int((balanced == 0).sum()) - int((balanced == 1).sum()))
    assert bal_gap <= plain_gap
    assert bal_gap <= 2
    assert accuracy(balanced, truth) >= accuracy(plain, truth)


def test_too_few_tracks_falls_back():
    colors, _truth = two_teams(n_a=1, n_b=1)
    got = split_two_teams_balanced(colors)
    assert len(got) == 2
