"""pose.fill_unseen: a limb nobody saw takes the path between the poses either side."""
import numpy as np

from nfl_gsplat.pose.fill_unseen import fill_track, slerp_rotvec, unseen_runs


def test_runs_need_a_seen_frame_on_both_sides():
    assert unseen_runs(range(6), [True, False, False, True, False, True]) == [(0, 3), (3, 5)]
    assert unseen_runs(range(4), [False, False, True, True]) == []      # opens the track: nothing to come from
    assert unseen_runs(range(4), [True, True, False, False]) == []      # ends it: nothing to go to
    assert unseen_runs(range(3), [True, True, True]) == []


def test_the_interpolation_walks_between_the_ends():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, np.pi / 2])
    mid = slerp_rotvec(a, b, 0.5)
    assert np.allclose(mid, [0, 0, np.pi / 4], atol=1e-6)
    assert np.allclose(slerp_rotvec(a, b, 0.0), a, atol=1e-9)
    assert np.allclose(slerp_rotvec(a, b, 1.0), b, atol=1e-9)


def test_only_the_blacked_out_frames_move():
    T = 5
    bp = np.zeros((T, 63))
    j = 18                                            # the left elbow
    k = (j - 1) * 3
    bp[0, k:k + 3] = [0, 0, 0.0]
    bp[4, k:k + 3] = [0, 0, 1.0]
    bp[1:4, k:k + 3] = [0, 0, 5.0]                    # what the fit left there, unconstrained
    bp[:, 0:3] = 0.7                                  # another joint, seen throughout
    seen = {j: np.array([True, False, False, False, True])}
    out, n = fill_track(bp, seen)
    assert n == 3
    assert np.allclose(out[:, 0:3], 0.7)              # the seen joint is untouched
    assert np.allclose(out[0, k:k + 3], [0, 0, 0.0]) and np.allclose(out[4, k:k + 3], [0, 0, 1.0])
    assert 0.2 < out[1, k + 2] < 0.3 and 0.7 < out[3, k + 2] < 0.8


def test_a_long_blackout_is_left_alone():
    T = 60
    bp = np.zeros((T, 63))
    seen = {18: np.array([True] + [False] * 58 + [True])}
    out, n = fill_track(bp, seen, max_run=40)
    assert n == 0
