"""VPoser as a ruler and a shift (pose.pose_prior). The model tests need the checkpoint and human_body_prior (the
smplx312 venv); the weight tests are numpy only."""
from pathlib import Path

import numpy as np
import pytest

from nfl_gsplat.pose import pose_prior as pp


def _have_vposer():
    try:
        import human_body_prior  # noqa: F401
    except Exception:
        return False
    return Path(pp.VPOSER_DIR, "snapshots").exists()


def test_weights_from_scores_spend_the_shift_on_the_tail_only():
    s = np.array([0.0, 5.0, 6.0, 7.0, 8.0, 12.0])
    w = pp.weights_from_scores(s, lo=6.0, hi=8.0)
    assert np.allclose(w, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
    # the module's defaults: nothing under three quarters of NORM_HI, everything at NORM_HI
    w2 = pp.weights_from_scores(np.array([0.0, pp.NORM_HI * 0.75, pp.NORM_HI]))
    assert np.allclose(w2, [0.0, 0.0, 1.0])


def test_blend_moves_each_pose_by_its_own_weight():
    bp = np.zeros((3, 21, 3)); pr = np.ones((3, 21, 3))
    out = pp.blend(bp, pr, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(out[0], 0) and np.allclose(out[1], 0.5) and np.allclose(out[2], 1)
    assert np.allclose(pp.blend(bp, pr, 0.25), 0.25)


@pytest.mark.skipif(not _have_vposer(), reason="VPoser checkpoint or human_body_prior not present")
def test_vposer_scores_the_mean_pose_low_and_noise_high_and_projects_onto_the_manifold():
    vp = pp.load()
    zero = np.zeros((1, 21, 3))
    noise = np.random.default_rng(0).normal(0, 0.8, (8, 21, 3))
    s0, sn = pp.scores(vp, zero), pp.scores(vp, noise)
    assert s0[0] < 5.0 and np.median(sn) > 2 * s0[0]          # measured: the rest pose 3.4, noise 10-16
    pr = pp.project(vp, noise)
    assert pr.shape == (8, 21, 3)
    # a projected pose sits closer to the manifold: it scores lower, and projecting it again moves it less than
    # the first projection did (the decoder is not exactly idempotent: 0.3 rad on the first round trip)
    again = pp.project(vp, pr)
    assert np.abs(again - pr).max() < np.abs(pr - noise).max()
    assert np.median(pp.scores(vp, pr)) < np.median(sn)


def test_smooth_weights_spread_a_lone_moved_frame_onto_its_neighbours():
    w = np.zeros(21); w[10] = 1.0
    sw = pp.smooth_weights(w, 2.0)
    assert sw[10] < 1.0 and sw[9] > 0.05 and sw[8] > 0.01 and sw[0] < 1e-3
    assert np.allclose(pp.smooth_weights(w, 0.0), w)
    assert sw.max() <= 1.0 and sw.min() >= 0.0


def test_support_weights_keep_film_backed_poses_still():
    w = pp.support_weights(np.array([2.0, 8.0, 12.0, 16.0, 30.0, np.nan]), lo=8.0, hi=16.0)
    assert np.allclose(w, [0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
