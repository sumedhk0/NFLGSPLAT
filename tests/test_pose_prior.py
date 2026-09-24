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


def test_row_support_reads_each_bones_child_joint_and_the_body_median_elsewhere():
    rs = pp.row_support({13: 20.0, 9: 30.0, 16: 4.0}, body_median=7.0)
    assert rs[0] == 20.0 and rs[17] == 30.0 and rs[4] == 4.0          # L hip <- L knee, L elbow <- L wrist, R knee <- R ankle
    assert rs[2] == 7.0 and rs[11] == 7.0 and rs[1] == 7.0             # spine, neck, R hip (no R knee given): the median
    assert np.isnan(pp.row_support({}, body_median=np.nan)).all()


@pytest.mark.skipif(not _have_vposer(), reason="VPoser checkpoint or human_body_prior not present")
def test_per_row_support_moves_only_the_unsupported_rows():
    class S:
        def __init__(self, pid, bp):
            self.pid, self.body_pose = pid, bp
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.9, (10, 21, 3))                             # ten implausible frames of one man
    tl = type("T", (), {})(); tl.states = {f: [S(7, noise[f].copy())] for f in range(10)}
    vp = pp.load()
    # the arms off their keypoints (40 px), everything else on (2 px): only the arm rows may move
    rs = {(7, f): pp.row_support({7: 40.0, 8: 40.0, 9: 40.0, 10: 40.0, 13: 2.0, 14: 2.0, 15: 2.0, 16: 2.0}, 2.0) for f in range(10)}
    rep = pp.shift_timeline(tl, vp, lo=1.0, hi=2.0, sigma=0.0, row_support_by=rs)
    assert rep["moved"] == 10
    arm_rows = [15, 16, 17, 18]; other = [r for r in range(21) if r not in arm_rows]
    for f in range(10):
        bp = tl.states[f][0].body_pose
        assert np.abs(bp[arm_rows] - noise[f][arm_rows]).max() > 0.05
        assert np.allclose(bp[other], noise[f][other])


def test_row_ramps_free_an_arm_sooner_than_a_leg(monkeypatch):
    lo, hi = pp.row_ramps()
    assert lo[15] == pp.SUPPORT_ARM_LO and hi[15] == pp.SUPPORT_ARM_HI and lo[0] == pp.SUPPORT_LO and hi[0] == pp.SUPPORT_HI
    resid = np.full(21, 7.0)                                       # every row 7 px off its keypoint
    w = pp.support_weights(resid, lo=lo, hi=hi)
    assert w[15] == 0.5 and w[0] == 0.0                            # the arm half freed, the leg held
    monkeypatch.setattr(pp, "SUPPORT_ARM_LO", 8.0); monkeypatch.setattr(pp, "SUPPORT_ARM_HI", 16.0)
    lo2, hi2 = pp.row_ramps()
    assert pp.support_weights(resid, lo=lo2, hi=hi2)[15] == 0.0


@pytest.mark.skipif(not _have_vposer(), reason="VPoser checkpoint or human_body_prior not present")
def test_numpy_encoder_matches_torch():
    vp = pp.load()
    enc = pp.NumpyEncoder(vp)
    bps = np.random.default_rng(5).normal(0, 0.5, (16, 21, 3))
    z_np = enc(bps)
    z_t = pp.encode(vp, bps)
    assert z_np.shape == (16, 32) and np.abs(z_np - z_t).max() < 1e-4
    assert enc(bps[0]).shape == (32,) and np.allclose(enc(bps[0]), z_np[0])
    assert pp.encoder() is pp.encoder()                             # cached


def test_canonical_folds_wrapped_rotation_vectors_and_keeps_the_rotation():
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(3)
    bp = rng.normal(0, 0.6, (4, 21, 3))
    axis = bp[1, 0] / np.linalg.norm(bp[1, 0])
    bp[1, 0] = axis * (np.linalg.norm(bp[1, 0]) + 2 * np.pi)      # the same rotation, wrapped past pi
    bp[2, 5] = -bp[2, 5] / np.linalg.norm(bp[2, 5]) * (2 * np.pi - np.linalg.norm(bp[2, 5]))   # antipodal form
    c = pp.canonical(bp)
    assert c.shape == bp.shape and (np.linalg.norm(c.reshape(-1, 3), axis=1) <= np.pi + 1e-9).all()
    for i in range(4):
        for j in range(21):
            assert np.allclose(Rot.from_rotvec(bp[i, j]).as_matrix(), Rot.from_rotvec(c[i, j]).as_matrix(), atol=1e-9)
    assert np.allclose(c[0], bp[0]) and np.allclose(c[3], bp[3])            # already canonical: untouched
    assert np.allclose(pp.canonical(bp[1].reshape(63)), c[1].reshape(63))  # flat input keeps its shape


@pytest.mark.skipif(not _have_vposer(), reason="VPoser checkpoint or human_body_prior not present")
def test_the_prior_scores_a_wrapped_representation_like_its_canonical_form():
    vp = pp.load()
    bp = np.random.default_rng(4).normal(0, 0.4, (2, 21, 3))
    wrapped = bp.copy()
    wrapped[0, 0] *= (np.linalg.norm(bp[0, 0]) + 2 * np.pi) / np.linalg.norm(bp[0, 0])
    assert np.allclose(pp.scores(vp, wrapped), pp.scores(vp, bp), atol=1e-4)
    assert np.allclose(pp.encoder()(wrapped), pp.encoder()(bp), atol=1e-4)
    assert np.allclose(pp.blend(wrapped, pp.project(vp, wrapped), 0.0), pp.canonical(wrapped))
