import numpy as np
import pytest

from nfl_gsplat.pose import infill as inf


def _seq(pid, K=120, seed=0, noise=0.0):
    """A body whose rows swing on distinct sinusoids (a stride), one row-pair phase-locked (the knees follow the hips)."""
    rng = np.random.default_rng(seed)
    frames = np.arange(0, 2 * K, 2)
    t = np.arange(K) / 4.0                                                     # a stride every 25 keyframes
    pose = np.zeros((K, inf.J, 3))
    for j in range(inf.J):
        amp = 0.3 + 0.4 * ((j * 7) % 5) / 5.0
        pose[:, j, 0] = amp * np.sin(t + 0.3 * j)
        pose[:, j, 1] = 0.2 * amp * np.cos(0.5 * t + j)
    pose[:, 3] = 0.8 * pose[:, 0] + 0.1                                        # left knee follows the left hip
    pose[:, 4] = 0.8 * pose[:, 1] + 0.1
    pose += noise * rng.normal(size=pose.shape)
    conf = np.full((K, inf.J), 0.9)
    conf[:, [2, 5, 8, 9, 10, 11, 12, 13, 14]] = np.nan                          # the trunk: no keypoint vouches
    return inf.Sequence(pid, frames, pose, conf)


def test_masks_and_windows_have_the_right_shape():
    s = _seq(1)
    sure = inf.sure_mask(s.conf)
    assert sure[:, 0].all() and not sure[:, 2].any()
    train_m, test_m = inf.holdout_split([s], every=20, run=4)
    assert not (train_m[0] & test_m[0]).any()                                    # never both
    assert test_m[0][19:23, 15].all() and not test_m[0][18, 15] and not test_m[0][23, 15]   # the L arm, a run of 4 from keyframe 19
    assert not test_m[0][19:23, 16].any() and test_m[0][39:43, 16].all()         # the R arm's turn comes at the next run
    assert not test_m[0][:, 2].any()                                              # trunk rows are never sure
    hidden = inf.hide([s], test_m)
    assert np.isnan(hidden[0].conf[19, 15]) and hidden[0].conf[18, 15] == 0.9 and hidden[0].conf[19, 0] == 0.9
    win = inf.build_windows(hidden, train_m)
    W = 2 * inf.HALF + 1
    assert win.x.shape[1:] == (W, inf.J * 5) and win.y.shape[1:] == (inf.J, 3) and win.base.shape == win.y.shape
    i = 10
    centre = win.x[i, inf.HALF]
    masked = np.flatnonzero(win.mask[i])
    rows = centre[:inf.J * 3].reshape(inf.J, 3)
    assert np.allclose(rows[masked], win.base[i][masked])                             # the masked rows carry the SLERP fill
    assert not centre[inf.J * 3:inf.J * 4][masked].any()                              # ... read as not sure
    assert centre[inf.J * 4:][masked].all()                                           # ... and flagged masked


def test_slerp_fill_interpolates_between_the_kept_keyframes():
    frames = np.array([0, 2, 4, 6, 8])
    pose = np.zeros((5, inf.J, 3)); pose[:, 5, 2] = [0.0, 9.0, 9.0, 9.0, 1.0]         # garbage in between
    keep = np.ones((5, inf.J), bool); keep[1:4, 5] = False
    out = inf.slerp_fill(frames, pose, keep)
    assert np.allclose(out[2, 5], [0, 0, 0.5], atol=1e-6) and np.allclose(out[0, 5], 0.0) and np.allclose(out[4, 5], [0, 0, 1.0])


def test_geodesic_is_zero_for_the_same_rotation_and_90_for_a_quarter_turn():
    a = np.array([0.0, 0.0, 0.0]); b = np.array([np.pi / 2, 0.0, 0.0])
    assert abs(inf.geodesic_deg(a, a)[0]) < 1e-9 and abs(inf.geodesic_deg(a, b)[0] - 90) < 1e-6


@pytest.mark.slow
def test_holdout_never_loses_to_slerp_and_beats_it_over_a_long_hole():
    seqs = [_seq(p, seed=p, noise=0.01, K=240) for p in (1, 2, 3, 4)]
    short = inf.holdout_score(seqs, every=24, run=2, epochs=30, width=64)          # a two-keyframe hole: SLERP is near-perfect
    assert short["n"] > 100 and short["model_mean"] <= short["slerp_mean"] * 1.02   # the guard: never worse
    long = inf.holdout_score(seqs, every=24, run=8, epochs=30, width=64)           # an eight-keyframe hole: the stride curves
    assert long["n"] > 100 and long["model_mean"] < long["slerp_mean"] and long["best_epoch"] > 0
    filled = inf.fill(seqs, long["net"])
    assert set(filled) == {1, 2, 3, 4} and filled[1][0].shape == (inf.J, 3)
