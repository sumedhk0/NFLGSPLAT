"""team_color.split_by_saturation_votes (rule D): per-camera two-means on
per-detection saturation, detection votes, track majority."""
import numpy as np
import pytest

from nfl_gsplat.identity import team_color as tc


def _play(rng, *, flip=0.2):
    """Two cameras exposing the same kits differently: the red kit reads
    S about 150 on the sideline and 110 in the endzone, the white one 40
    and 25. Each track has 30 detections per camera; ``flip`` of them are
    corrupted to the other kit's saturation (a helmet or an arm in the crop)."""
    truth = {t: (1 if t % 2 else 0) for t in range(12)}
    sats = {}
    for cam, (lo, hi) in {"sideline": (40.0, 150.0), "endzone": (25.0, 110.0)}.items():
        keys, s = [], []
        for t, red in truth.items():
            for _ in range(30):
                kit = red if rng.random() > flip else 1 - red
                s.append((hi if kit else lo) + rng.normal(0, 8))
                keys.append((cam, t))
        sats[cam] = (keys, np.array(s))
    return truth, sats


def test_two_means_1d_finds_the_two_kits():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(40, 8, 200), rng.normal(150, 8, 100)])
    c, lab = tc.two_means_1d(x)
    assert abs(c[0] - 40) < 4 and abs(c[1] - 150) < 4
    assert lab[:200].mean() < 0.02 and lab[200:].mean() > 0.98


def test_votes_recover_every_track_despite_flipped_detections():
    rng = np.random.default_rng(1)
    truth, sats = _play(rng, flip=0.2)
    labels, centres = tc.split_by_saturation_votes(sats)
    per_cam = {k: labels[k] for k in labels}
    for (cam, t), lab in per_cam.items():
        assert lab == truth[t], (cam, t)
    # the saturated label is the HIGHER centre in every camera
    for cam, (lo, hi) in centres.items():
        assert hi > lo + 50, (cam, lo, hi)


def test_one_global_split_would_mix_exposure_with_team_but_per_camera_does_not():
    """The endzone's red (110) sits between the sideline's white (40) and red
    (150): a single two-means over both cameras puts the endzone's red near
    the sideline's split point; the per-camera split does not care."""
    rng = np.random.default_rng(2)
    truth, sats = _play(rng, flip=0.0)
    all_s = np.concatenate([sats[c][1] for c in sats])
    c, _ = tc.two_means_1d(all_s)
    mid = 0.5 * (c[0] + c[1])
    assert 60 < mid < 120           # the global midpoint is not far from the endzone's red
    labels, _ = tc.split_by_saturation_votes(sats)
    assert all(labels[(cam, t)] == truth[t] for (cam, t) in labels)


def test_nan_abstains_and_too_few_detections_refuse():
    keys = [("sideline", 0)] * 10 + [("sideline", 1)] * 10
    s = np.array([150.0] * 10 + [40.0] * 10)
    s[3] = np.nan
    labels, _ = tc.split_by_saturation_votes({"sideline": (keys, s)})
    assert labels[("sideline", 0)] == tc.SATURATED and labels[("sideline", 1)] == 0
    with pytest.raises(ValueError, match="detections carry a torso colour"):
        tc.split_by_saturation_votes({"endzone": (keys[:4], np.array([1.0, 2.0, np.nan, 4.0]))})


def test_a_tie_goes_to_the_side_of_the_mean_margin():
    keys = [("sideline", 7)] * 2 + [("sideline", 8)] * 20 + [("sideline", 9)] * 20
    s = np.array([100.0, 141.0] + [40.0] * 20 + [150.0] * 20)     # track 7: one vote each way
    labels, centres = tc.split_by_saturation_votes({"sideline": (keys, s)})
    lo, hi = centres["sideline"]
    mid = 0.5 * (lo + hi)
    expect = tc.SATURATED if (100.0 - mid) + (141.0 - mid) > 0 else 0
    assert labels[("sideline", 7)] == expect
