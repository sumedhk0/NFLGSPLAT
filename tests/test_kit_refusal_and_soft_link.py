"""saturation_split refuses one Gaussian; link3d's soft label penalty keeps a mislabelled box on its own track."""
import numpy as np
import pytest

from nfl_gsplat.identity import team_color as tc
from nfl_gsplat.tracking import link3d


def test_saturation_split_refuses_a_single_cluster_and_accepts_two_kits():
    rng = np.random.default_rng(0)
    one = rng.normal(45, 14, 400)                       # both teams in white
    with pytest.raises(tc.KitSplitError, match="not two kits"):
        tc.saturation_split(one)
    two = np.concatenate([rng.normal(44, 17, 300), rng.normal(141, 24, 250)])
    c, lab, sep = tc.saturation_split(two)
    assert sep > 1.6 and c[1] - c[0] > 50
    assert lab[:300].mean() < 0.05 and lab[300:].mean() > 0.95


def test_split_by_saturation_votes_refuses_per_camera():
    rng = np.random.default_rng(1)
    keys = [("sideline", i % 5) for i in range(200)]
    with pytest.raises(tc.KitSplitError, match="sideline"):
        tc.split_by_saturation_votes({"sideline": (keys, rng.normal(40, 10, 200))})


def _one_player_with_a_mislabelled_frame():
    """A player walking along x at 60 Hz, red every frame but one; a white
    player 0.8 m to the side the whole time."""
    fps = 60.0
    placements, labels = {}, {}
    for f in range(40):
        x = 0.05 * f
        placements[f] = np.array([[x, 0.0], [x, 0.8]])
        labels[f] = np.array([1, 0]) if f != 20 else np.array([0, 0])   # frame 20: red box read as white
    return placements, labels, fps


def test_hard_gate_breaks_the_track_and_the_soft_penalty_does_not():
    placements, labels, fps = _one_player_with_a_mislabelled_frame()
    hard = link3d.link(placements, labels=labels, fps=fps, min_frames=3)
    soft = link3d.link(placements, labels=labels, fps=fps, min_frames=3, label_penalty_m=1.0)

    # soft: two players, two tracks, the red one unbroken across frame 20
    assert sorted(len(t.frames) for t in soft) == [40, 40]
    # hard: the red track cannot take frame 20's box; it survives the gap but
    # the frame is lost to it
    assert sorted(len(t.frames) for t in hard) == [39, 40]
    red = [t for t in hard if 20 not in t.frames]
    assert len(red) == 1 and red[0].label == 1
