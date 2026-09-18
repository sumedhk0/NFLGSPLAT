import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.render import stance


def test_stance_body_pose_is_the_fit_at_zero_the_stance_at_one_and_between_halfway():
    rng = np.random.default_rng(3)
    bp = rng.normal(scale=0.3, size=(21, 3))
    assert np.allclose(stance.stance_body_pose(bp, 0.0), bp)
    assert np.allclose(stance.stance_body_pose(bp, 1.0), stance.UNDER_CENTRE)
    half = stance.stance_body_pose(bp, 0.5)
    for row in (0, 3, 15):
        a = Rotation.from_rotvec(bp[row]); b = Rotation.from_rotvec(stance.UNDER_CENTRE[row]); h = Rotation.from_rotvec(half[row])
        assert abs((a.inv() * h).magnitude() - 0.5 * (a.inv() * b).magnitude()) < 1e-6


def test_stance_weights_are_one_then_ramp_down_before_the_track_begins():
    w = stance.stance_weights(range(213, 377), blend=6)
    assert w[213] == 1.0 and w[370] == 1.0
    assert abs(w[371] - 6 / 7) < 1e-9 and abs(w[376] - 1 / 7) < 1e-9
    assert stance.stance_weights([]) == {}
    # the stance is a bent, soft-kneed, arms-down pose, not the rest pose
    assert stance.UNDER_CENTRE[3, 0] > 0.5 and stance.UNDER_CENTRE[0, 0] < -0.3 and stance.UNDER_CENTRE[15, 2] < -0.7
