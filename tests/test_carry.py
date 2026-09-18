import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.render import carry


def test_holder_weights_ramp_in_and_out_over_blend_frames_and_are_one_inside():
    frames = list(range(100, 130))
    held = set(range(110, 121))
    w = carry.holder_weights(frames, held, blend=4)
    assert all(f not in w for f in range(100, 110)) and all(f not in w for f in range(121, 130))
    assert w[110] == 0.25 and w[111] == 0.5 and w[112] == 0.75 and w[113] == 1.0
    assert w[120] == 0.25 and w[119] == 0.5 and w[116] == 1.0
    assert carry.holder_weights([], held) == {}


def test_carry_body_pose_is_the_fit_at_zero_and_the_carry_pose_at_one():
    rng = np.random.default_rng(0)
    bp = rng.normal(scale=0.4, size=(21, 3))
    same = carry.carry_body_pose(bp, 0.0)
    assert np.allclose(same, bp)
    full = carry.carry_body_pose(bp, 1.0)
    for row, target in carry.CARRY_ROWS.items():
        # the same rotation, whatever the rotation vector's representation
        assert np.degrees((Rotation.from_rotvec(full[row]) * Rotation.from_rotvec(target).inv()).magnitude()) < 1e-6
    for row in carry.WRIST_ROWS:
        assert np.allclose(full[row], 0.0)
    # rows the carry does not own are untouched
    other = [r for r in range(21) if r not in carry.CARRY_ROWS and r not in carry.WRIST_ROWS]
    assert np.allclose(full[other], bp[other])
    # halfway is between: the angle to each end is half the angle between the ends
    half = carry.carry_body_pose(bp, 0.5)
    a = Rotation.from_rotvec(bp[15]); b = Rotation.from_rotvec(carry.CARRY_ROWS[15]); h = Rotation.from_rotvec(half[15])
    assert abs((a.inv() * h).magnitude() - 0.5 * (a.inv() * b).magnitude()) < 1e-6


def test_throw_rows_run_carry_to_cocked_to_release_and_mirror_for_a_lefty():
    def same(a, b):
        return all(np.degrees((Rotation.from_rotvec(a[r]) * Rotation.from_rotvec(b[r]).inv()).magnitude()) < 1e-6 for r in a)
    assert same(carry.throw_rows(0.0), carry.CARRY_ROWS)
    assert same(carry.throw_rows(1.0), carry.COCKED_ROWS)
    assert same(carry.throw_rows(2.0), carry.RELEASE_ROWS)
    lefty = carry.throw_rows(1.0, "L")
    assert np.allclose(lefty[15], carry.COCKED_ROWS[16] * carry.MIRROR) and np.allclose(lefty[18], carry.COCKED_ROWS[17] * carry.MIRROR)


def test_throw_schedule_cocks_swings_and_fades():
    s = carry.throw_schedule(100, cock=4, swing=8, follow=4)
    assert min(s) == 88 and max(s) == 103                     # 104 would have weight 0
    assert s[88] == (0.25, 1.0) and s[91] == (1.0, 1.0)         # cocked by the end of the cock
    assert s[92][0] > 1.0 and s[100] == (2.0, 1.0)              # the release pose on the release frame
    assert s[101] == (2.0, 0.75) and s[103] == (2.0, 0.25)
    bp = np.zeros((21, 3)); bp[19] = [0.3, 0.0, 0.0]
    out = carry.throw_body_pose(bp, 2.0, 1.0)
    assert np.allclose(out[16], carry.RELEASE_ROWS[16]) and np.allclose(out[19], 0.0)
    assert np.allclose(carry.throw_body_pose(bp, 2.0, 0.0), bp)


def test_ball_at_hand_sits_beyond_the_wrist_along_the_forearm():
    j = np.zeros((22, 3)); j[19] = [0.0, 0.0, 1.0]; j[21] = [0.0, 0.0, 1.3]
    assert np.allclose(carry.ball_at_hand(j, "R", palm_m=0.1), [0.0, 0.0, 1.4])
    j[18] = [1.0, 0.0, 1.0]; j[20] = [1.4, 0.0, 1.0]
    assert np.allclose(carry.ball_at_hand(j, "L", palm_m=0.1), [1.5, 0.0, 1.0])


def test_ball_between_hands_is_the_wrist_midpoint_pushed_forward():
    j = np.zeros((22, 3))
    j[20] = [1.0, 2.0, 1.1]; j[21] = [1.2, 2.0, 1.1]
    xyz = carry.ball_between_hands(j, np.pi / 2, fwd_m=0.1)
    assert np.allclose(xyz, [1.1, 2.1, 1.1])
