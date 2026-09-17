"""Foot lock (render.foot_lock): stance detection on a leg cycle and the leg solve that pins a foot."""
import numpy as np
import pytest

from nfl_gsplat.render import foot_lock as fl


def test_stance_segments_find_the_slow_foot_of_a_moving_body_and_ignore_a_standing_one():
    T = 40
    pel = np.stack([0.1 * np.arange(T), np.zeros(T)], axis=1)                 # 0.1 m/frame along x
    # the ankle cycles: it stands still (relative to the world) on 8-11 and 24-27, swings otherwise
    ank = pel.copy()
    for a, b in ((8, 11), (24, 27)):
        ank[a:b + 1] = ank[a]                                                # planted: no world motion
    segs = fl.stance_segments(pel, ank)
    assert segs == [(8, 11), (24, 27)] or all(8 <= s[0] <= 9 and 10 <= s[1] <= 12 for s in segs[:1])
    # a standing body has no stance to find
    still = np.zeros((T, 2))
    assert fl.stance_segments(still, still) == []
    # a long dip is cut to max_stance frames around its slowest point
    ank2 = pel.copy()
    ank2[5:25] = ank2[5]
    segs2 = fl.stance_segments(pel, ank2, max_stance=8)
    assert len(segs2) == 1 and segs2[0][1] - segs2[0][0] + 1 == 8


@pytest.mark.skipif(not __import__("pathlib").Path("data/body_models/smplx/SMPLX_NEUTRAL.npz").exists(),
                    reason="SMPL-X model not present")
def test_solve_leg_pins_the_ankle_and_leaves_the_rest_of_the_pose_alone():
    rest, parents = fl.load_smplx_skeleton("data/body_models", betas=np.zeros(10))
    bp = np.zeros((21, 3))
    bp[0] = [0.4, 0.0, 0.0]                      # left hip flexed forward
    bp[3] = [0.6, 0.0, 0.0]                      # left knee bent
    go = np.array([np.pi / 2, 0.0, 0.0])         # stood up: +z world up (the fit's convention on this rig)
    pelvis_xy = np.array([5.0, 2.0])
    J = fl.relative_joints(bp, go, rest, parents)
    ankle0 = pelvis_xy + J[7, :2]
    target = ankle0 + np.array([-0.15, 0.05])    # pin the foot 16 cm behind where the fit has it
    bp2, miss = fl.solve_leg(bp, go, rest, "L", target, pelvis_xy)
    assert miss < 0.01
    J2 = fl.relative_joints(bp2, go, rest, parents)
    assert np.linalg.norm(pelvis_xy + J2[7, :2] - target) < 0.01
    # everything but the left hip and knee rows is untouched, the right leg included
    keep = [i for i in range(21) if i not in (0, 3)]
    assert np.allclose(bp2[keep], bp[keep])
    assert np.allclose(J2[8], J[8])


@pytest.mark.skipif(not __import__("pathlib").Path("data/body_models/smplx/SMPLX_NEUTRAL.npz").exists(),
                    reason="SMPL-X model not present")
def test_foot_lock_sequence_removes_skate_on_a_cycling_leg():
    """A body moving 0.1 m/frame whose left leg swings back and forth: the world ankle speed dips
    when the swing runs against the motion; after the lock those frames' ankle stays on one point."""
    rest, parents = fl.load_smplx_skeleton("data/body_models", betas=np.zeros(10))
    T = 30
    go = np.array([np.pi / 2, 0.0, 0.0])         # stood up, facing world -y
    seq = []
    for t in range(T):
        bp = np.zeros((21, 3))
        bp[0] = [0.5 * np.sin(2 * np.pi * t / 15), 0.0, 0.0]       # left hip swinging fore and aft
        bp[3] = [0.3, 0.0, 0.0]
        seq.append((np.array([0.0, -0.1 * t]), bp, go))            # the body moves the way it faces
    pel, ank = fl.ankle_world_xy(seq, rest, parents)
    before = fl.stance_segments(pel, ank[:, 0])
    assert before, "the fixture must produce a stance dip"
    out, rep = fl.foot_lock_sequence(seq, np.zeros(10), "data/body_models")
    assert rep["segments"] >= 1 and rep["frames"] >= 2 and max(rep["miss_m"]) < 0.02
    seq2 = [(xy, out[t], go) for t, (xy, _bp, _go) in enumerate(seq)]
    _pel2, ank2 = fl.ankle_world_xy(seq2, rest, parents)
    a, b = before[0]
    spread_before = np.ptp(ank[a:b + 1, 0], axis=0).max()
    spread_after = np.ptp(ank2[a:b + 1, 0], axis=0).max()
    assert spread_after < 0.02 and spread_after < spread_before
