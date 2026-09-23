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
    out, rep = fl.foot_lock_sequence(seq, np.zeros(10), "data/body_models", mode="dips")
    assert rep["segments"] >= 1 and rep["frames"] >= 2 and max(rep["miss_m"]) < 0.02
    seq2 = [(xy, out[t], go) for t, (xy, _bp, _go) in enumerate(seq)]
    _pel2, ank2 = fl.ankle_world_xy(seq2, rest, parents)
    a, b = before[0]
    spread_before = np.ptp(ank[a:b + 1, 0], axis=0).max()
    spread_after = np.ptp(ank2[a:b + 1, 0], axis=0).max()
    assert spread_after < 0.02 and spread_after < spread_before


# ---- the rhythm mode ------------------------------------------------------------------------------------------

def _jogger(T=60, period=24, amp=0.35, step=0.06, knee=0.3):
    """A body jogging ``step`` m/frame the way it faces (world -y), the left hip swinging as a sinusoid of
    ``amp`` rad and ``period`` frames, the right half a cycle behind: the fitted-leg shape that plants a foot for an
    instant per cycle and skates the rest."""
    go = np.array([np.pi / 2, 0.0, 0.0])
    seq = []
    for t in range(T):
        bp = np.zeros((21, 3))
        ph = 2 * np.pi * t / period
        bp[0] = [-amp * np.sin(ph), 0.0, 0.0]          # left hip: forward flexion is about -x
        bp[1] = [amp * np.sin(ph), 0.0, 0.0]           # right hip, half a cycle behind
        bp[3] = [knee, 0.0, 0.0]; bp[4] = [knee, 0.0, 0.0]
        seq.append((np.array([0.0, -step * t]), bp, go))
    return seq, go


def test_strikes_read_the_flexion_maxima_and_ignore_a_flat_leg():
    t = np.arange(96)
    flex = 0.3 * np.sin(2 * np.pi * t / 24)
    st = fl.strikes(flex, sigma=0)
    assert st == [6, 30, 54, 78]
    assert fl.strikes(np.zeros(96)) == [] and fl.strikes(0.05 * np.sin(2 * np.pi * t / 24)) == []
    # two maxima closer than a cycle: the later yields
    assert fl.strikes(flex, sigma=0, min_cycle=30) == [6, 54]
    wins = fl.stance_windows(st, 96, duty=0.5)
    assert wins == [(6, 18), (30, 42), (54, 66), (78, 90)]
    assert fl.stance_windows([6], 96) == []


@pytest.mark.skipif(not __import__("pathlib").Path("data/body_models/smplx/SMPLX_NEUTRAL.npz").exists(),
                    reason="SMPL-X model not present")
def test_rhythm_mode_plants_the_stance_foot_of_a_jogger_and_leaves_the_swing():
    seq, go = _jogger()
    rest, parents = fl.load_smplx_skeleton("data/body_models", betas=np.zeros(10))
    st = fl.rhythm_stances(seq, rest, parents, sigma=0)
    assert st and all(s[2] > s[1] for s in st)
    sides = {s[0] for s in st}
    assert sides == {"L", "R"}
    out, rep = fl.foot_lock_sequence(seq, np.zeros(10), "data/body_models", mode="rhythm", sigma=0)
    assert rep["segments"] == len(st) and rep["frames"] > 0 and max(rep["miss_m"]) < 0.02
    seq2 = [(xy, out[t], go) for t, (xy, _bp, _go) in enumerate(seq)]
    _pel, ank = fl.ankle_world_xy(seq, rest, parents)
    _pel2, ank2 = fl.ankle_world_xy(seq2, rest, parents)
    for side, t0, t1, pin in st:
        li = 0 if side == "L" else 1
        mid = [t for t in range(t0, t1 + 1) if t - t0 >= fl.EDGE - 1 and t1 - t >= fl.EDGE - 1]
        if not mid:
            continue
        # inside the stance (past the edge ramps) the foot stands on the pin; over the whole stance it had moved
        after = np.ptp(ank2[mid, li], axis=0).max()
        assert after < 0.02
        assert np.ptp(ank[t0:t1 + 1, li], axis=0).max() > after
    # frames outside every stance are untouched
    touched = {t for _s, t0, t1, _p in st for t in range(t0, t1 + 1)}
    for t in range(len(seq)):
        if t not in touched:
            assert np.allclose(out[t], seq[t][1])
    # out of the band (a walk, a sprint) nothing happens
    for step in (0.02, 0.1):
        s2, _ = _jogger(step=step)
        o2, r2 = fl.foot_lock_sequence(s2, np.zeros(10), "data/body_models", mode="rhythm", sigma=0)
        assert r2["segments"] == 0 and np.allclose(o2, np.array([s[1] for s in s2]))


def test_rhythm_knobs_are_read_at_call_time(monkeypatch):
    seq, _go = _jogger()
    monkeypatch.setattr(fl, "MODE", "off")
    out, rep = fl.foot_lock_sequence(seq, np.zeros(10), "data/body_models")
    assert rep["segments"] == 0 and np.allclose(out, np.array([s[1] for s in seq]))
    t = np.arange(96); flex = 0.3 * np.sin(2 * np.pi * t / 24)
    monkeypatch.setattr(fl, "MIN_SWEEP", 1.0)
    assert fl.strikes(flex, sigma=0) == []
    monkeypatch.setattr(fl, "MIN_SWEEP", 0.15)
    monkeypatch.setattr(fl, "DUTY", 0.25)
    assert fl.stance_windows([6, 30, 54], 96) == [(6, 12), (30, 36), (54, 60)]
