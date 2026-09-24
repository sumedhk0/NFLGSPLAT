"""The running gait (render.gait): the cycle plants the foot by construction, the phase follows the
distance, the blend hands the legs back to the fit, and a standing body is untouched."""
import numpy as np
import pytest

from nfl_gsplat.render import gait


def test_leg_angles_cycle_between_strike_and_swing():
    amp = 0.4
    hip0, knee0 = gait.leg_angles(0.0, amp)                     # foot strike: hip forward, knee near straight
    assert abs(hip0 - amp) < 1e-9 and abs(knee0 - gait.KNEE_STANCE) < 1e-9
    hip_end, _ = gait.leg_angles(2 * np.pi * gait.DUTY - 1e-6, amp)   # end of stance: hip back
    assert abs(hip_end + amp) < 1e-3
    mid = 2 * np.pi * gait.DUTY + 0.5 * (2 * np.pi - 2 * np.pi * gait.DUTY)
    _hip_mid, knee_mid = gait.leg_angles(mid, amp)              # mid-swing: the knee at its peak
    assert abs(knee_mid - gait.KNEE_SWING) < 1e-6
    hip_back, _ = gait.leg_angles(2 * np.pi - 1e-6, amp)         # back to strike
    assert abs(hip_back - amp) < 1e-3


def test_stance_sweep_plants_the_foot_to_first_order():
    """Over a stance the hip sweeps 2 A radians while the body travels d L metres: with A = d L / (2 l)
    the foot's forward position (l sin(hip) + travel) stays put to first order."""
    L, d, l = 2.0, gait.DUTY, gait.LEG_M
    amp = np.arcsin(d * L / (2 * l))
    n = 20
    drift = []
    for k in range(n + 1):
        phase = 2 * np.pi * d * k / n
        travel = d * L * k / n
        hip, _ = gait.leg_angles(phase, amp)
        drift.append(l * np.sin(hip) + travel)
    assert np.ptp(drift) < 1e-9                                  # planted exactly in the sagittal plane


def test_phase_follows_forward_travel_and_holds_when_slow():
    speed = np.array([0.0, 0.1, 0.1, 0.1, 0.0, 0.0])
    adv = np.array([0.0, 0.1, 0.1, -0.1, 0.0, 0.0])            # forward, forward, then a step backwards
    phi, on = gait.phases(adv, speed, off_share=1.0, min_run=1, min_on=1)      # the plain threshold: this test is about the phase
    assert on.tolist() == [False, True, True, True, False, False]
    L = gait.stride_length(0.1)
    assert abs(phi[2] - 2 * np.pi * 0.2 / L) < 1e-9 and abs(phi[3] - 2 * np.pi * 0.1 / L) < 1e-9
    assert 2.0 < gait.stride_length(0.06) < 3.0 and 4.0 < gait.stride_length(0.15) < 5.0   # jog ~2.4 m, sprint ~4.6 m per cycle
    assert gait.duty_share(0.02) > gait.duty_share(0.15) >= 0.22
    assert phi[4] == phi[3] and phi[5] == phi[3]


def test_blend_weights_ramp_at_the_edges():
    on = np.array([False] * 3 + [True] * 12 + [False] * 3)
    w = gait.blend_weights(on, blend=4)
    assert w[:3].tolist() == [0, 0, 0] and w[-3:].tolist() == [0, 0, 0]
    assert np.allclose(w[3:7], [0.25, 0.5, 0.75, 1.0]) and np.allclose(w[11:15], [1.0, 0.75, 0.5, 0.25])
    assert np.all(w[7:11] == 1.0)


def test_gait_sequence_runs_the_legs_of_a_runner_and_leaves_a_standing_man():
    go = np.array([np.pi / 2, 0.0, 0.0])                        # stood up, facing world -y
    T = 40
    run = [(np.array([0.0, -0.12 * t]), np.zeros((21, 3)), go) for t in range(T)]     # 7 m/s along its facing
    out, rep = gait.gait_sequence(run)
    assert rep["on"] == T and 0.9 < rep["cycles"] < 1.5     # 40 frames at 7 m/s = 4.7 m, a cycle and a bit
    hips = out[:, 0, 0]
    assert hips.min() < -0.2 and hips.max() > 0.2                # the left hip swings both ways
    assert np.all(out[:, 3, 0] >= gait.KNEE_STANCE - 1e-9) and out[:, 3, 0].max() > 1.0   # the knee bends in swing
    keep = [2, 5] + list(range(8, 21))
    assert np.allclose(out[:, keep], 0.0)                          # spine, feet joints and arms untouched (hips, knees, ankles are the gait's)
    # the two legs are half a cycle apart: somewhere in the run they differ by most of the swing
    assert np.abs(out[:, 0, 0] - out[:, 1, 0]).max() > 0.4
    still = [(np.array([0.0, 0.0]), np.full((21, 3), 0.1), go) for _ in range(T)]
    out2, rep2 = gait.gait_sequence(still)
    assert rep2["on"] == 0 and np.allclose(out2, 0.1)


def test_leg_yaw_turns_the_legs_onto_the_motion_and_runs_a_backpedal_backwards():
    fwd = np.array([0.0, -1.0])                                 # facing -y
    yaw, adv = gait.leg_yaw(fwd, np.array([0.0, -0.1]))          # moving where it faces
    assert abs(yaw) < 1e-9 and abs(adv - 0.1) < 1e-9
    yaw, adv = gait.leg_yaw(fwd, np.array([0.1, 0.0]))           # moving to +x: facing -y, +x is the body's LEFT
    assert abs(yaw - np.pi / 2) < 1e-9 and abs(adv - 0.1) < 1e-9
    yaw, adv = gait.leg_yaw(fwd, np.array([0.0, 0.1]))           # moving backwards
    assert abs(abs(yaw) - 0.0) < 1e-9 and abs(adv + 0.1) < 1e-9
    # a hip flexed forward in a plane turned 90 deg to the left swings the knee toward +x of the pelvis
    from scipy.spatial.transform import Rotation
    r = Rotation.from_rotvec(gait.hip_rotvec(0.5, np.pi / 2))
    thigh = r.apply([0.0, -1.0, 0.0])                           # the thigh hangs down -y in the rest pose
    assert thigh[0] > 0.4 and abs(thigh[2]) < 1e-6


def test_gait_sequence_plants_along_the_velocity_when_the_body_faces_across_it():
    """A body facing -y but moving along +x at 7 m/s: the stance ankle must stand still in the WORLD."""
    from scipy.spatial.transform import Rotation as R_
    from nfl_gsplat.pose.forward_kinematics import load_smplx_skeleton, pose_params_to_rotmats, posed_joint_positions
    import pathlib
    if not pathlib.Path("data/body_models/smplx/SMPLX_NEUTRAL.npz").exists():
        pytest.skip("SMPL-X model not present")
    rest, parents = load_smplx_skeleton("data/body_models", betas=np.zeros(10))
    go = np.array([np.pi / 2, 0.0, 0.0])                        # stood up, facing world -y
    T = 40
    seq = [(np.array([0.12 * t, 0.0]), np.zeros((21, 3)), go) for t in range(T)]   # moving +x, across the facing
    out, rep = gait.gait_sequence(seq)
    assert rep["on"] == T
    ank = []
    for t in range(T):
        J = posed_joint_positions(rest, parents, pose_params_to_rotmats(go, out[t]))
        J = J - J[0]
        ank.append(seq[t][0] + J[7, :2])
    ank = np.array(ank)
    v = np.linalg.norm(ank[2:] - ank[:-2], axis=1) / 2
    assert v.min() < 0.03 and (v < 0.04).mean() > 0.15           # the left foot plants for a share of the run


def test_stance_foot_stays_level_with_the_turf():
    """Under the gait the stance foot (ankle to foot joint) stays near horizontal whatever the hip and
    knee do; a flexed swing foot is allowed to point down."""
    import pathlib
    if not pathlib.Path("data/body_models/smplx/SMPLX_NEUTRAL.npz").exists():
        pytest.skip("SMPL-X model not present")
    from nfl_gsplat.pose.forward_kinematics import load_smplx_skeleton, pose_params_to_rotmats, posed_joint_positions
    rest, parents = load_smplx_skeleton("data/body_models", betas=np.zeros(10))
    go = np.array([np.pi / 2, 0.0, 0.0])
    T = 40
    seq = [(np.array([0.0, -0.12 * t]), np.zeros((21, 3)), go) for t in range(T)]
    out, _ = gait.gait_sequence(seq)
    def pitch(bp):
        J = posed_joint_positions(rest, parents, pose_params_to_rotmats(go, bp))
        foot = J[10] - J[7]                                        # left ankle -> left foot joint
        return np.degrees(np.arctan2(foot[2], np.linalg.norm(foot[:2])))
    rest_pitch = pitch(np.zeros((21, 3)))                          # the foot joint sits below and ahead of the ankle at rest
    pitches = np.array([pitch(out[t]) for t in range(T)]) - rest_pitch
    # the flattest quarter of frames (stances) is within 8 deg of the rest pitch; the swing toes drop, never lift past 20
    assert np.sort(np.abs(pitches))[: T // 4].max() < 8.0 and pitches.max() < 20.0



def test_arms_swing_opposite_their_own_leg_and_hang_down_with_the_elbow_bent():
    from scipy.spatial.transform import Rotation

    sh0, el0 = gait.arm_rotvecs(0.0, "L")                       # the left foot strikes: the left arm is BACK
    shp, elp = gait.arm_rotvecs(np.pi, "L")                     # the left leg is back: the arm is forward
    d0 = Rotation.from_rotvec(sh0).apply([1.0, 0.0, 0.0])       # the upper arm's direction from the T-pose
    dp = Rotation.from_rotvec(shp).apply([1.0, 0.0, 0.0])
    assert d0[1] < -0.8 and dp[1] < -0.8                        # hanging down either way
    assert d0[2] < -0.2 and dp[2] > 0.2                         # back, then forward
    assert el0[1] < -1.0 and np.allclose(el0, elp)              # the left elbow flexed forward about -y
    shr, elr = gait.arm_rotvecs(0.0, "R")
    assert elr[1] > 1.0 and Rotation.from_rotvec(shr).apply([-1.0, 0.0, 0.0])[1] < -0.8
    # in gait_sequence (ARMS on) the arm rows follow the legs where the gait is on, and stay fitted where it is off
    xy = [(0.15 * t, 0.0) for t in range(40)]
    seq = [(np.array(p), np.zeros((21, 3)), np.zeros(3)) for p in xy]
    was = gait.ARMS
    try:
        gait.ARMS = True
        out, rep = gait.gait_sequence(seq)
        assert rep["on"] > 0 and abs(out[20, 17, 1]) > 0.5 and abs(out[20, 15]).max() > 0.5
        still = [(np.array([0.0, 0.0]), np.zeros((21, 3)), np.zeros(3)) for _ in range(40)]
        out2, rep2 = gait.gait_sequence(still)
        assert rep2["on"] == 0 and np.allclose(out2[:, 15:19], 0.0)
        gait.ARMS = False
        out3, _ = gait.gait_sequence(seq)
        assert np.allclose(out3[:, 15:19], 0.0)                     # off: the arms stay the fit's
    finally:
        gait.ARMS = was


def test_on_flags_hysteresis_and_minimum_run_kill_the_flicker():
    from nfl_gsplat.render import gait as _g
    t = np.arange(120)
    flicker = np.full(120, 0.08) + 0.006 * np.sin(t * 1.3)          # a jogger about the threshold
    plain = _g.on_flags(flicker, run_m=0.08, off_share=1.0, min_run=1, min_on=1)
    assert 5 < plain.sum() < 115 and np.count_nonzero(np.diff(plain)) > 10     # the plain threshold flickers
    on = _g.on_flags(flicker, run_m=0.08, off_share=0.75, min_run=12)
    first = int(np.argmax(flicker > 0.08))
    assert on[first:].all() and not on[:first].any()                 # one run from the first crossing
    stop = np.concatenate([np.full(40, 0.12), np.full(30, 0.02), np.full(40, 0.12)])   # a real stop
    on = _g.on_flags(stop, run_m=0.08, off_share=0.75, min_run=12)
    assert on[:40].all() and not on[40:70].any() and on[70:].all()
    dip = np.concatenate([np.full(40, 0.12), np.full(5, 0.02), np.full(40, 0.12)])     # a five-frame dip
    assert _g.on_flags(dip, run_m=0.08, off_share=0.75, min_run=12).all()
    spike = np.concatenate([np.full(40, 0.02), np.full(5, 0.12), np.full(40, 0.02)])   # a five-frame spike
    assert not _g.on_flags(spike, run_m=0.08, off_share=0.75, min_run=12).any()


def test_gait_sequence_knee_jerk_stays_low_on_a_flickering_speed():
    from nfl_gsplat.render import gait as _g, timeline as _tl
    speeds = np.full(120, 0.08) + 0.006 * np.sin(np.arange(120) * 1.3)
    xy = np.zeros((120, 2)); xy[:, 0] = np.cumsum(speeds)
    up = _tl.upright_from_yaw(0.0)
    seq = [(xy[i], np.zeros((21, 3)), up) for i in range(120)]
    def jerk(**kw):
        out, _ = _g.gait_sequence(seq, **kw)
        knee = np.degrees(out[::2, _g.KNEE_ROW["L"], 0])              # stride-2 sampling, as rendered
        return np.abs(np.diff(knee, 2)).max()
    assert jerk(off_share=1.0, min_run=1, min_on=1, blend=6) > 25     # the plain threshold snaps on the old 6-frame blend
    assert jerk(off_share=0.75, min_run=12, min_on=1, blend=6) < 25    # 49 -> 22: what is left is the switch-on ramp
    assert jerk(off_share=1.0, min_run=1) < 25                        # BLEND 16 alone cures the synthetic flicker too


def test_smooth_blend_meets_both_ends_without_a_corner():
    on = np.array([False] * 4 + [True] * 30 + [False] * 4)
    lin = gait.blend_weights(on, blend=8, shape="linear")
    sm = gait.blend_weights(on, blend=8, shape="smooth")
    assert np.allclose(sm[:4], 0) and np.allclose(sm[-4:], 0) and np.allclose(sm[12:26], 1)
    assert (np.diff(sm[3:13]) >= -1e-12).all() and (np.diff(sm[25:35]) <= 1e-12).all()      # monotone ramps
    assert sm[4] < lin[4] and sm[10] > lin[10]                                                # eased at both ends
    assert abs(sm[7] - 0.5) < 0.2 and np.allclose(sm, lin * lin * (3 - 2 * lin))
    import pytest
    with pytest.raises(ValueError):
        gait.blend_weights(on, blend=8, shape="bezier")


def test_gait_sequence_reads_blend_and_run_m_at_call_time(monkeypatch):
    from nfl_gsplat.render import timeline as _tl
    speeds = np.concatenate([np.full(20, 0.02), np.full(40, 0.12), np.full(20, 0.02)])
    xy = np.zeros((80, 2)); xy[:, 0] = np.cumsum(speeds)
    up = _tl.upright_from_yaw(0.0)
    seq = [(xy[i], np.zeros((21, 3)), up) for i in range(80)]
    monkeypatch.setattr(gait, "BLEND", 2)
    short, _ = gait.gait_sequence(seq)
    monkeypatch.setattr(gait, "BLEND", 16)
    long, _ = gait.gait_sequence(seq)
    knee_s = np.abs(short[:, gait.KNEE_ROW["L"], 0]); knee_l = np.abs(long[:, gait.KNEE_ROW["L"], 0])
    assert knee_s[21] > knee_l[21] and not np.allclose(short, long)          # the longer ramp is still low at frame 21
    monkeypatch.setattr(gait, "RUN_M", 0.5)                                  # nobody runs this fast: the gait stays off
    off, rep = gait.gait_sequence(seq)
    assert rep["on"] == 0 and np.allclose(off, 0.0)


def test_gait_smoothing_rounds_the_ramp_and_leaves_the_fit_alone():
    from nfl_gsplat.render import timeline as _tl
    speeds = np.concatenate([np.full(30, 0.02), np.full(40, 0.12), np.full(30, 0.02)])
    xy = np.zeros((100, 2)); xy[:, 0] = np.cumsum(speeds)
    up = _tl.upright_from_yaw(0.0)
    fit = np.zeros((21, 3)); fit[gait.KNEE_ROW["L"]] = [0.9, 0, 0]; fit[gait.ELBOW_ROW["L"]] = [0, -1.0, 0]
    seq = [(xy[i], fit, up) for i in range(100)]
    raw, _ = gait.gait_sequence(seq, blend=6, smooth_sigma=0.0)
    sm, _ = gait.gait_sequence(seq, blend=6, smooth_sigma=3.0)
    knee_r = raw[::2, gait.KNEE_ROW["L"], 0]; knee_s = sm[::2, gait.KNEE_ROW["L"], 0]
    assert np.abs(np.diff(knee_s, 2)).max() < np.abs(np.diff(knee_r, 2)).max()     # the corners are rounder
    assert np.allclose(sm[:, gait.ELBOW_ROW["L"]], fit[gait.ELBOW_ROW["L"]])         # arms untouched
    assert np.allclose(sm[:10, gait.KNEE_ROW["L"]], fit[gait.KNEE_ROW["L"]])         # the fit's legs far from the gait untouched
    assert np.allclose(sm[-10:, gait.KNEE_ROW["L"]], fit[gait.KNEE_ROW["L"]])


def test_gait_smoothing_survives_a_sequence_shorter_than_its_reach():
    from nfl_gsplat.render import timeline as _tl
    xy = np.zeros((10, 2)); xy[:, 0] = np.cumsum(np.full(10, 0.12))
    seq = [(xy[i], np.zeros((21, 3)), _tl.upright_from_yaw(0.0)) for i in range(10)]
    out, rep = gait.gait_sequence(seq, blend=6, smooth_sigma=3.0)
    assert out.shape == (10, 21, 3) and rep["on"] == 10


def test_min_on_drops_short_on_runs_and_never_fills_off_gaps():
    stop = np.concatenate([np.full(40, 0.12), np.full(5, 0.02), np.full(40, 0.12)])     # a five-frame dip
    on = gait.on_flags(stop, run_m=0.08, off_share=1.0, min_run=1, min_on=12)
    assert on[:40].all() and not on[40:45].any() and on[45:].all()                    # the gap stays a gap
    spike = np.concatenate([np.full(40, 0.02), np.full(5, 0.12), np.full(40, 0.02)])   # a five-frame crossing
    assert not gait.on_flags(spike, run_m=0.08, off_share=1.0, min_run=1, min_on=12).any()
    assert gait.on_flags(spike, run_m=0.08, off_share=1.0, min_run=1, min_on=1)[40:45].all()
    edge = np.concatenate([np.full(5, 0.12), np.full(40, 0.02)])                        # a short run at the edge stays
    assert gait.on_flags(edge, run_m=0.08, off_share=1.0, min_run=1, min_on=12)[:5].all()


def test_phase_match_finds_the_phase_whose_legs_match():
    amp, d = 0.5, 0.4
    for target in (0.3, 1.7, 3.1, 4.6, 5.9):
        hl, _ = gait.leg_angles(target, amp, duty=d)
        hr, _ = gait.leg_angles(target + np.pi, amp, duty=d)
        p = gait.phase_match(hl, hr, amp, duty=d, grid=96)
        assert abs(((p - target + np.pi) % (2 * np.pi)) - np.pi) < 2 * np.pi / 96 + 1e-9
    hip = gait.hip_rotvec(0.4, 0.0)                                                      # the gait's own hip at 0.4 rad forward
    assert abs(gait.fit_hip_flexion(hip) - 0.4) < 1e-6


def test_phase_matching_starts_the_gait_at_the_fits_legs():
    from nfl_gsplat.render import timeline as _tl
    speeds = np.concatenate([np.full(30, 0.02), np.full(50, 0.12)])
    xy = np.zeros((80, 2)); xy[:, 0] = np.cumsum(speeds)
    up = _tl.upright_from_yaw(0.0)
    fit = np.zeros((21, 3))
    fit[gait.HIP_ROW["L"]] = gait.hip_rotvec(-0.45, 0.0)                                # left thigh BACK, right forward: far from phase 0
    fit[gait.HIP_ROW["R"]] = gait.hip_rotvec(0.45, 0.0)
    seq = [(xy[i], fit, up) for i in range(80)]
    plain, _ = gait.gait_sequence(seq, blend=1, phase_match_on=False)
    matched, _ = gait.gait_sequence(seq, blend=1, phase_match_on=True)
    t0 = 30
    def gap(out):
        return sum(abs(gait.fit_hip_flexion(out[t0, gait.HIP_ROW[s]]) - gait.fit_hip_flexion(fit[gait.HIP_ROW[s]])) for s in ("L", "R"))
    assert gap(matched) < 0.5 * gap(plain) and gap(matched) < 0.3


def test_leg_yaw_hysteresis_keeps_a_twisted_torso_running_forward():
    """The sprinter's tackler turned his torso to 75 deg off his run (v106, 491-493): with the plain threshold the
    legs ran backwards for three frames and the leg plane flipped. With the previous decision carried, the flip
    needs 90 + margin, and un-flipping needs 90 - margin; a true backpedal still flips."""
    def vel_at(deg):                                        # facing -y; the velocity turned `deg` toward the body's left
        a = np.radians(deg)
        return np.array([np.sin(a) * 0.1, -np.cos(a) * 0.1])
    fwd = np.array([0.0, -1.0])
    yaw, adv = gait.leg_yaw(fwd, vel_at(100))                                  # plain: past 90, backwards
    assert adv < 0
    yaw, adv = gait.leg_yaw(fwd, vel_at(100), backwards=False, margin_deg=30)  # carried: still forward under 120
    assert adv > 0 and abs(np.degrees(yaw) - 100) < 1e-6
    yaw, adv = gait.leg_yaw(fwd, vel_at(125), backwards=False, margin_deg=30)  # past 120: backwards
    assert adv < 0
    yaw, adv = gait.leg_yaw(fwd, vel_at(75), backwards=True, margin_deg=30)    # backwards, 75 is not under 60: stays
    assert adv < 0
    yaw, adv = gait.leg_yaw(fwd, vel_at(50), backwards=True, margin_deg=30)    # under 60: forward again
    assert adv > 0
    yaw, adv = gait.leg_yaw(fwd, vel_at(170), backwards=False, margin_deg=30)  # a backpedal
    assert adv < 0 and abs(abs(np.degrees(yaw)) - 10) < 1e-6
    assert gait.leg_yaw(fwd, vel_at(100), backwards=False, margin_deg=0)[1] < 0  # margin 0 = the plain cliff


def test_gait_sequence_carries_the_backward_decision_along_a_run(monkeypatch):
    """A body running along +x whose fitted facing swings from 30 to 100 deg off the motion and back: the legs
    never flip. Measured on the hip rows of consecutive frames: with the cliff the leg plane turns 180 deg for the
    frames past 90; with the margin the hip rotation vectors change smoothly."""
    from scipy.spatial.transform import Rotation as R_
    T = 40
    base = R_.from_euler("x", -np.pi / 2)                    # tips the pelvis's +z (its forward) onto the ground
    f0 = base.apply([0.0, 0.0, 1.0])[:2]
    ang0 = np.arctan2(f0[1], f0[0])
    seq2 = []
    for t in range(T):
        off = 30 + 70 * np.sin(np.pi * t / (T - 1)) ** 2      # the facing 30 -> 100 -> 30 deg off the motion (+x)
        rot = R_.from_euler("z", np.radians(off) - ang0) * base
        seq2.append((np.array([0.12 * t, 0.0]), np.zeros((21, 3)), rot.as_rotvec()))
    for xy, bp, go in seq2:
        f = gait.forward_on_ground(go)
        assert f is not None
    monkeypatch.setattr(gait, "MIN_ON", 1)
    def max_hip_step(margin):
        monkeypatch.setattr(gait, "BACK_MARGIN_DEG", margin)
        out, rep = gait.gait_sequence(seq2, run_m=0.05, blend=1)
        assert rep["on"] == T
        hips = out[:, gait.HIP_ROW["L"]]
        return float(np.linalg.norm(np.diff(hips, axis=0), axis=1).max())
    assert max_hip_step(0.0) > 1.0            # the cliff: the leg plane flips
    assert max_hip_step(30.0) < 0.5           # the margin: no flip
