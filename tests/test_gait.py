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
    phi, on = gait.phases(adv, speed)
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
    assert np.allclose(out[:, 5:], 0.0) and np.allclose(out[:, 2], 0.0)   # spine and arms untouched
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

