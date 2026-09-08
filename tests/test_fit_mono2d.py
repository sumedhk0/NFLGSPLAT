"""fit_mono2d: a posed body projected into one camera is recovered from the mean pose, feet on the turf."""
import numpy as np

from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at
from nfl_gsplat.pose.fit_mono2d import Mono2DConfig, fit_sequence_2d, project
from nfl_gsplat.pose.forward_kinematics import fk_forward
from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig, _pack_params


def _rest():
    """A stick figure in SMPL-X body order (22 joints), metres, pelvis at 0.95 m."""
    r = np.zeros((22, 3))
    r[0] = [0, 0, 0.95]                                   # pelvis
    r[1], r[2] = [0.09, 0, 0.9], [-0.09, 0, 0.9]          # hips
    r[3], r[6], r[9] = [0, 0, 1.05], [0, 0, 1.18], [0, 0, 1.32]   # spine
    r[4], r[5] = [0.1, 0, 0.5], [-0.1, 0, 0.5]            # knees
    r[7], r[8] = [0.1, 0, 0.08], [-0.1, 0, 0.08]          # ankles
    r[10], r[11] = [0.1, 0.12, 0.0], [-0.1, 0.12, 0.0]    # feet
    r[12], r[15] = [0, 0, 1.45], [0, 0, 1.65]             # neck, head
    r[13], r[14] = [0.08, 0, 1.4], [-0.08, 0, 1.4]        # collars
    r[16], r[17] = [0.2, 0, 1.4], [-0.2, 0, 1.4]          # shoulders
    r[18], r[19] = [0.45, 0, 1.4], [-0.45, 0, 1.4]        # elbows
    r[20], r[21] = [0.7, 0, 1.4], [-0.7, 0, 1.4]          # wrists
    return r


def test_recovers_a_swinging_arm_from_one_view():
    rest = _rest()
    forward = fk_forward(rest)
    base = SMPLXFitConfig()
    K = intrinsics(1920, 1080, fov_deg=12.0)
    R, t = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 1.0]))
    cam = (K, R, t)
    # truth: standing at (2, 1), facing +x, right arm raised (elbow joint bends 1.2 rad)
    bp = np.zeros(63)
    bp[(17 - 1) * 3 + 1] = 1.2                             # right shoulder about y: lifts an arm that lies along x
    bp[(19 - 1) * 3 + 1] = 0.6                             # right elbow
    from scipy.spatial.transform import Rotation
    go = Rotation.from_euler("z", np.pi / 2).as_rotvec()
    tr = np.array([2.0, 1.0, 0.0])
    p_true = _pack_params(bp, go, tr)
    J = forward(p_true)
    z_shift = min(J[7, 2], J[8, 2])
    p_true[-1] -= z_shift
    J = forward(p_true)
    uv, _ = project(K, R, t, J)
    conf = np.ones(22)
    conf[[3, 6, 9, 13, 14, 10, 11]] = 0.0                 # the joints COCO never sees
    rng = np.random.default_rng(0)
    uv_noisy = uv + rng.normal(0, 1.0, uv.shape)
    T = 4
    # the regressor's rough pose is the prior that breaks the one-view tie between an
    # arm raised 0.45 m and one extended 1.1 m toward an elevated camera (they project alike)
    rough = np.stack([bp + rng.normal(0, 0.3, 63)] * T)
    # and its heading, 20 deg off: a symmetric body cannot tell front from back in one view
    rough_go = np.stack([(Rotation.from_euler("z", np.radians(20.0)) * Rotation.from_rotvec(go)).as_rotvec()] * T)
    params, valid, rep = fit_sequence_2d(np.stack([uv_noisy] * T), np.stack([conf] * T), [cam] * T,
                                         np.stack([J[0, :2]] * T), rest, forward,
                                         cfg=Mono2DConfig(up_axis=(0.0, 0.0, 1.0)),      # this stick figure is z-up
                                         base_cfg=base, init_body_pose_seq=rough, init_orient_seq=rough_go)
    assert valid.all()
    assert rep[-1] < 3.0, rep
    Jf = forward(params[-1])
    assert abs(min(Jf[7, 2], Jf[8, 2])) < 0.05                        # feet on the turf
    assert np.linalg.norm(Jf[0, :2] - J[0, :2]) < 0.3                  # placed on the ground point
    # the raised arm is raised as in the truth: the right wrist's height within 0.15 m of the truth's
    assert J[21, 2] > J[17, 2] + 0.2, "the synthetic arm must actually be raised"
    assert abs(Jf[21, 2] - J[21, 2]) < 0.15, (Jf[21], J[21])


def test_body_frame_speeds_ignore_orient_and_transl_and_merge_keeps_fused():
    from scipy.spatial.transform import Rotation

    from nfl_gsplat.pose.fit_mono2d import body_frame_speeds, merge_into_refit

    rest = _rest()
    forward = fk_forward(rest)
    bp = np.zeros(63)
    a = _pack_params(bp, np.zeros(3), np.zeros(3))
    # same pose, moved 5 m and turned: no body-frame motion
    b = _pack_params(bp, Rotation.from_euler("z", 1.0).as_rotvec(), np.array([5.0, 0, 0]))
    v = body_frame_speeds([a, b], [0, 6], forward, fps=60.0)
    assert v.shape == (1, 22) and v.max() < 1e-9
    # the arm swings 0.5 rad in 6 frames at 60 fps: the wrist moves, the pelvis does not
    bp2 = bp.copy()
    bp2[(17 - 1) * 3 + 1] = 0.5
    v = body_frame_speeds([a, _pack_params(bp2, np.zeros(3), np.zeros(3))], [0, 6], forward, fps=60.0)
    assert v[0, 21] > 1.0 and v[0, 0] < 1e-9
    blob = {"cam": "fused", "world": True, "frames": {10: {1: {"betas": np.zeros(10), "body_pose": np.ones(63),
                                                               "global_orient": np.zeros(3), "transl": np.zeros(3)}}}}
    fits = {1: (np.array([10, 12]), np.stack([a, b]), np.array([True, True])),
            2: (np.array([10, 12]), np.stack([a, b]), np.array([True, False]))}
    out, added = merge_into_refit(blob, fits, {1: np.zeros(10), 2: np.ones(10)})
    assert added == 2                          # (12, 1) and (10, 2); (10, 1) fused wins, (12, 2) invalid
    assert np.all(out["frames"][10][1]["body_pose"] == 1.0)
    assert np.allclose(out["frames"][12][1]["transl"], [5, 0, 0])
    assert 2 in out["frames"][10] and 2 not in out["frames"][12]
    assert out["mono"]["records"] == {1: 1, 2: 1}
    assert 2 not in blob["frames"][10]         # the input blob is not mutated


def test_prev_seq_anchors_a_frame_to_the_given_params():
    """With prev_seq the frame warm-starts from (and is pulled toward) the given
    solution instead of the previous fitted frame's."""
    rest = _rest()
    forward = fk_forward(rest)
    base = SMPLXFitConfig()
    K = intrinsics(1920, 1080, fov_deg=12.0)
    R, t = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 1.0]))
    cam = (K, R, t)
    from scipy.spatial.transform import Rotation
    go = Rotation.from_euler("z", np.pi / 2).as_rotvec()
    p0 = _pack_params(np.zeros(63), go, np.array([2.0, 1.0, 0.0]))
    J = forward(p0)
    p0[-1] -= min(J[7, 2], J[8, 2])
    J = forward(p0)
    uv, _ = project(K, R, t, J)
    conf = np.ones(22)
    conf[[3, 6, 9, 13, 14, 10, 11]] = 0.0
    T = 3
    cfg = Mono2DConfig(up_axis=(0.0, 0.0, 1.0))
    # the given anchor has the right arm raised; with no keypoints on that arm the fit keeps it
    anchor = p0.copy()
    anchor[(17 - 1) * 3 + 1] = 1.2
    conf[[17, 19, 21]] = 0.0
    params, valid, _ = fit_sequence_2d(np.stack([uv] * T), np.stack([conf] * T), [cam] * T, np.stack([J[0, :2]] * T),
                                       rest, forward, cfg=cfg, base_cfg=base, init_orient_seq=np.stack([go] * T),
                                       prev_seq=[None, anchor, None])
    assert valid.all()
    assert abs(params[0][(17 - 1) * 3 + 1]) < 0.3                        # frame 0: no anchor, arm down
    assert params[1][(17 - 1) * 3 + 1] > 0.8                             # frame 1: the anchor's raised arm
    # a per-frame override: a strong pull to an init pose with the arm raised holds it at frame 2
    params2, valid2, _ = fit_sequence_2d(np.stack([uv] * T), np.stack([conf] * T), [cam] * T,
                                         np.stack([J[0, :2]] * T), rest, forward, cfg=cfg, base_cfg=base,
                                         init_orient_seq=np.stack([go] * T),
                                         init_body_pose_seq=np.stack([anchor[:63]] * T),
                                         cfg_overrides=[None, None, {"init_weight": 5.0}])
    assert valid2.all()
    assert params2[2][(17 - 1) * 3 + 1] > 0.8


def test_blend_params_slerps_toward_the_anchor():
    from scipy.spatial.transform import Rotation

    from nfl_gsplat.pose.fit_mono2d import blend_params

    a = _pack_params(np.zeros(63), np.zeros(3), np.array([0.0, 0.0, 0.0]))
    b = a.copy()
    b[(17 - 1) * 3 + 1] = 1.0                                            # a joint turned 1 rad
    b[63:66] = Rotation.from_euler("z", np.radians(90)).as_rotvec()      # the body turned 90 deg
    b[66:69] = [2.0, 0.0, 0.0]
    h = blend_params(a, b, 0.5)
    assert abs(h[(17 - 1) * 3 + 1] - 0.5) < 1e-6
    assert abs(np.degrees(Rotation.from_rotvec(h[63:66]).magnitude()) - 45) < 1e-6
    assert np.allclose(h[66:69], [1.0, 0.0, 0.0])
    assert np.allclose(blend_params(a, b, 0.0), a) and np.allclose(blend_params(a, b, 1.0), b)
