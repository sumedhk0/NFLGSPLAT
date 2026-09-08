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
                                         np.stack([J[0, :2]] * T), rest, forward, cfg=Mono2DConfig(), base_cfg=base,
                                         init_body_pose_seq=rough, init_orient_seq=rough_go)
    assert valid.all()
    assert rep[-1] < 3.0, rep
    Jf = forward(params[-1])
    assert abs(min(Jf[7, 2], Jf[8, 2])) < 0.05                        # feet on the turf
    assert np.linalg.norm(Jf[0, :2] - J[0, :2]) < 0.3                  # placed on the ground point
    # the raised arm is raised as in the truth: the right wrist's height within 0.15 m of the truth's
    assert J[21, 2] > J[17, 2] + 0.2, "the synthetic arm must actually be raised"
    assert abs(Jf[21, 2] - J[21, 2]) < 0.15, (Jf[21], J[21])
