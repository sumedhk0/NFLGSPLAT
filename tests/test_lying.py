"""Bodies on the ground: the fit's tilt prior turns round on frames the detector's box says are
lying, the timeline finds those frames from the boxes and leaves their tilt unclamped."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from nfl_gsplat.pose.fit_mono2d import Mono2DConfig, tilt_penalty
from nfl_gsplat.render import timeline as tl


def _orient_with_tilt(deg):
    """A world global_orient (z up) leaning ``deg`` from upright."""
    return (Rotation.from_euler("x", deg, degrees=True) * Rotation.from_rotvec(tl.upright_from_yaw(0.0))).as_rotvec()


def test_tilt_penalty_turns_round_for_a_body_on_the_ground():
    cfg = Mono2DConfig(up_axis=(0.0, 0.0, 1.0), tilt_weight=10.0)
    # Mono2DConfig.up_axis names the rest skeleton's up; with z up here the world orient's lean is
    # measured directly, so build the leaning orient in that frame
    up = np.zeros(3)                                              # identity: upright
    lean75 = Rotation.from_euler("x", 75, degrees=True).as_rotvec()
    assert tilt_penalty(up, cfg) == 0.0 and tilt_penalty(lean75, cfg) > 5.0        # standing: lean is paid for
    lying = Mono2DConfig(up_axis=(0.0, 0.0, 1.0), tilt_weight=10.0, lying=True)
    assert tilt_penalty(lean75, lying) == 0.0 and tilt_penalty(up, lying) > 5.0    # on the ground: upright is paid for
    assert abs(tilt_penalty(up, lying) - 10.0 * np.radians(60.0)) < 1e-9


def test_lying_frames_from_boxes_wider_than_tall():
    df = pd.DataFrame([
        {"cam": "sideline", "track_id": 1, "global_player_id": 7, "frame": 10, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 40},   # lying
        {"cam": "sideline", "track_id": 1, "global_player_id": 7, "frame": 11, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 50, "bbox_y2": 120},   # standing
        {"cam": "endzone", "track_id": 2, "global_player_id": 8, "frame": 10, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 40},    # other camera
        {"cam": "sideline", "track_id": -1, "global_player_id": -1, "frame": 10, "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 40},  # unlinked
    ])
    assert tl.lying_frames(df) == {(10, 7)}
    assert tl.lying_frames(df, aspect=0.3) == set()


def test_timeline_leaves_the_tilt_of_a_body_on_the_ground_unclamped():
    frames = list(range(0, 40))
    ground = {f: {1: np.array([f * 0.05, 1.0])} for f in frames}
    tipped = _orient_with_tilt(80)
    poses = {1: {0: (np.zeros((21, 3)), tl.upright_from_yaw(0.0), np.zeros(10), "fused"),
                 30: (np.zeros((21, 3)), tipped, np.zeros(10), "fused")}}
    kw = dict(default_pose=np.zeros((21, 3)), pose_smooth=0, pose_sigma=0, clamp_joints=False, orient_sigma=0)
    clamped = tl.build_timeline(frames, ground, poses, **kw)
    free = tl.build_timeline(frames, ground, poses, lying={(f, 1) for f in range(28, 33)}, **kw)
    s_c = [s for s in clamped.states[30] if s.pid == 1][0]
    s_f = [s for s in free.states[30] if s.pid == 1][0]
    assert tl.tilt_deg(s_c.global_orient) <= tl.MAX_TILT_TWO_VIEW_DEG + 1e-6 and s_c.clamped
    assert tl.tilt_deg(s_f.global_orient) > 75.0 and not s_f.clamped
    assert free.n_clamped < clamped.n_clamped
