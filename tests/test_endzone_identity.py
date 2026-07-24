import numpy as np
import pandas as pd
from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points


def _sideline_track(C, target, n_frames, f=6000.0, wh=(1920, 1080)):
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    fwd = np.asarray(target, float) - np.asarray(C, float); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]); t = -R @ np.asarray(C, float)
    K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
    return CameraTrack(K=np.repeat(K[None], n_frames, 0), R=np.repeat(R[None], n_frames, 0),
                       t=np.repeat(t[None], n_frames, 0), conf=np.ones(n_frames),
                       width=wh[0], height=wh[1])


def _rows(recs):
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    df = pd.DataFrame(recs)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c not in ("cam",) else ""
    if "player_uid" not in df.columns:
        df["player_uid"] = ""
    return df


def test_identity_correspondences_joins_on_uid():
    sl = _sideline_track([-3.6, 80.0, 36.0], [0, 0, 0], n_frames=2)
    K, R, t = sl.at(0)[0].K(), sl.at(0)[1].R, sl.at(0)[1].t
    # two players at known field points; get their sideline foot pixels
    p_a = np.array([[-10.0, 4.0, 0.0]]); p_b = np.array([[-15.0, -6.0, 0.0]])
    ua = project_points(p_a, K, R, t)[0]; ub = project_points(p_b, K, R, t)[0]
    recs = [
        {"frame": 0, "cam": "sideline", "player_uid": "2025_A_58", "foot_u": ua[0], "foot_v": ua[1], "conf": 1},
        {"frame": 0, "cam": "sideline", "player_uid": "2025_A_20", "foot_u": ub[0], "foot_v": ub[1], "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "2025_A_58", "foot_u": 800.0, "foot_v": 500.0, "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "2025_A_20", "foot_u": 900.0, "foot_v": 520.0, "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "",          "foot_u": 10.0,  "foot_v": 10.0,  "conf": 1},  # OTHER: excluded
    ]
    from nfl_gsplat.calibration.endzone_identity import identity_correspondences
    corr = identity_correspondences(_rows(recs), sl, smooth_window=1)
    assert set(corr) == {0}
    world, uv = corr[0]
    assert world.shape == (2, 3) and uv.shape == (2, 2)
    # #58's field point recovered near (-10, 4), matched to its endzone pixel (800,500)
    i58 = np.argmin(np.linalg.norm(uv - [800.0, 500.0], axis=1))
    assert np.allclose(world[i58, :2], [-10.0, 4.0], atol=0.3) and world[i58, 2] == 0.0


def test_excludes_non_players_and_requires_both_cams():
    from nfl_gsplat.calibration.endzone_identity import identity_correspondences
    sl = _sideline_track([-3.6, 80.0, 36.0], [0, 0, 0], n_frames=1)
    recs = [
        {"frame": 0, "cam": "sideline", "player_uid": "2025_A_58", "foot_u": 960.0, "foot_v": 540.0, "conf": 1},
        {"frame": 0, "cam": "endzone",  "player_uid": "__referee__", "foot_u": 5.0, "foot_v": 5.0, "conf": 1},
    ]
    # #58 only seen in sideline; referee only in endzone -> no correspondences
    assert identity_correspondences(_rows(recs), sl, smooth_window=1) == {}
