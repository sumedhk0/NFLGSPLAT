"""Cross-camera endzone calibration. Synthetic 2-camera scenes, no video/YOLO."""
import numpy as np
import pytest

from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points


def _look_at(C, target):
    fwd = target - C; fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _cam(C, target, f=2000.0, wh=(1920, 1080)):
    R = _look_at(np.asarray(C, float), np.asarray(target, float))
    t = -R @ np.asarray(C, float)
    K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
    return K, R, t


def test_match_frame_recovers_correct_pairs():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0], [-30.0, -8.0, 0.0], [-20.0, 2.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    uv = project_points(world, K, R, t)                 # exact endzone pixels
    # shuffle the detections; matcher must re-pair them to the right world pts
    perm = [2, 0, 1]
    mw, mu = match_frame(world, uv[perm], K, R, t, max_px=5.0)
    assert mw.shape == (3, 3) and mu.shape == (3, 2)
    # each returned world point's projection equals its matched uv
    proj = project_points(mw, K, R, t)
    assert np.abs(proj - mu).max() < 1e-6


def test_match_frame_distance_gate_drops_outlier():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0], [-30.0, -8.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    uv = project_points(world, K, R, t)
    uv = np.vstack([uv, [10.0, 10.0]])                  # a detection with no world match
    mw, mu = match_frame(world, uv, K, R, t, max_px=5.0)
    assert mw.shape == (2, 3)                            # the 2 real players only


def test_match_frame_far_projection_no_match():
    from nfl_gsplat.calibration.cross_cam_calib import match_frame
    world = np.array([[-40.0, 5.0, 0.0]])
    K, R, t = _cam([-100.0, 0.0, 40.0], [-30.0, 0.0, 0.0])
    mw, mu = match_frame(world, np.array([[5.0, 5.0]]), K, R, t, max_px=5.0)
    assert mw.shape == (0, 3) and mu.shape == (0, 2)


def _tracks_df(rows):
    import pandas as pd
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    df = pd.DataFrame(rows)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c not in ("cam",) else ""
    return df


def _sideline_track(C, target, n_frames=3, f=2600.0, wh=(1920, 1080)):
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    R = _look_at(np.asarray(C, float), np.asarray(target, float))
    t = -R @ np.asarray(C, float)
    K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
    return CameraTrack(K=np.repeat(K[None], n_frames, 0), R=np.repeat(R[None], n_frames, 0),
                       t=np.repeat(t[None], n_frames, 0), conf=np.ones(n_frames),
                       width=wh[0], height=wh[1])


def test_sideline_field_by_frame_projects_feet_to_z0():
    from nfl_gsplat.calibration.cross_cam_calib import sideline_field_by_frame
    sl = _sideline_track([-3.6, 80.0, 36.0], [0.0, 0.0, 0.0])
    # a player truly at field (X=-10, Y=4, 0): find its sideline foot pixel
    K, R, t = sl.at(0)[0].K(), sl.at(0)[1].R, sl.at(0)[1].t
    uv = project_points(np.array([[-10.0, 4.0, 0.0]]), K, R, t)[0]
    df = _tracks_df([{"frame": 0, "cam": "sideline", "foot_u": uv[0], "foot_v": uv[1],
                      "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1}])
    fb = sideline_field_by_frame(df, sl)
    assert 0 in fb
    assert np.allclose(fb[0][0, :2], [-10.0, 4.0], atol=0.2) and fb[0][0, 2] == 0.0


def test_endzone_feet_by_frame_rotates():
    from nfl_gsplat.calibration.cross_cam_calib import endzone_feet_by_frame
    from nfl_gsplat.calibration.view_rotation import rotate_uv
    df = _tracks_df([{"frame": 0, "cam": "endzone", "foot_u": 100.0, "foot_v": 200.0,
                      "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1},
                     {"frame": 0, "cam": "sideline", "foot_u": 5.0, "foot_v": 6.0,
                      "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1}])
    fb = endzone_feet_by_frame(df, deg=90, orig_wh=(1920, 1080))
    assert list(fb) == [0]
    assert np.allclose(fb[0][0], rotate_uv(100.0, 200.0, 90, (1920, 1080)))


def test_solve_endzone_cross_camera_recovers_synthetic():
    # Ground truth: a fixed endzone camera + ~20 players moving over 40 frames.
    # Sideline gives the players' true field points; endzone sees their feet.
    # The solve must recover the endzone camera from a grid/measured init that
    # is NOT the truth.
    from nfl_gsplat.calibration.cross_cam_calib import solve_endzone_cross_camera
    from nfl_gsplat.calibration.view_rotation import rotated_wh
    rng = np.random.default_rng(0)
    ow = (1920, 1080)
    ez_wh = rotated_wh(90, ow)                      # endzone works in rotated frame
    C_true = np.array([-110.0, -8.0, 45.0])
    n_frames, n_players = 40, 20
    field_by, feet_by = {}, {}
    f_by = {}
    for i in range(n_frames):
        # players scattered on the field, drifting frame to frame
        base = rng.uniform([-45, -22], [10, 22], size=(n_players, 2))
        world = np.column_stack([base, np.zeros(n_players)])
        field_by[i] = world
        tx = -20.0 + 30.0 * i / (n_frames - 1)
        R = _look_at(C_true, np.array([tx, 0.0, 0.0]))
        t = -R @ C_true
        f = 2400.0 + 400.0 * i / (n_frames - 1)
        K = CameraIntrinsics(f, f, ez_wh[0] / 2, ez_wh[1] / 2, ez_wh[0], ez_wh[1]).K()
        uv = project_points(world, K, R, t)
        ok = np.isfinite(uv).all(axis=1)
        feet_by[i] = uv[ok] + rng.normal(0, 0.5, uv[ok].shape)
        f_by[i] = f
    results = solve_endzone_cross_camera(
        field_by, feet_by, ez_wh, init_C=np.array([-90.0, 0.0, 35.0]), view_deg=90)
    solved = [r for r in results if r is not None]
    assert len(solved) >= 30
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_true) < 1.0
    for r in solved:
        assert np.allclose(r.pose.center_world(), C_rec)     # one fixed center
    # Assert focal within 2% of ground truth for each frame
    for i, r in enumerate(results):
        if r is None:
            continue
        assert abs(r.intrinsics.fx - f_by[i]) / f_by[i] < 0.02


def test_solve_endzone_cross_camera_fails_loud_no_matches():
    from nfl_gsplat.calibration.cross_cam_calib import solve_endzone_cross_camera
    from nfl_gsplat.errors import CalibrationError
    # players at (X,Y) that no plausible endzone camera can align to random feet
    field_by = {i: np.array([[-30.0, 0.0, 0.0]]) for i in range(15)}
    feet_by = {i: np.array([[9999.0, 9999.0]]) for i in range(15)}
    with pytest.raises(CalibrationError, match="cross-camera"):
        solve_endzone_cross_camera(field_by, feet_by, (1080, 1920),
                                   init_C=np.array([-90.0, 0.0, 35.0]), view_deg=90)
