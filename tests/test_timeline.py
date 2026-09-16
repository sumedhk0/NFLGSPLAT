"""Every player every frame, upright, interpolated (render.timeline)."""
import numpy as np
from scipy.spatial.transform import Rotation

from nfl_gsplat.render import timeline as tl


def test_upright_from_yaw_faces_the_yaw_and_stands_up():
    for yaw in (0.0, 1.0, -2.0):
        go = tl.upright_from_yaw(yaw)
        assert tl.tilt_deg(go) < 1e-6
        assert abs(((tl.yaw_of(go) - yaw) + np.pi) % (2 * np.pi) - np.pi) < 1e-6


def test_clamp_tilt_keeps_yaw_and_limits_tilt():
    yaw = 0.7
    go = tl.upright_from_yaw(yaw)
    # tip the body 70 degrees forward about a horizontal axis
    axis = np.cross(tl.body_up(go), tl.UP)
    tipped = (Rotation.from_rotvec(-axis / np.linalg.norm(axis) * np.radians(70)) *
              Rotation.from_rotvec(go)).as_rotvec()
    assert abs(tl.tilt_deg(tipped) - 70) < 1e-6
    out, clamped = tl.clamp_tilt(tipped, 35.0)
    assert clamped and abs(tl.tilt_deg(out) - 35.0) < 1e-6
    assert abs(((tl.yaw_of(out) - yaw) + np.pi) % (2 * np.pi) - np.pi) < 0.2
    same, c2 = tl.clamp_tilt(go, 35.0)
    assert not c2 and np.allclose(same, go)


def test_interp_axis_angle_slerps_between_posed_frames():
    a = np.zeros((21, 3))
    b = np.zeros((21, 3))
    b[3] = [0.0, 0.0, 1.0]                      # one joint turns one radian
    out = tl.interp_axis_angle([0, 10], [a, b], [0, 5, 10, 20])
    assert np.allclose(out[0], a) and np.allclose(out[2], b) and np.allclose(out[3], b)
    assert abs(np.linalg.norm(out[1][3]) - 0.5) < 1e-6


def test_fill_gaps_and_smooth():
    frames = list(range(10))
    xy = np.array([[i, 0.0] for i in frames], float)
    xy[3:6] = np.nan
    filled = tl.fill_gaps(frames, xy, max_gap=5)
    assert np.allclose(filled[:, 0], np.arange(10))
    xy[3:6] = np.nan
    unfilled = tl.fill_gaps(frames, xy, max_gap=1)
    assert np.isnan(unfilled[4]).all()
    sm = tl.smooth_xy(filled, window=3)
    assert np.allclose(sm[:, 0], np.arange(10), atol=0.5)


def test_yaw_from_motion_follows_travel_and_holds_when_still():
    xy = np.array([[i * 0.5, 0.0] for i in range(20)] + [[9.5, 0.0]] * 10, float)
    yaw = tl.yaw_from_motion(xy, window=4)
    assert abs(yaw[5]) < 1e-6                   # moving +x
    assert abs(yaw[-1]) < 1e-6                  # still: last heading held


def test_build_timeline_gives_every_player_a_body_every_frame():
    frames = list(range(0, 60))
    ground = {f: {1: np.array([f * 0.1, 1.0]), 2: np.array([0.0, 6.0 + f * 0.05])} for f in frames}
    for f in range(20, 25):                      # player 2 undetected briefly
        del ground[f][2]
    tipped = (Rotation.from_euler("x", 75, degrees=True) *              # past the two-view limit
              Rotation.from_rotvec(tl.upright_from_yaw(0.0))).as_rotvec()
    poses = {1: {0: (np.zeros((21, 3)), tl.upright_from_yaw(0.0), np.zeros(10), "fused"),
                 30: (np.ones((21, 3)) * 0.2, tipped, np.zeros(10), "fused")}}
    # both smoothers off: this checks the interpolation, and the Gaussian's held edge dips a
    # ramp's last frame by a few percent, which is its job and not this test's question
    out = tl.build_timeline(frames, ground, poses, default_pose=np.ones((21, 3)) * 0.1, pose_smooth=0, pose_sigma=0,
                            clamp_joints=False, orient_sigma=0)          # 0.2 rad on every axis is a -11 deg elbow: the clamp would move it
    assert all(len(out.states[f]) == 2 for f in frames), "both players every frame"
    s1 = {f: [s for s in out.states[f] if s.pid == 1][0] for f in frames}
    s2 = {f: [s for s in out.states[f] if s.pid == 2][0] for f in frames}
    assert abs(np.linalg.norm(s1[15].body_pose) - 0.5 * np.linalg.norm(s1[30].body_pose)) < 1e-3
    assert all(tl.tilt_deg(s1[f].global_orient) <= tl.MAX_TILT_TWO_VIEW_DEG + 1e-6 for f in frames)
    assert out.n_clamped > 0
    assert s2[22].source == "default" and np.isfinite(s2[22].xy).all()
    assert abs(tl.yaw_of(s2[40].global_orient) - np.pi / 2) < 0.2    # travelling +y


def test_dedupe_keeps_the_posed_body_of_two_ids_on_one_spot():
    frames = list(range(0, 20))
    ground = {f: {1: np.array([1.0, 1.0]), 2: np.array([1.3, 1.2]), 3: np.array([8.0, 0.0])}
              for f in frames}
    poses = {2: {0: (np.zeros((21, 3)), tl.upright_from_yaw(0.0), np.zeros(10), "fused")}}
    out = tl.build_timeline(frames, ground, poses)
    for f in frames:
        pids = sorted(s.pid for s in out.states[f])
        assert pids == [2, 3], pids                  # 1 (default-posed) dropped for 2 (fused)
    assert out.n_duplicates == len(frames)


def test_endzone_only_ids_along_their_depth_axis_are_duplicates_and_sideline_detections_never():
    frames = list(range(0, 12))
    ground = {f: {1: np.array([10.0, 2.0]), 2: np.array([13.0, 2.4]), 3: np.array([10.3, 5.5]),
                  4: np.array([30.0, 0.0]), 5: np.array([10.0, 2.6])} for f in frames}
    views = {f: {1: ("endzone", "sideline"), 2: ("endzone",), 3: ("sideline",), 4: ("endzone",),
                 5: ("sideline",)} for f in frames}
    out = tl.build_timeline(frames, ground, {}, views_by_frame=views)
    for f in frames:
        pids = sorted(s.pid for s in out.states[f])
        # 2: endzone-only 3 m along x (its depth) from 1 -> the endzone's copy, dropped
        # 3: sideline-seen 3.5 m along y from 1 -> a person the sideline detected, kept
        # 4: far from everyone -> kept
        # 5: sideline-seen 0.6 m from 1 (a lineman beside another) -> kept
        assert pids == [1, 3, 4, 5], pids
    assert out.n_duplicates == len(frames)


def test_smooth_xy_never_averages_across_a_gap():
    """Two stationary segments of one track, metres apart, separated by a gap fill_gaps will not bridge.

    The old smoother compacted every finite row into one array before convolving, so the last window // 2
    frames of the first segment were averaged with the first frames of the second: play 1's id 21 stood
    still at (-27.1, -4.0) and was drawn marching 0.88 m/frame toward where its track resumed 77 frames
    later (2026-09-15). Smoothing must stay inside each contiguous run.
    """
    xy = np.full((60, 2), np.nan)
    xy[0:20] = [-27.1, -4.0]                  # segment A, stationary
    xy[40:60] = [-20.0, +3.0]                 # segment B, stationary, 8 m away, after a 20-frame gap
    out = tl.smooth_xy(xy, window=9)
    assert np.allclose(out[0:20], [-27.1, -4.0]), out[15:20]      # A's tail is not pulled toward B
    assert np.allclose(out[40:60], [-20.0, +3.0]), out[40:45]     # B's head is not pulled toward A
    assert np.isnan(out[20:40]).all()                              # the gap stays a gap


def test_the_endzone_copy_strung_along_its_depth_axis_stays_a_duplicate():
    """The endzone's unreconciled copy of a player sits FAR along x (its blind depth axis) and near in y.

    Play 2 is the case this guards: lowering ONE_VIEW_ACROSS_M to 1.0 readmitted 123 such states, |dx| to
    the body they duplicate p50 2.87 m (id 2 on 53 frames at 3.46). Measured 2026-09-13, which is why that
    change was reverted.

    The open defect this does NOT cover: on play 1 the sideline merges linemen into one box, and the
    endzone's separate detection of the next man stands |dx| 0.38-0.79 m with |dy| 1.19-1.30 -- beside him
    at the same depth, a different player -- and the 1.5 m across radius deletes him too (id 74 on 65 of 65
    frames). Distinguishing the two needs a depth-aware exception; see HANDOFF.
    """
    frames = list(range(0, 12))
    ground = {f: {1: np.array([10.0, 2.0]), 2: np.array([12.9, 3.2])} for f in frames}
    views = {f: {1: ("sideline",), 2: ("endzone",)} for f in frames}
    out = tl.build_timeline(frames, ground, {}, views_by_frame=views)
    for f in frames:
        pids = sorted(s.pid for s in out.states[f])
        assert pids == [1], (f, pids)       # 2.9 m along x, 1.2 m across: the endzone's own copy
    assert out.n_duplicates == len(frames)


def test_a_short_detection_gap_keeps_the_body_beside_its_neighbour():
    frames = list(range(0, 12))
    ground = {f: {1: np.array([10.0, 2.0]), 2: np.array([10.4, 2.3])} for f in frames}
    for f in range(4, 8):                                  # id 2 undetected for four frames
        del ground[f][2]
    views = {f: {1: ("sideline",), **({2: ("sideline",)} if 2 in ground[f] else {})} for f in frames}
    out = tl.build_timeline(frames, ground, {}, views_by_frame=views)
    for f in frames:
        pids = sorted(s.pid for s in out.states[f])
        assert pids == [1, 2], (f, pids)                   # the sideline saw 2 within the gap: a person, kept
    assert out.n_duplicates == 0


def test_an_interpolated_fragment_on_top_of_a_detected_body_is_dropped():
    frames = list(range(0, 12))
    # id 2 is a second fragment of id 1's player: detected 0-3, then interpolated on top of id 1
    ground = {f: {1: np.array([10.0, 2.0]), **({2: np.array([10.1, 2.1])} if f < 4 or f >= 10 else {})} for f in frames}
    views = {f: {1: ("sideline",), **({2: ("sideline",)} if 2 in ground[f] else {})} for f in frames}
    out = tl.build_timeline(frames, ground, {}, views_by_frame=views)
    assert sorted(s.pid for s in out.states[6]) == [1]              # interpolated 0.14 m from a detected body: dropped
    assert sorted(s.pid for s in out.states[1]) == [1, 2]           # both detected: two people (the split's job)


def test_an_id_unseen_for_long_dedupes_at_the_plain_radius():
    frames = list(range(0, 80))
    ground = {f: {1: np.array([10.0, 2.0]), 2: np.array([10.4, 2.3])} for f in frames}
    for f in range(10, 70):                                # id 2 undetected for sixty frames
        del ground[f][2]
    views = {f: {1: ("sideline",), **({2: ("sideline",)} if 2 in ground[f] else {})} for f in frames}
    out = tl.build_timeline(frames, ground, {}, views_by_frame=views, min_frames=6)
    mid = sorted(s.pid for s in out.states[40])
    assert mid == [1], mid                                  # no sighting within 30 frames: not anchored (and not filled: gap > max_gap)
    assert sorted(s.pid for s in out.states[5]) == [1, 2]


def test_relabel_merges_fragments_under_the_stitch_map():
    ground = {0: {1: np.array([0.0, 0.0]), 5: np.array([9.0, 9.0])},
              1: {2: np.array([0.2, 0.0]), 5: np.array([9.1, 9.0])}}
    views = {0: {1: ("sideline",), 5: ("endzone", "sideline")}, 1: {2: ("endzone",), 5: ("sideline",)}}
    poses = {1: {0: ("bp1", "go1", "b1", "sideline")}, 2: {0: ("bp2", "go2", "b2", "fused"),
                                                          1: ("bp2b", "go2b", "b2", "sideline")}}
    g, v, p, members = tl.relabel(ground, views, poses, {2: 1})
    assert sorted(g[1]) == [1, 5] and np.allclose(g[1][1], [0.2, 0.0])
    assert v[1][1] == ("endzone",) and v[0][5] == ("endzone", "sideline")
    assert p[1][0][3] == "fused" and p[1][1][3] == "sideline"      # fused wins the collision
    assert members == {1: [1, 2], 5: [5]}


def test_place_from_refit_moves_two_view_bodies_to_the_refit_translation():
    from nfl_gsplat.render.play_timeline import place_from_refit

    ground = {0: {1: np.array([0.0, 0.0]), 2: np.array([5.0, 5.0])}, 1: {1: np.array([0.5, 0.0])}}
    refit = {0: {1: {"transl": np.array([0.6, -0.2, 0.9])}, 3: {"transl": np.array([9.0, 9.0, 0.9])}},
             1: {1: {"transl": np.array([8.0, 0.0, 0.9])}}}             # frame 1: 7.5 m away, refused
    out, shifts = place_from_refit(ground, refit)
    assert np.allclose(out[0][1], [0.6, -0.2]) and np.allclose(out[0][2], [5.0, 5.0])
    assert np.allclose(out[1][1], [0.5, 0.0])
    assert len(shifts) == 1 and abs(shifts[0] - np.hypot(0.6, 0.2)) < 1e-9
    assert np.allclose(ground[0][1], [0.0, 0.0])                       # input untouched


def test_two_view_poses_keep_a_45_degree_bend_single_view_do_not():
    from scipy.spatial.transform import Rotation

    # An upright body (Rx(90) stands SMPL-X up) bent 45 deg forward.
    bent = (Rotation.from_euler("y", 45, degrees=True)
            * Rotation.from_euler("x", 90, degrees=True)).as_rotvec()
    ground = {0: {1: np.array([0.0, 0.0]), 2: np.array([5.0, 0.0])}}
    poses = {1: {0: (np.zeros((21, 3)), bent, np.zeros(10), "fused")},
             2: {0: (np.zeros((21, 3)), bent, np.zeros(10), "sideline")}}
    out = tl.build_timeline([0], ground, poses, default_pose=np.zeros((21, 3)), default_betas=np.zeros(10),
                            min_frames=1)
    by = {s.pid: s for s in out.states[0]}
    assert not by[1].clamped and by[2].clamped


def test_interpolated_frames_inherit_the_nearest_recorded_views():
    ground = {f: {1: np.array([float(f) * 0.1, 0.0])} for f in range(0, 30)}
    views = {0: {1: ("endzone",)}, 29: {1: ("endzone",)}}       # recorded twice, one camera
    tl_ = tl.build_timeline(list(range(30)), ground, {}, default_pose=np.zeros((21, 3)),
                            default_betas=np.zeros(10), views_by_frame=views, min_frames=1)
    assert all(s.views == ("endzone",) for f in tl_.frames for s in tl_.states[f])


def test_excluded_ids_are_left_out():
    ground = {0: {1: np.array([0.0, 0.0]), 2: np.array([5.0, 0.0])}}
    tl_ = tl.build_timeline([0], ground, {}, default_pose=np.zeros((21, 3)),
                            default_betas=np.zeros(10), min_frames=1, exclude={2})
    assert [s.pid for s in tl_.states[0]] == [1]


def test_place_from_refit_interpolates_the_translation_across_a_short_gap():
    from nfl_gsplat.render.play_timeline import place_from_refit

    ground = {f: {1: np.array([1.0, 0.0])} for f in range(0, 6)}          # the box point, 0-0.4 m from the records
    ground[10] = {1: np.array([1.0, 0.0])}
    ground[30] = {1: np.array([5.0, 0.0])}
    refit = {0: {1: {"transl": np.array([1.0, 0.0, 0.9])}}, 4: {1: {"transl": np.array([1.4, 0.0, 0.9])}},
             30: {1: {"transl": np.array([5.0, 0.0, 0.9])}}}
    out, _ = place_from_refit(ground, refit, max_gap=12)
    assert np.allclose(out[2][1], [1.2, 0.0])                            # inside the 3-frame gap: interpolated
    # with a pelvis function the records place at their pelvis, the interpolation too
    out2, _ = place_from_refit(ground, refit, max_gap=12, pelvis_xy=lambda r: r["transl"][:2] + np.array([0.0, -0.35]))
    assert np.allclose(out2[0][1], [1.0, -0.35]) and np.allclose(out2[2][1], [1.2, -0.35])
    assert np.allclose(out[10][1], [1.0, 0.0])                           # the 25-frame gap: left alone
    assert np.allclose(out[5][1], [1.0, 0.0])                            # after the last record: left alone


def test_smooth_axis_angles_damps_noise_and_keeps_a_ramp():
    rng = np.random.default_rng(0)
    T = 60
    ramp = np.linspace(0, 1.0, T)[:, None, None] * np.ones((1, 21, 3))
    noisy = ramp + rng.normal(0, 0.1, ramp.shape)
    sm = tl.smooth_axis_angles(noisy, window=9)
    assert sm.shape == noisy.shape
    d2 = lambda a: np.abs(np.diff(a, n=2, axis=0)).mean()
    assert d2(sm) < 0.3 * d2(noisy)                                    # the noise goes
    assert abs(sm[30, 0, 0] - ramp[30, 0, 0]) < 0.05                   # the ramp stays
    assert np.allclose(tl.smooth_axis_angles(ramp, window=1), ramp)   # off


def test_gaussian_pose_smoother_kills_per_frame_noise_the_median_keeps_and_passes_a_swing():
    """The fits are noisy on EVERY frame at the extremities (play 1, 2026-09-15): a median drops
    isolated spikes and leaves that alone; a Gaussian averages it down. A real arm swing (1 Hz at
    60 fps) has to pass through nearly whole."""
    T = 120
    t = np.arange(T)
    swing = 0.6 * np.sin(2 * np.pi * t / 60.0)                         # 1 Hz, 0.6 rad amplitude
    noise = 0.1 * np.where(t % 2 == 0, 1.0, -1.0)                      # +-0.1 rad every frame
    seq = np.zeros((T, 21, 3))
    seq[:, 5, 0] = swing + noise
    g = tl.smooth_axis_angles_gaussian(seq, sigma=2.0)
    m = tl.smooth_axis_angles(seq, window=7)
    resid = lambda a: np.abs(a[:, 5, 0] - swing)[10:-10].max()         # away from the held edges
    assert resid(g) < 0.03                                             # noise gone, swing kept (< 5 %)
    assert resid(m) > 0.08                                             # the median keeps the toggling
    assert g.shape == seq.shape and np.allclose(g[:, :5], 0.0)         # untouched joints stay zero
    assert np.allclose(tl.smooth_axis_angles_gaussian(seq, sigma=0), seq)   # off


def test_hinge_clamp_fixes_a_backwards_knee_and_a_sideways_elbow_and_leaves_the_rest():
    """Play 1's fits bend knees backwards (-104 deg) and elbows sideways (121 deg off the hinge axis)
    on the ids whose limbs jitter most. The clamp names those and nothing else: a collar turning
    150 deg is left alone on purpose (clamping it was measured to move the arm off the keypoints)."""
    seq = np.zeros((5, 21, 3))
    seq[:, 3, 0] = np.radians(-104)                   # L_knee bent backwards
    seq[:, 4, 0] = np.radians(120)                    # R_knee: a sprint, fine
    seq[:, 18, 1] = np.radians(40)                    # R_elbow flexed 40 (positive about y)
    seq[:, 18, 0] = np.radians(60)                    # ... and turned 60 off its axis
    seq[:, 17, 1] = np.radians(-30)                   # L_elbow flexed 30 (negative about y), fine
    seq[:, 12, 2] = np.radians(150)                   # L_collar: absurd, untouched
    out = tl.clamp_hinges(seq)
    assert np.allclose(np.degrees(out[:, 3, 0]), -5)                          # clipped to the limit
    assert np.allclose(out[:, 4], seq[:, 4]) and np.allclose(out[:, 17], seq[:, 17])
    assert np.allclose(np.degrees(out[:, 18, 1]), 40)                          # flexion kept
    assert np.allclose(np.degrees(out[:, 18, 0]), 25) and np.allclose(out[:, 18, 2], 0)   # off-axis scaled
    assert np.allclose(out[:, 12], seq[:, 12])
    assert np.allclose(tl.clamp_hinges(np.zeros((3, 21, 3))), 0.0)


def test_ground_positions_take_the_foot_above_the_box_margin():
    import pandas as pd

    from nfl_gsplat.calibration.cameras_io import CameraTrack
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at
    from nfl_gsplat.render.play_timeline import BOX_MARGIN_FRAC, ground_positions

    K = intrinsics(1920, 1080, fov_deg=12.0)
    R, t = look_at(np.array([0.0, -100.0, 40.0]), np.array([0.0, 0.0, 0.0]))
    track = CameraTrack(K=K[None], R=R[None], t=t[None], conf=np.ones(1), width=1920, height=1080)
    # both ids, as the real table carries them; ground_positions keys by the player, not the tracker
    box = pd.DataFrame([{"cam": "sideline", "frame": 0, "track_id": 1, "global_player_id": 1,
                         "bbox_x1": 940, "bbox_y1": 400, "bbox_x2": 980, "bbox_y2": 540}])
    g0 = ground_positions(box, {"sideline": track}, margin_frac=0.0)[0][1]
    g1 = ground_positions(box, {"sideline": track})[0][1]
    assert BOX_MARGIN_FRAC > 0
    # the box bottom is 140 px tall; its ground point lies toward the camera (smaller y) of the true foot's
    assert g1[1] > g0[1] and 0.05 < g1[1] - g0[1] < 0.6


def test_orientation_gets_the_gaussian_by_default_and_a_yaw_toggle_is_damped():
    """global_orient kept the 7-frame median after body_pose got its Gaussian (2026-09-16: yaw jitter
    p90 2.7 -> 0.5 deg/frame^2 at sigma 4 for ~1 px of reprojection). The median leaves a per-frame
    toggle at +-0.1 rad; the Gaussian removes it."""
    frames = list(range(0, 40))
    ground = {f: {1: np.array([f * 0.05, 1.0])} for f in frames}
    yaws = [0.3 + 0.1 * (1 if f % 2 == 0 else -1) for f in frames]           # facing +-0.1 rad about 0.3
    poses = {1: {f: (np.zeros((21, 3)), tl.upright_from_yaw(y), np.zeros(10), "fused") for f, y in zip(frames, yaws)}}
    out = tl.build_timeline(frames, ground, poses)
    got = np.array([tl.yaw_of([s for s in out.states[f] if s.pid == 1][0].global_orient) for f in frames[8:-8]])
    assert np.abs(got - 0.3).max() < 0.02, got                              # the toggle is gone
    out_med = tl.build_timeline(frames, ground, poses, orient_sigma=0)
    got_med = np.array([tl.yaw_of([s for s in out_med.states[f] if s.pid == 1][0].global_orient) for f in frames[8:-8]])
    assert np.abs(got_med - 0.3).max() > 0.05                                # the median kept it
    assert tl.ORIENT_SMOOTH_SIGMA == 4.0
