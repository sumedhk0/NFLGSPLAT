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
    frames = list(range(0, 24))
    # id 2 is a second fragment of id 1's player: detected 0-3, then interpolated on top of id 1 for
    # twelve frames (longer than HOLE_REACH: a tail, not a blink), detected again 16-23
    ground = {f: {1: np.array([10.0, 2.0]), **({2: np.array([10.1, 2.1])} if f < 4 or f >= 16 else {})} for f in frames}
    views = {f: {1: ("sideline",), **({2: ("sideline",)} if 2 in ground[f] else {})} for f in frames}
    out = tl.build_timeline(frames, ground, {}, views_by_frame=views)
    assert sorted(s.pid for s in out.states[6]) == [1]              # interpolated 0.14 m from a detected body: dropped
    assert sorted(s.pid for s in out.states[1]) == [1, 2]           # both detected: two people (the split's job)
    # a hole within HOLE_REACH is the same id blinking and is kept -- dropping only the hole frames of
    # a twin that is drawn on its detected frames was the flicker (2026-09-16, 14 of 32 live pops)
    ground2 = {f: {1: np.array([10.0, 2.0]), **({2: np.array([10.1, 2.1])} if f < 4 or f >= 8 else {})} for f in frames}
    views2 = {f: {1: ("sideline",), **({2: ("sideline",)} if 2 in ground2[f] else {})} for f in frames}
    out2 = tl.build_timeline(frames, ground2, {}, views_by_frame=views2)
    assert sorted(s.pid for s in out2.states[6]) == [1, 2]


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


def test_a_short_fragment_riding_a_teammate_is_a_rider_and_a_long_or_lone_one_is_not():
    """Play 1's id 162: 19 frames, within 0.6 m of a Chiefs body on 53 % of them, its own detections
    flipping between two adjacent men. A long track beside a teammate (a lineman) and a short
    fragment on its own are kept."""
    frames = list(range(0, 100))
    ground = {f: {1: np.array([10.0, 0.0]), 2: np.array([10.4, 0.2]), 3: np.array([30.0, 5.0])} for f in frames}
    for f in range(20, 100):                       # id 2 exists for 20 frames only, on id 1's shoulder
        del ground[f][2]
    for f in range(0, 100):                        # id 4: short, alone
        if f < 15:
            ground[f][4] = np.array([50.0, 50.0])
    # sideline views: the dedupe never touches a sideline detection, which is exactly why 162 survived it
    views = {f: {p: ("sideline",) for p in g} for f, g in ground.items()}
    out = tl.build_timeline(frames, ground, {}, views_by_frame=views)
    team = {1: "KC", 2: "KC", 3: "BAL", 4: "BAL"}
    assert tl.rider_ids(out, team) == {2}
    assert tl.rider_ids(out, {**team, 2: "BAL"}) == set()      # a different team beside him is not a rider
    n = tl.drop_ids(out, {2})
    assert n == 20 and all(2 not in [s.pid for s in out.states[f]] for f in frames)


def _sweep_through_half_turn(n=41, lo_deg=165.0, hi_deg=195.0, axis=(0.0, 0.66, 0.75)):
    """Canonical axis-angle rows (as the keyframe SLERP returns them) of a body turning smoothly
    THROUGH a half turn about one axis: the magnitude passes pi mid-way and the canonical vector
    flips sign there."""
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    angles = np.radians(np.linspace(lo_deg, hi_deg, n))
    return Rotation.from_rotvec(angles[:, None] * a[None]).as_rotvec()


def _angular_error_deg(a, b):
    return np.degrees((Rotation.from_rotvec(a).inv() * Rotation.from_rotvec(b)).magnitude())


def test_unwrap_makes_a_half_turn_sweep_continuous():
    seq = _sweep_through_half_turn()
    assert np.any(np.sum(seq[:-1] * seq[1:], axis=1) < 0), "the fixture must flip sign somewhere"
    un = tl.unwrap_axis_angles(seq)
    # every row is still the same rotation, and neighbours are now close in the vector space
    assert np.all(_angular_error_deg(seq, un) < 1e-6)
    steps = np.linalg.norm(np.diff(un, axis=0), axis=1)
    assert steps.max() < np.radians(1.0)
    # a joint-shaped [T, J, 3] input is unwrapped per joint
    stacked = np.stack([seq, seq[::-1]], axis=1)
    un2 = tl.unwrap_axis_angles(stacked)
    assert un2.shape == stacked.shape and np.allclose(un2[:, 0], un)


def test_orientation_gaussian_survives_a_half_turn():
    """A body with its back to the camera has |global_orient| near pi; when it turns through pi the
    canonical vectors flip sign and a component-wise Gaussian averages antipodal vectors into a
    garbage orientation for ~2 sigma frames (play 1's id 0: legs splayed for 12 frames at the snap,
    cache legs 4 px on the keypoints). Unwrapping first makes the Gaussian exact again."""
    seq = _sweep_through_half_turn()
    raw = tl.smooth_axis_angles_gaussian(seq, sigma=4.0)
    fixed = tl.smooth_axis_angles_gaussian(tl.unwrap_axis_angles(seq), sigma=4.0)
    err_raw = np.array([_angular_error_deg(s, r) for s, r in zip(seq, raw)])
    err_fixed = np.array([_angular_error_deg(s, r) for s, r in zip(seq, fixed)])
    assert err_raw.max() > 30.0, f"negative control: the plain Gaussian should break, got {err_raw.max():.1f} deg"
    # the held edges bias a ramp by ~sigma * slope (1.2 deg here); the interior is exact
    assert err_fixed.max() < 2.0, f"unwrapped Gaussian off by {err_fixed.max():.2f} deg"
    assert err_fixed[8:-8].max() < 0.1


def test_orphan_ids_drops_only_short_teamless_fragments():
    """Play 1's id 203: eight frames, no team, drawn in the default kit on the pile. A teamless id drawn
    for longer is a real unidentified man and stays; a short TEAMED fragment is the rider rule's."""
    frames = list(range(100, 200))
    ground = {f: {1: np.array([0.0, 0.0]), 2: np.array([5.0, 0.0])} for f in frames}
    for f in frames[:8]:
        ground[f][203] = np.array([2.0, 0.0])         # teamless, 8 frames (clear of the dedupe box)
    for f in frames[:8]:
        ground[f][77] = np.array([7.0, 0.0])          # teamed, 8 frames: not this rule's
    for f in frames:
        ground[f][99] = np.array([10.0, 0.0])         # teamless, 100 frames: a real unidentified man
    tl_ = tl.build_timeline(frames, ground, {}, default_pose=np.zeros((21, 3)), default_betas=np.zeros(10),
                            min_frames=1, pose_smooth=0, pose_sigma=0, clamp_joints=False, orient_sigma=0)
    team_of = {1: "KC", 2: "BAL", 77: "KC"}
    assert tl.orphan_ids(tl_, team_of) == {203}
    assert tl.orphan_ids(tl_, team_of, max_frames=4) == set()
    n = tl.drop_ids(tl_, tl.orphan_ids(tl_, team_of))
    assert n == 8 and all(203 not in {s.pid for s in tl_.states[f]} for f in frames)


def test_fill_gap_bridge_is_ten_frames_by_default():
    """A body glides across a gap of at most FILL_GAP_FRAMES; a longer one is left empty (measured
    2026-09-16: the 30-frame bridge let two bodies glide 2-3 m across the pile unseen)."""
    assert tl.FILL_GAP_FRAMES == 10 and tl.MAX_GAP_FRAMES == 30
    frames = list(range(40))
    xy = np.full((40, 2), np.nan)
    xy[0] = [0.0, 0.0]
    xy[9] = [9.0, 0.0]          # an 8-frame gap: bridged
    xy[39] = [39.0, 0.0]        # a 29-frame gap: not bridged by default, bridged at 30
    filled = tl.fill_gaps(frames, xy)
    assert np.allclose(filled[5], [5.0, 0.0]) and np.isnan(filled[20]).all()
    assert np.allclose(tl.fill_gaps(frames, xy, max_gap=tl.MAX_GAP_FRAMES)[20], [20.0, 0.0])


def test_a_short_detection_hole_keeps_its_filled_frame_but_a_fragment_tail_does_not():
    """Two linemen 0.3 m apart. Id 1's sideline detection drops on frame 105 only: its filled frame
    is a hole (detected on both sides within HOLE_REACH) and stays. Id 3 is detected up to 104 and
    never again: its filled frames after are a tail riding id 2 and go, as before."""
    frames = list(range(100, 112))
    ground = {f: {1: np.array([0.0, 0.0]), 2: np.array([0.3, 0.0])} for f in frames}
    for f in frames:
        ground[f][3] = np.array([0.3, 0.05])
    views = {f: {1: ("sideline",), 2: ("sideline",), 3: ("sideline",)} for f in frames}
    views[105] = {2: ("sideline",), 3: ("sideline",)}                   # id 1 blinks on 105
    for f in range(105, 112):
        views[f] = {k: v for k, v in views[f].items() if k != 3}         # id 3 ends at 104
    kw = dict(default_pose=np.zeros((21, 3)), default_betas=np.zeros(10), min_frames=1, pose_smooth=0,
              pose_sigma=0, clamp_joints=False, orient_sigma=0, views_by_frame=views)
    tl_ = tl.build_timeline(frames, ground, {}, **kw)
    assert 1 in {s.pid for s in tl_.states[105]}                         # the hole is kept
    assert all(3 not in {s.pid for s in tl_.states[f]} for f in range(106, 112))   # the tail is not
    tl_old = tl.build_timeline(frames, ground, {}, hole_reach=0, **kw)
    assert 1 not in {s.pid for s in tl_old.states[105]}                  # the old rule dropped the hole


def test_twin_frames_drops_the_shorter_id_on_a_long_close_stretch_only():
    """Ids 1 and 2 (same team) 0.15 m apart for 12 consecutive frames: the one drawn on fewer frames
    loses those 12. A 5-frame brush, a cross-team pair, and a pair 0.5 m apart (past TWIN_M, 0.4 since
    2026-09-18) are left alone."""
    frames = list(range(0, 60))
    ground = {f: {1: np.array([0.0, 0.0]), 3: np.array([5.0, 0.0]), 4: np.array([5.15, 0.0]), 5: np.array([9.0, 0.0])} for f in frames}
    for f in range(10, 22):
        ground[f][2] = np.array([0.15, 0.0])            # twin of 1 for 12 frames
    for f in range(40, 45):
        ground[f][2] = np.array([0.15, 0.0])            # a 5-frame brush, past the bridge and the hole reach: not a run
    for f in frames:
        ground[f][6] = np.array([9.5, 0.0])             # 0.5 m from 5: a pile, not a twin
    team_of = {1: "KC", 2: "KC", 3: "KC", 4: "BAL", 5: "BAL", 6: "BAL"}
    views = {f: {pid: ("sideline",) for pid in ground[f]} for f in frames}     # all detected: the dedupe keeps them
    tl_ = tl.build_timeline(frames, ground, {}, default_pose=np.zeros((21, 3)), default_betas=np.zeros(10),
                            min_frames=1, pose_smooth=0, pose_sigma=0, clamp_joints=False, orient_sigma=0,
                            views_by_frame=views)
    drop = tl.twin_frames(tl_, team_of)
    assert drop == {(f, 2) for f in range(10, 22)}
    n = tl.drop_frames(tl_, drop)
    assert n == 12 and all(2 not in {s.pid for s in tl_.states[f]} for f in range(10, 22))
    assert all(2 in {s.pid for s in tl_.states[f]} for f in range(40, 45))
    assert all({3, 4, 5, 6} <= {s.pid for s in tl_.states[f]} for f in frames)



def test_box_twin_frames_drops_the_shorter_id_where_two_sideline_boxes_coincide():
    """Ids 1 and 2 (same team) share one man's box on frames 10-19 (IoU 0.9); id 3 (same team) stands
    beside them with a box overlapping id 1 at 0.4 like an engaged lineman; a run of 5 is too short."""
    import pandas as pd
    from nfl_gsplat.render.timeline import PlayerState, Timeline, box_twin_frames

    frames = list(range(0, 30))
    tl = Timeline(frames=frames)
    rows = []
    for f in frames:
        for pid in (1, 2, 3):
            tl.states.setdefault(f, []).append(PlayerState(pid=pid, xy=np.array([float(pid), 0.0]), body_pose=np.zeros((21, 3)),
                                                          global_orient=np.zeros(3), betas=np.zeros(10), source="sideline",
                                                          clamped=False, views=("sideline",)))
        rows.append({"cam": "sideline", "track_id": 1, "global_player_id": 1, "frame": f, "bbox_x1": 100, "bbox_y1": 100, "bbox_x2": 140, "bbox_y2": 200})
        if 10 <= f <= 19 or 25 <= f <= 29:                                               # a run of 10, then one of 5
            rows.append({"cam": "sideline", "track_id": 2, "global_player_id": 2, "frame": f, "bbox_x1": 102, "bbox_y1": 100, "bbox_x2": 142, "bbox_y2": 200})
        rows.append({"cam": "sideline", "track_id": 3, "global_player_id": 3, "frame": f, "bbox_x1": 124, "bbox_y1": 100, "bbox_x2": 164, "bbox_y2": 200})
    df = pd.DataFrame(rows)
    drop = box_twin_frames(tl, df, {1: "KC", 2: "KC", 3: "KC"}, iou_min=0.6, min_run=8)
    assert drop == {(f, 2) for f in range(10, 20)}                                          # id 2 has fewer boxes: it loses
    assert box_twin_frames(tl, df, {1: "KC", 2: "BAL", 3: "KC"}, iou_min=0.6, min_run=8) == set()   # different teams: never


def test_despike_xy_removes_a_single_frame_spike_and_keeps_a_cut():
    from nfl_gsplat.render.timeline import despike_xy

    xy = np.stack([0.1 * np.arange(20), np.zeros(20)], axis=1)      # walking +x
    xy[8] += [0.0, 0.5]                                              # a half-metre spike sideways for one frame
    out = despike_xy(xy, excess_m=0.15)
    assert abs(out[8, 1]) < 1e-9 and abs(out[8, 0] - 0.8) < 1e-9 and np.allclose(out[:8], xy[:8]) and np.allclose(out[9:], xy[9:])
    cut = np.stack([0.1 * np.arange(20), np.where(np.arange(20) >= 10, 0.3 * (np.arange(20) - 9), 0.0)], axis=1)   # turns hard at 10
    assert np.allclose(despike_xy(cut, excess_m=0.15), cut)         # every frame follows the new trend: nothing to remove
    assert np.allclose(despike_xy(xy, excess_m=None), xy)
    holes = xy.copy(); holes[7] = np.nan
    out2 = despike_xy(holes, excess_m=0.15)
    assert np.isnan(out2[7]).all() and abs(out2[8, 1] - 0.5) < 1e-9      # a hole in the window: the frame is left alone


def test_yaw_from_motion_faces_the_first_heading_before_it_moves_and_turns_smoothly():
    from nfl_gsplat.render.timeline import yaw_from_motion

    xy = np.zeros((30, 2))
    xy[10:, 0] = 0.2 * np.arange(20)                                  # still for 10 frames, then runs +x
    yaw = yaw_from_motion(xy, smooth=1)
    assert np.allclose(yaw[:10], yaw[12]) and abs(yaw[15]) < 1e-6      # the still frames face where it will run
    # a turn from +x to +y over the run is smoothed, not stepped: no single-frame jump over 60 deg
    xy2 = np.zeros((40, 2)); xy2[:20, 0] = 0.2 * np.arange(20); xy2[20:, 0] = xy2[19, 0]; xy2[20:, 1] = 0.2 * np.arange(1, 21)
    y2 = yaw_from_motion(xy2, smooth=5)
    d = np.abs(np.degrees(np.angle(np.exp(1j * (y2[1:] - y2[:-1])))))
    assert d.max() < 60.0 and abs(np.degrees(y2[5])) < 15.0 and abs(np.degrees(y2[-3]) - 90.0) < 15.0


def test_lying_frames_take_a_wider_pose_smoothing():
    """A body whose left-hip angle flips every six frames: with the wider Gaussian on its lying
    frames the swing there is damped, the standing frames keep the ordinary smoothing."""
    from nfl_gsplat.render import timeline as tl

    frames = list(range(0, 60))
    ground = {f: {1: np.array([0.1 * f, 0.0])} for f in frames}
    poses = {1: {}}
    for f in range(0, 60, 2):
        bp = np.zeros((21, 3)); bp[0, 0] = 0.6 if (f // 6) % 2 == 0 else -0.6
        poses[1][f] = (bp, tl.upright_from_yaw(0.0), np.zeros(10), "fused")
    kw = dict(default_pose=np.zeros((21, 3)), clamp_joints=False, orient_sigma=0)
    plain = tl.build_timeline(frames, ground, poses, **kw)
    wide = tl.build_timeline(frames, ground, poses, lying={(f, 1) for f in range(30, 60)}, lying_sigma_mult=4.0, **kw)
    def swing(t, lo, hi):
        return np.ptp([[s for s in t.states[f] if s.pid == 1][0].body_pose[0, 0] for f in range(lo, hi)])
    assert swing(wide, 36, 54) < 0.5 * swing(plain, 36, 54)             # damped where lying
    assert abs(swing(wide, 6, 24) - swing(plain, 6, 24)) < 1e-6          # untouched where standing


def test_unreadable_kit_ids_drops_short_unnamed_fragments_only():
    import pandas as pd
    from nfl_gsplat.render.timeline import PlayerState, Timeline, unreadable_kit_ids

    tl = Timeline(frames=list(range(0, 60)))
    rows = []
    for f in range(0, 60):
        for pid, n in ((1, 60), (2, 20), (3, 20), (4, 20)):
            if f < n:
                tl.states.setdefault(f, []).append(PlayerState(pid=pid, xy=np.zeros(2), body_pose=np.zeros((21, 3)), global_orient=np.zeros(3),
                                                              betas=np.zeros(10), source="sideline"))
                rows.append({"cam": "sideline", "track_id": pid, "global_player_id": pid, "frame": f,
                             "kit_margin": {1: 0.05, 2: 0.05, 3: 0.7, 4: 0.05}[pid]})
    df = pd.DataFrame(rows)
    # 1: long (kept); 2: short, unreadable, unnamed (dropped); 3: short but a clear kit (kept); 4: short, unreadable, but named (kept)
    assert unreadable_kit_ids(tl, df, {4: True}, margin=0.2, max_frames=40) == {2}


def test_impossible_runs_marks_a_sustained_teleport_and_not_a_sprint():
    from nfl_gsplat.render.timeline import PlayerState, Timeline, impossible_runs

    tl = Timeline(frames=list(range(0, 40)))
    for f in range(0, 40):
        x1 = 0.18 * f                                                   # id 1 sprints at 0.18 m/frame (11 m/s): legal
        x2 = 0.05 * f + (0.3 * (f - 10) if 10 <= f <= 15 else (1.5 if f > 15 else 0.0))   # id 2 slides 1.5 m over 10-15
        for pid, x in ((1, x1), (2, x2)):
            tl.states.setdefault(f, []).append(PlayerState(pid=pid, xy=np.array([x, 0.0]), body_pose=np.zeros((21, 3)),
                                                          global_orient=np.zeros(3), betas=np.zeros(10), source="sideline"))
    d = impossible_runs(tl, max_m=0.2, min_run=4)
    assert {p for _f, p in d} == {2} and {f for f, _p in d} == set(range(10, 16))
    assert impossible_runs(tl, max_m=0.2, min_run=7) == set()


def test_build_timeline_keep_exempts_a_vouched_unanchored_body_from_the_dedupe():
    """A body with no sideline sighting standing 0.8 m from a detected one is a duplicate to the
    dedupe -- unless a rule vouched for it (``keep``): the quarterback held under centre."""
    from nfl_gsplat.render import timeline as tl

    frames = list(range(0, 30))
    ground = {f: {1: np.array([0.0, 0.0]), 2: np.array([0.8, 0.0])} for f in frames}
    views = {f: {1: ["sideline"]} for f in frames}                      # only id 1 is ever detected
    poses = {1: {0: (np.zeros((21, 3)), tl.upright_from_yaw(0.0), np.zeros(10), "fused")},
             2: {0: (np.zeros((21, 3)), tl.upright_from_yaw(0.0), np.zeros(10), "fused")}}
    kw = dict(default_pose=np.zeros((21, 3)), views_by_frame=views, pose_smooth=0, pose_sigma=0, clamp_joints=False, orient_sigma=0)
    plain = tl.build_timeline(frames, ground, poses, **kw)
    kept = tl.build_timeline(frames, ground, poses, keep={f: {2} for f in frames}, **kw)
    assert all(not any(s.pid == 2 for s in plain.states[f]) for f in frames)
    assert all(any(s.pid == 2 for s in kept.states[f]) for f in frames)


def test_hold_to_end_keeps_a_body_that_ends_at_the_dead_ball_through_the_tail():
    import numpy as np

    from nfl_gsplat.render import timeline as tlm

    def st(pid, x):
        return tlm.PlayerState(pid=pid, xy=np.array([x, 0.0]), body_pose=np.zeros((21, 3)), global_orient=np.zeros(3),
                               betas=np.zeros(10), source="sideline")
    tl = tlm.Timeline(frames=list(range(630, 648)), states={f: [] for f in range(630, 648)})
    for f in range(630, 639):
        tl.states[f].append(st(71, float(f)))                  # the receiver, out of bounds after 638
    for f in range(630, 648):
        tl.states[f].append(st(5, 1.0))                          # a man drawn to the end
    for f in range(630, 633):
        tl.states[f].append(st(9, 2.0))                          # a fragment that ended long before the dead ball
    n = tlm.hold_to_end(tl, end=639, last_frame=647)
    assert n == 647 - 638
    assert all(any(s.pid == 71 and s.xy[0] == 638.0 for s in tl.states[f]) for f in range(639, 648))
    assert not any(s.pid == 9 for s in tl.states[640])
    assert sum(1 for s in tl.states[645] if s.pid == 5) == 1


def test_hold_to_end_drops_a_track_born_at_the_dead_ball_beside_the_held_man():
    import numpy as np

    from nfl_gsplat.render import timeline as tlm

    def st(pid, x, y=0.0):
        return tlm.PlayerState(pid=pid, xy=np.array([x, y]), body_pose=np.zeros((21, 3)), global_orient=np.zeros(3),
                               betas=np.zeros(10), source="sideline")
    tl = tlm.Timeline(frames=list(range(630, 648)), states={f: [] for f in range(630, 648)})
    for f in range(630, 639):
        tl.states[f].append(st(71, 10.0))                       # the receiver's track ends at 638
    for f in range(639, 648):
        tl.states[f].append(st(185, 12.0))                      # born at 639, 2 m away: the same man re-identified
        tl.states[f].append(st(2, 12.5))                        # a defender born there too: kept (other team)
    for f in range(630, 648):
        tl.states[f].append(st(5, 11.0))                        # a teammate present all along, 1 m away: kept
    teams = {71: "KC", 185: "KC", 2: "BAL", 5: "KC"}
    n = tlm.hold_to_end(tl, end=639, last_frame=647, teams=teams)
    assert n == 9
    for f in range(639, 648):
        pids = sorted(int(s.pid) for s in tl.states[f])
        assert pids == [2, 5, 71]
