from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration.cameras_io import CameraTrack
from nfl_gsplat.calibration.run_autocalib import assemble_track_from_results
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose


def _res(fx, z):
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    return CalibrationResult(
        intrinsics=CameraIntrinsics(fx, fx, 960, 540, 1920, 1080),
        pose=CameraPose(R=np.eye(3), t=np.array([0.0, 0.0, z])),
        rms_px=1.0, num_correspondences=8, refined_with_ba=True,
    )


def test_assemble_fills_short_gap():
    results = [_res(2600, 20.0), None, _res(2604, 22.0)]
    tr = assemble_track_from_results(results, width=1920, height=1080, max_gap=2)
    assert isinstance(tr, CameraTrack)
    assert tr.num_frames == 3
    assert np.isfinite(tr.K).all() and np.isfinite(tr.t).all()


def test_assemble_fails_loud_on_long_gap():
    results = [_res(2600, 20.0), None, None, None, _res(2604, 22.0)]
    with pytest.raises(CalibrationError, match="frames 1-3"):
        assemble_track_from_results(results, width=1920, height=1080, max_gap=2)


def test_assemble_tolerates_long_trailing_gap():
    # post-play frames (camera on crowd) are a trailing None run; docstring
    # promises clamp-extrapolation with conf=0, not a hard failure
    results = [_res(2600, 20.0), _res(2602, 21.0)] + [None] * 10
    tr = assemble_track_from_results(results, width=1920, height=1080, max_gap=2)
    assert tr.num_frames == 12
    assert (tr.conf[2:] == 0).all() and (tr.conf[:2] == 1).all()
    assert np.isfinite(tr.K).all() and np.isfinite(tr.t).all()


def test_assemble_tolerates_long_leading_gap():
    results = [None] * 8 + [_res(2600, 20.0), _res(2602, 21.0)]
    tr = assemble_track_from_results(results, width=1920, height=1080, max_gap=2)
    assert (tr.conf[:8] == 0).all()


def test_assemble_still_fails_loud_on_long_interior_gap():
    results = [_res(2600, 20.0)] + [None] * 4 + [_res(2604, 22.0)] + [None] * 3
    with pytest.raises(CalibrationError, match="frames 1-4"):
        assemble_track_from_results(results, width=1920, height=1080, max_gap=2)


def test_assemble_smooths_jitter():
    rng = np.random.default_rng(0)
    res = [_res(2600 + rng.normal(0, 30), 20 + rng.normal(0, 0.5)) for _ in range(20)]
    tr = assemble_track_from_results(res, width=1920, height=1080, max_gap=2)
    fx = tr.K[:, 0, 0]
    # Smoothed focal length should vary less frame-to-frame than the raw injected jitter (~30 std).
    assert np.mean(np.abs(np.diff(fx))) < 30.0
    assert np.isfinite(tr.K).all()


def _stub_state():
    # Simulates register_frame's returned IdentityState carrying a homography
    # (the propagation signal). These tests exercise _register_sequence's sweep
    # orchestration, not identify_correspondences' geometry (tested elsewhere).
    from nfl_gsplat.calibration.field_identify import IdentityState
    return IdentityState(homography=np.eye(3), anchor_label=("away", 30),
                         anchor_x=410.0, direction=1)


def test_sweep_tolerates_none_frames(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.utils.meta import CalibHint

    def fake_register(feats, prior, image_size, **kw):
        return object(), _stub_state()
    monkeypatch.setattr(ra, "register_frame", fake_register)

    frames = ["f0", None, "f2", "f3", None]                  # opaque; only None matters
    hint = CalibHint(ref_frame=2, ref_x=410.0, yard=30, side="away", increasing="right")
    results = ra._register_sequence(frames, hint, (1920, 1080))
    assert len(results) == 5
    assert results[1] is None and results[4] is None         # None frames -> gaps, no crash
    assert results[2] is not None                            # ref frame registered


def test_sweep_seeds_and_propagates(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.utils.meta import CalibHint

    seen_priors = []

    def fake_register(feats, prior, image_size, **kw):
        seen_priors.append(prior)
        return object(), _stub_state()
    monkeypatch.setattr(ra, "register_frame", fake_register)

    hint = CalibHint(ref_frame=2, ref_x=410.0, yard=30, side="away", increasing="right")
    results = ra._register_sequence([f"f{i}" for i in range(5)], hint, (1920, 1080))
    assert len(results) == 5
    assert all(r is not None for r in results)   # seeded at ref, propagated both ways
    # after the ref frame, propagation carries the stub state's homography forward
    assert any(p.homography is not None for p in seen_priors[1:])


def test_check_ckpt_classes_mismatch_raises():
    import pytest
    from nfl_gsplat.calibration.run_autocalib import _check_ckpt_classes
    from nfl_gsplat.errors import SetupError
    _check_ckpt_classes(["a", "b"], ["a", "b"])          # match → no raise
    with pytest.raises(SetupError, match="do not match"):
        _check_ckpt_classes(["a", "b"], ["a", "c"])


def test_learned_register_sequence_with_stub_detector():
    import numpy as np
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, NUMBER_CENTER_Y_M, _yardline_x_m
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

    intr = CameraIntrinsics(1400.0, 1400.0, 960, 540, 1920, 1080)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    pose = CameraPose(R=R, t=np.array([0.0, 6.0, 55.0]))
    pts = {}
    for y in [20, 30, 40]:
        for lr, sgn in (("left", +1), ("right", -1)):
            for tag, Y in (("hash", sgn * HASH_OFFSET_M),
                           ("number", sgn * NUMBER_CENTER_Y_M)):
                name = f"away_{y}_{lr}_{tag}"
                X = _yardline_x_m(f"away_{y}")
                uv = project_points(np.array([[X, Y, 0.0]]), intr.K(), pose.R, pose.t)[0]
                pts[name] = (float(uv[0]), float(uv[1]))

    def stub_detector(frame_bgr):
        return list(pts.items())

    results = ra._register_sequence_learned(
        ["f0", "f1", "f2"], detector=stub_detector, image_size=(1920, 1080))
    assert len(results) == 3 and all(r is not None for r in results)


def test_pretrained_register_sequence_with_stub_fusion(monkeypatch):
    # Frames with cached model kps register; frames without -> None (gap).
    from nfl_gsplat.calibration import run_autocalib as ra

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        return DetectedFeatures(yard_lines=["L"], sidelines=[], hashes=[(1.0, 2.0)],
                                numbers=[], image_size=(1920, 1080))
    monkeypatch.setattr(ra, "detect_field_features", fake_detect, raising=False)

    def fake_fuse(yard_lines, hashes, model_kps, *, territory, image_size, **kw):
        return [(f"c{i}", (10.0 * i, 5.0)) for i in range(8)] if model_kps else []
    monkeypatch.setattr(ra, "fuse_frame", fake_fuse, raising=False)

    def fake_register_corrs(corrs, image_size, **kw):
        return object() if len(corrs) >= 6 else None
    monkeypatch.setattr(ra, "_register_corrs", fake_register_corrs)

    kps = {0: [("30", 1.0, 2.0, 0.9)], 2: [("30", 1.0, 2.0, 0.9)]}   # frame 1 absent
    results = ra._register_sequence_pretrained(
        ["f0", "f1", "f2"], kps_by_frame=kps, territory="away",
        image_size=(1920, 1080))
    assert results[0] is not None and results[2] is not None
    assert results[1] is None                       # no cached kps -> gap


def test_pretrained_none_frame_is_gap(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    results = ra._register_sequence_pretrained(
        [None], kps_by_frame={0: [("30", 1.0, 2.0, 0.9)]}, territory="away",
        image_size=(1920, 1080))
    assert results == [None]


def test_solve_sweep_rescues_frames_with_prior(monkeypatch):
    # A frame that fails blind-init but succeeds given an intrinsics prior is
    # rescued by the forward sweep; an early failing frame (no earlier success
    # to seed it) is rescued by the backward fill.
    from nfl_gsplat.calibration import run_autocalib as ra

    calls = []

    class _Res:
        def __init__(self, tag):
            self.intrinsics = tag

    def fake_register_corrs(corrs, image_size, *, initial_intrinsics=None, **kw):
        calls.append((corrs, initial_intrinsics))
        if corrs == "blind_ok":
            return _Res("K1")
        if corrs == "needs_prior":
            return _Res("K2") if initial_intrinsics is not None else None
        return None

    monkeypatch.setattr(ra, "_register_corrs", fake_register_corrs)

    corrs_by_frame = {0: "needs_prior", 1: "blind_ok", 2: "needs_prior"}
    results = ra._solve_sweep(corrs_by_frame, 3, (1920, 1080))
    assert results[1] is not None          # blind success
    assert results[2] is not None          # forward prior rescue
    assert results[0] is not None          # backward prior rescue


def test_solve_sweep_propagates_identity_across_multiframe_gap(monkeypatch):
    # Frames 2, 3, 4 have model-vote correspondences too sparse to solve on
    # their own ("sparse" always fails, mirroring the real 77-frame gap where
    # model votes only pin 2 lines). Frame 5's model votes solve cleanly. Only
    # frames 2-4 carry cached classical detections (feats_by_frame) -- frames
    # 0/1 have none, so they cannot be rescued and must stay gaps. Unit-level:
    # predict_identities/correspondences_from_identities are stubbed so no
    # cv2/homography geometry runs here (that's covered in test_fuse_pretrained.py).
    from nfl_gsplat.calibration import run_autocalib as ra

    calls = []

    def fake_register_corrs(corrs, image_size, *, initial_intrinsics=None, **kw):
        calls.append(corrs)
        if corrs == "blind_ok":
            return _res(1400.0, 50.0)
        if isinstance(corrs, list):              # rescued via identity propagation
            return _res(1400.0, 50.0)
        return None                               # "sparse" (or missing) never solves alone
    monkeypatch.setattr(ra, "_register_corrs", fake_register_corrs)

    predict_calls = []

    def fake_predict_identities(yard_lines, rows, H_plane, **kw):
        predict_calls.append((yard_lines, H_plane is not None))
        return {0: "away_30"} if H_plane is not None else {}
    monkeypatch.setattr(ra, "predict_identities", fake_predict_identities)

    def fake_correspondences_from_identities(ident, yard_lines, rows):
        return [(f"c{i}", (0.0, 0.0)) for i in range(6)] if ident else []
    monkeypatch.setattr(ra, "correspondences_from_identities", fake_correspondences_from_identities)

    corrs_by_frame = {2: "sparse", 3: "sparse", 4: "sparse", 5: "blind_ok"}
    feats_by_frame = {
        2: ("yard_lines_2", "rows_2", "hashes_2"),
        3: ("yard_lines_3", "rows_3", "hashes_3"),
        4: ("yard_lines_4", "rows_4", "hashes_4"),
    }
    results = ra._solve_sweep(corrs_by_frame, 6, (1920, 1080), feats_by_frame=feats_by_frame)

    assert results[5] is not None                  # frame 5 solves on its own model votes
    # frames 2, 3, 4 all rescued in sequence during the backward sweep, chained
    # from frame 5's plane homography across the multi-frame gap
    assert results[4] is not None and results[3] is not None and results[2] is not None
    # frames 0/1 have no cached classical detections -> cannot be rescued -> stay gaps
    assert results[1] is None and results[0] is None
    # the propagation path was actually exercised on all three rescued frames
    assert len(predict_calls) >= 3


def test_build_pretrained_uses_joint_solve(monkeypatch):
    # phase 3 gate: build_autocalib_npz_pretrained must pass the sweep output
    # into solve_fixed_center and assemble ITS results, not the sweep's
    from nfl_gsplat.calibration import run_autocalib as ra

    captured = {}

    def fake_joint(corrs_by_frame, image_size, *, init_results, **kw):
        captured["init"] = init_results
        return ["JOINT0", "JOINT1"], False
    monkeypatch.setattr(ra, "solve_fixed_center", fake_joint)

    assembled = {}
    def fake_assemble(results, *, width, height, **kw):
        assembled["results"] = results
        return "TRACK"
    monkeypatch.setattr(ra, "assemble_track_from_results", fake_assemble)
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    class _Meta:
        num_frames, width, height = 2, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter([]))
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: {})

    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={"sideline": "v.mp4"}, fps=30.0,
        kps_json="kps.json", territory="away")
    assert assembled["results"] == ["JOINT0", "JOINT1"]     # joint output assembled
    assert "init" in captured                               # sweep fed the init


def test_pretrained_build_filters_static_hashes(monkeypatch):
    # Broadcast graphics (watermark/score bug) sit at the SAME pixel every
    # frame; a real field tick moves under pan/zoom. Phase 1 must census
    # across the whole play and strip the static candidate before fuse_frame
    # ever sees it (per spec 2026-07-06 task 5).
    from nfl_gsplat.calibration import run_autocalib as ra

    N = 20
    STATIC = (100.0, 50.0)

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        moving = (500.0 + 30.0 * frame, 400.0)
        return DetectedFeatures(yard_lines=["L"], sidelines=[],
                                hashes=[STATIC, moving], numbers=[],
                                image_size=(1920, 1080))
    monkeypatch.setattr(ra, "detect_field_features", fake_detect)

    captured = []

    def fake_fuse(yard_lines, hashes, model_kps, *, territory, image_size, **kw):
        captured.append(list(hashes))
        return []
    monkeypatch.setattr(ra, "fuse_frame", fake_fuse)

    monkeypatch.setattr(ra, "_solve_sweep", lambda *a, **kw: [None] * N)
    monkeypatch.setattr(ra, "solve_fixed_center", lambda *a, **kw: ([None] * N, False))
    monkeypatch.setattr(ra, "assemble_track_from_results", lambda *a, **kw: "TRACK")
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    class _Meta:
        num_frames, width, height = N, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())

    def fake_iter_frames(v, start_frame=0):
        for i in range(N):
            yield i, i     # "frame" payload is just the index; fake_detect uses it directly
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames", fake_iter_frames)

    kps = {i: [("30", 1.0, 2.0, 0.9)] for i in range(N)}
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: kps)

    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={"sideline": "v.mp4"}, fps=30.0,
        kps_json="kps.json", territory="away")

    assert len(captured) == N
    for i, hashes in enumerate(captured):
        assert STATIC not in hashes
        assert (500.0 + 30.0 * i, 400.0) in hashes


def test_pretrained_multicam_rejects_single_kps_path():
    # one shared cache can only match one camera's frame count -> fail loud
    from nfl_gsplat.calibration.run_autocalib import build_autocalib_npz_pretrained
    from nfl_gsplat.errors import SetupError
    with pytest.raises(SetupError, match="per-camera"):
        build_autocalib_npz_pretrained(
            play_dir=".", videos={"sideline": "s.mp4", "endzone": "e.mp4"},
            fps=30.0, kps_json="one.json", territory="away")


def test_pretrained_multicam_missing_camera_cache_fails_loud():
    from nfl_gsplat.calibration.run_autocalib import build_autocalib_npz_pretrained
    from nfl_gsplat.errors import SetupError
    with pytest.raises(SetupError, match="endzone"):
        build_autocalib_npz_pretrained(
            play_dir=".", videos={"sideline": "s.mp4", "endzone": "e.mp4"},
            fps=30.0, kps_json={"sideline": "s.json"}, territory="away")


def _pretrained_build_harness(monkeypatch, rotations=None, cam="endzone"):
    """Run build_autocalib_npz_pretrained with one fake camera and capture
    what reaches detection/fusion/joint-solve. Follows the monkeypatch
    pattern of test_build_pretrained_uses_joint_solve."""
    from nfl_gsplat.calibration import run_autocalib as ra

    captured = {}

    class _Meta:
        num_frames, width, height = 2, 1920, 1080

    frames = [np.full((1080, 1920, 3), i, np.uint8) for i in range(2)]
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter(enumerate(frames)))
    monkeypatch.setattr(
        "nfl_gsplat.calibration.roboflow_kps.load_kps_json",
        lambda p, expect_num_frames=None: {0: [("30", 10.0, 20.0, 0.9)],
                                           1: [("30", 11.0, 21.0, 0.9)]})

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        captured.setdefault("frame_shapes", []).append(frame.shape)
        return DetectedFeatures(yard_lines=[], sidelines=[], hashes=[],
                                numbers=[], image_size=frame.shape[:2][::-1])
    monkeypatch.setattr(ra, "detect_field_features", fake_detect)

    def fake_fuse(yard_lines, hashes, model_kps, *, territory, image_size, **kw):
        captured.setdefault("fuse_kps", []).append(model_kps)
        captured["fuse_image_size"] = image_size
        return [(f"c{i}", (float(i), 0.0)) for i in range(8)]
    monkeypatch.setattr(ra, "fuse_frame", fake_fuse)

    monkeypatch.setattr(ra, "_solve_sweep",
                        lambda *a, **k: [None, None], raising=False)

    def fake_joint(corrs_by_frame, image_size, *, init_results, **kw):
        captured["joint_image_size"] = image_size
        return ["R0", "R1"], False
    monkeypatch.setattr(ra, "solve_fixed_center", fake_joint)

    def fake_derotate(result, deg, orig_wh):
        captured.setdefault("derotated", []).append((result, deg, orig_wh))
        return result
    monkeypatch.setattr(ra, "derotate_result", fake_derotate)

    monkeypatch.setattr(ra, "assemble_track_from_results",
                        lambda results, *, width, height, **kw: ("TRACK", width, height))
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={cam: "v.mp4"}, fps=30.0,
        kps_json={cam: "k.json"}, territory="away", rotations=rotations)
    return captured


def test_pretrained_endzone_rotates_by_default(monkeypatch):
    cap = _pretrained_build_harness(monkeypatch, rotations=None, cam="endzone")
    # frames reach detection rotated 90 deg: (1080,1920,3) -> (1920,1080,3)
    assert all(s == (1920, 1080, 3) for s in cap["frame_shapes"])
    assert cap["fuse_image_size"] == (1080, 1920)      # rotated (W,H)
    assert cap["joint_image_size"] == (1080, 1920)
    # cached kps rotated: (10,20) in 1920x1080 under 90CW -> (1080-1-20, 10)
    assert cap["fuse_kps"][0][0][1:3] == (1059.0, 10.0)
    # every joint result de-rotated back to the original dims
    assert [d for (_r, d, _wh) in cap["derotated"]] == [90, 90]
    assert all(wh == (1920, 1080) for (_r, _d, wh) in cap["derotated"])


def test_pretrained_sideline_not_rotated(monkeypatch):
    cap = _pretrained_build_harness(monkeypatch, rotations=None, cam="sideline")
    assert all(s == (1080, 1920, 3) for s in cap["frame_shapes"])
    assert cap["fuse_image_size"] == (1920, 1080)
    assert cap["derotated"] == [] or all(d == 0 for (_r, d, _wh) in cap["derotated"])


def test_pretrained_rotations_override(monkeypatch):
    cap = _pretrained_build_harness(monkeypatch, rotations={"endzone": 0}, cam="endzone")
    assert all(s == (1080, 1920, 3) for s in cap["frame_shapes"])


def test_pretrained_error_names_camera(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.errors import CalibrationError

    class _Meta:
        num_frames, width, height = 2, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter([]))
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: {})
    with pytest.raises(CalibrationError, match="endzone"):
        ra.build_autocalib_npz_pretrained(
            play_dir=".", videos={"endzone": "e.mp4"}, fps=30.0,
            kps_json={"endzone": "e.json"}, territory="away")


def test_pretrained_rotates_player_boxes_for_masking(monkeypatch):
    # boxes are original-pixel; on a rotated (endzone) camera they must reach
    # detect_field_features rotated into the working frame
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.view_rotation import rotate_box

    captured = {"boxes": []}

    class _Meta:
        num_frames, width, height = 2, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(2)]
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter(enumerate(frames)))
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: {0: [("30", 10.0, 20.0, 0.9)],
                                                           1: [("30", 10.0, 20.0, 0.9)]})

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        captured["boxes"].append(player_boxes)
        return DetectedFeatures(yard_lines=[], sidelines=[], hashes=[],
                                numbers=[], image_size=frame.shape[:2][::-1])
    monkeypatch.setattr(ra, "detect_field_features", fake_detect)
    monkeypatch.setattr(ra, "fuse_frame",
                        lambda *a, **k: [(f"c{i}", (float(i), 0.0)) for i in range(8)])
    monkeypatch.setattr(ra, "_solve_sweep", lambda *a, **k: [None, None], raising=False)
    monkeypatch.setattr(ra, "solve_fixed_center", lambda *a, **k: (["R0", "R1"], False))
    monkeypatch.setattr(ra, "derotate_result", lambda r, deg, wh: r)
    monkeypatch.setattr(ra, "assemble_track_from_results",
                        lambda results, *, width, height, **kw: ("TRACK", width, height))
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    orig_box = (100.0, 200.0, 150.0, 400.0)

    def masks(cam):
        return lambda fidx: [orig_box]
    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={"endzone": "e.mp4"}, fps=30.0,
        kps_json={"endzone": "e.json"}, territory="away",
        masks_provider=masks, rotations={"endzone": 90})
    assert captured["boxes"][0] == [rotate_box(orig_box, 90, (1920, 1080))]


def test_pretrained_sideline_boxes_unrotated(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    captured = {"boxes": []}

    class _Meta:
        num_frames, width, height = 1, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter([(0, np.zeros((1080, 1920, 3), np.uint8))]))
    monkeypatch.setattr("nfl_gsplat.calibration.roboflow_kps.load_kps_json",
                        lambda p, expect_num_frames=None: {0: [("30", 1.0, 2.0, 0.9)]})

    def fake_detect(frame, *, cfg=None, player_boxes=None):
        from nfl_gsplat.calibration.field_features import DetectedFeatures
        captured["boxes"].append(player_boxes)
        return DetectedFeatures(yard_lines=[], sidelines=[], hashes=[], numbers=[],
                                image_size=(1920, 1080))
    monkeypatch.setattr(ra, "detect_field_features", fake_detect)
    monkeypatch.setattr(ra, "fuse_frame", lambda *a, **k: [("c", (1.0, 0.0))])
    monkeypatch.setattr(ra, "_solve_sweep", lambda *a, **k: [None], raising=False)
    monkeypatch.setattr(ra, "solve_fixed_center", lambda *a, **k: (["R0"], False))
    monkeypatch.setattr(ra, "derotate_result", lambda r, deg, wh: r)
    monkeypatch.setattr(ra, "assemble_track_from_results",
                        lambda results, *, width, height, **kw: "TRACK")
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    box = (10.0, 20.0, 30.0, 40.0)
    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={"sideline": "s.mp4"}, fps=30.0,
        kps_json={"sideline": "s.json"}, territory="away",
        masks_provider=lambda cam: (lambda f: [box]))    # rotations=None -> sideline deg 0
    assert captured["boxes"][0] == [box]                 # unrotated


def test_build_endzone_from_sideline_merges(monkeypatch, tmp_path):
    # sideline camera present in cameras.npz; endzone solved from cross-camera
    # stub; result merged as endzone_* without dropping sideline_*.
    import numpy as np
    import pandas as pd
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.cameras_io import CameraTrack, load_camera_track, write_camera_track

    T = 3
    sl = CameraTrack(K=np.repeat(np.eye(3)[None], T, 0), R=np.repeat(np.eye(3)[None], T, 0),
                     t=np.zeros((T, 3)), conf=np.ones(T), width=1920, height=1080)
    npz = tmp_path / "cameras.npz"
    write_camera_track(npz, {"sideline": sl}, fps=30.0)

    tracks = tmp_path / "tracks.parquet"
    pd.DataFrame({"frame": [0], "cam": ["endzone"], "foot_u": [1.0], "foot_v": [2.0],
                  "bbox_x1": [0.0], "bbox_y1": [0.0], "bbox_x2": [1.0], "bbox_y2": [1.0]}
                 ).to_parquet(tracks, index=False)

    # stub the heavy solve: return T frames of a fixed endzone camera
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
    def fake_solve(field_by, feet_by, image_size, *, init_C, view_deg=90, **kw):
        R = np.eye(3); t = np.array([100.0, 0.0, 40.0])
        return [CalibrationResult(intrinsics=CameraIntrinsics(2000, 2000, 540, 960, 1080, 1920),
                                  pose=CameraPose(R=R, t=t), rms_px=1.0,
                                  num_correspondences=8, refined_with_ba=True)] * T
    monkeypatch.setattr(ra, "solve_endzone_cross_camera", fake_solve, raising=False)
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda v: type("M", (), {"num_frames": T, "width": 1920, "height": 1080})())

    out = ra.build_endzone_from_sideline(
        play_dir=str(tmp_path), tracks_path=str(tracks), cameras_npz=str(npz),
        endzone_video="e.mp4", fps=30.0)
    cams = load_camera_track(out)
    assert "sideline" in cams and "endzone" in cams          # merged, sideline kept
    assert cams["endzone"].num_frames == T


def test_build_endzone_missing_sideline_fails_loud(tmp_path):
    from nfl_gsplat.calibration.run_autocalib import build_endzone_from_sideline
    from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
    from nfl_gsplat.errors import SetupError
    import numpy as np
    npz = tmp_path / "cameras.npz"
    write_camera_track(npz, {"endzone": CameraTrack(
        K=np.repeat(np.eye(3)[None], 2, 0), R=np.repeat(np.eye(3)[None], 2, 0),
        t=np.zeros((2, 3)), conf=np.ones(2), width=1920, height=1080)}, fps=30.0)
    with pytest.raises(SetupError, match="sideline"):
        build_endzone_from_sideline(play_dir=str(tmp_path), tracks_path=str(tmp_path / "t.parquet"),
                                    cameras_npz=str(npz), endzone_video="e.mp4", fps=30.0)


def test_build_endzone_identity_from_plays(tmp_path, monkeypatch):
    # Two play dirs share one physically-fixed endzone camera at C=(-112,0,24).
    # Each play carries a calibrated sideline CameraTrack + tracks.parquet whose
    # sideline/endzone foot pixels are projections of shared player_uids (sideline
    # from a static camera, endzone from the ground-truth endzone camera panning
    # frame-to-frame, mirroring endzone_multiplay's _play_corrs). No video I/O:
    # ffprobe_meta is monkeypatched.
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from nfl_gsplat.calibration.cameras_io import CameraTrack, load_camera_track, write_camera_track
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_identity_from_plays
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points

    def _look_at(C, target):
        fwd = np.asarray(target, float) - np.asarray(C, float); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd])

    def _sideline_track(C, target, n_frames, f=6000.0, wh=(1920, 1080)):
        R = _look_at(C, target); t = -R @ np.asarray(C, float)
        K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
        return CameraTrack(K=np.repeat(K[None], n_frames, 0), R=np.repeat(R[None], n_frames, 0),
                           t=np.repeat(t[None], n_frames, 0), conf=np.ones(n_frames),
                           width=wh[0], height=wh[1])

    C_true = np.array([-112.0, 0.0, 24.0])
    n_frames, n_players, wh = 25, 8, (1920, 1080)
    sl = _sideline_track([-3.6, 80.0, 36.0], [0, 0, 0], n_frames)
    K_sl, R_sl, t_sl = sl.K[0], sl.R[0], sl.t[0]

    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15), z_range=(10, 60),
                         focal_range=(1500, 3500))

    play_dirs = []
    for pl in range(2):
        rng = np.random.default_rng(pl)
        start_xy = rng.uniform([-20, -12], [20, 12], size=(n_players, 2))
        vel_xy = rng.uniform(-0.05, 0.05, size=(n_players, 2))

        rows = []
        for i in range(n_frames):
            tx = -20.0 + 30.0 * i / (n_frames - 1)
            R_ez = _look_at(C_true, [tx, 0.0, 0.0]); t_ez = -R_ez @ C_true
            f_ez = 2500.0 + 300.0 * i / (n_frames - 1)
            K_ez = CameraIntrinsics(f_ez, f_ez, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
            for p in range(n_players):
                xy = start_xy[p] + vel_xy[p] * i
                world_pt = np.array([[xy[0], xy[1], 0.0]])
                uv_sl = project_points(world_pt, K_sl, R_sl, t_sl)[0]
                uv_ez = project_points(world_pt, K_ez, R_ez, t_ez)[0]
                if not (np.isfinite(uv_sl).all() and np.isfinite(uv_ez).all()):
                    continue
                uid = f"2025_A_{p}"
                rows.append({"frame": i, "cam": "sideline", "track_id": p,
                            "foot_u": uv_sl[0], "foot_v": uv_sl[1], "player_uid": uid})
                rows.append({"frame": i, "cam": "endzone", "track_id": p,
                            "foot_u": uv_ez[0], "foot_v": uv_ez[1], "player_uid": uid})
        df = pd.DataFrame(rows)
        for c in TRACK_COLUMNS:
            if c not in df.columns:
                df[c] = -1 if c != "cam" else ""

        pdir = tmp_path / f"play_{pl}"
        pdir.mkdir()
        write_camera_track(pdir / "cameras.npz", {"sideline": sl}, fps=30.0)
        df.to_parquet(pdir / "tracks.parquet", index=False)
        play_dirs.append(pdir)

    monkeypatch.setattr(
        "nfl_gsplat.utils.video.ffprobe_meta",
        lambda p: SimpleNamespace(width=wh[0], height=wh[1], num_frames=n_frames, fps=30.0))

    written = build_endzone_identity_from_plays(play_dirs=play_dirs, prior=prior, fps=30.0)

    assert len(written) == 2
    for out_path in written:
        cams = load_camera_track(out_path)
        assert "sideline" in cams          # preserved
        assert "endzone" in cams
        ez = cams["endzone"]
        centers = np.array([ez.at(f)[1].center_world() for f in range(ez.num_frames)])
        valid = ez.conf > 0
        mean_center = centers[valid].mean(axis=0) if valid.any() else centers.mean(axis=0)
        assert -150 < mean_center[0] < -60    # inside the prior box, correct side


def _write_identity_play_dir(pdir, sl):
    """Minimal play dir for build_endzone_identity_from_plays: a valid sideline
    CameraTrack + an empty (but correctly-columned) tracks.parquet. No actual
    correspondences are needed for tests that only exercise the pre-solve
    resolution guard -- identity_correspondences tolerates an empty frame."""
    import pandas as pd
    from nfl_gsplat.calibration.cameras_io import write_camera_track
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS

    pdir.mkdir()
    write_camera_track(pdir / "cameras.npz", {"sideline": sl}, fps=30.0)
    df = pd.DataFrame(columns=[*TRACK_COLUMNS, "player_uid"])
    df.to_parquet(pdir / "tracks.parquet", index=False)


def test_build_endzone_identity_rejects_mismatched_resolutions(tmp_path, monkeypatch):
    # image_size used to be overwritten per-loop-iteration with no check, so
    # mismatched play resolutions silently solved every play against the LAST
    # play's (width, height). Must fail loud instead, naming the mismatch.
    from types import SimpleNamespace

    import numpy as np
    import pytest

    from nfl_gsplat.calibration.cameras_io import CameraTrack
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_identity_from_plays
    from nfl_gsplat.errors import SetupError

    T = 3
    sl = CameraTrack(K=np.repeat(np.eye(3)[None], T, 0), R=np.repeat(np.eye(3)[None], T, 0),
                     t=np.zeros((T, 3)), conf=np.ones(T), width=1920, height=1080)

    play_dirs = [tmp_path / "play_0", tmp_path / "play_1"]
    for pdir in play_dirs:
        _write_identity_play_dir(pdir, sl)

    sizes_by_video = {
        str(play_dirs[0] / "endzone.mp4"): (1920, 1080),
        str(play_dirs[1] / "endzone.mp4"): (1280, 720),
    }

    def fake_ffprobe_meta(video):
        w, h = sizes_by_video[str(video)]
        return SimpleNamespace(width=w, height=h, num_frames=T, fps=30.0)
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", fake_ffprobe_meta)

    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15), z_range=(10, 60),
                         focal_range=(1500, 3500))
    with pytest.raises(SetupError, match="resolution"):
        build_endzone_identity_from_plays(play_dirs=play_dirs, prior=prior, fps=30.0)


def test_build_endzone_missing_tracks_fails_loud(tmp_path):
    from nfl_gsplat.calibration.run_autocalib import build_endzone_from_sideline
    from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
    from nfl_gsplat.errors import SetupError
    import numpy as np
    T = 3
    npz = tmp_path / "cameras.npz"
    # Create a valid sideline camera (so the sideline check passes)
    sl = CameraTrack(K=np.repeat(np.eye(3)[None], T, 0), R=np.repeat(np.eye(3)[None], T, 0),
                     t=np.zeros((T, 3)), conf=np.ones(T), width=1920, height=1080)
    write_camera_track(npz, {"sideline": sl}, fps=30.0)
    # Point tracks_path at a nonexistent file
    with pytest.raises(SetupError, match="tracks"):
        build_endzone_from_sideline(play_dir=str(tmp_path), tracks_path=str(tmp_path / "nonexistent.parquet"),
                                    cameras_npz=str(npz), endzone_video="e.mp4", fps=30.0)


def _build_synthetic_endzone_clip(pdir, *, include_player_uid: bool = True,
                                  include_endzone_boxes: bool = True):
    """Shared scene builder for the mosaic-endzone tests: a calibrated
    sideline camera + tracks.parquet, and a real endzone clip (9 painted
    yard lines, 5-yd/4.572m spacing) rendered by warping a textured
    world-plane through each frame's TRUE planar homography for a fixed
    tripod camera that only zooms -- genuine perspective, not a 2D affine
    fake (decompose_homography's focal recovery is degenerate for a pure
    affine/orthographic view; a fake "zoom" would also leave nothing for
    SIFT-frame-to-frame homographies to be geometrically consistent with
    beyond the lines themselves). Returns a SimpleNamespace with everything
    a caller needs to invoke build_endzone_mosaic and check its output
    against ground truth: vid, K_by_frame, R, t, X_lines, ref_frame, wh, N,
    BASE_BGR, C_true.

    ``include_player_uid=False`` omits the player_uid column tracks.parquet
    would have, matching scripts/03b_detect_players.py's bare output before
    scripts/03c_identity_tracks.py ever runs (see Fix round 1, finding 1).

    ``include_endzone_boxes=False`` omits every cam=="endzone" row, matching
    a tracks.parquet whose player-detection step never ran on the endzone
    video -- the mosaic driver must fail loud on that (finding I5), since
    every boxes[idx] would silently come back empty and white jerseys would
    then be indistinguishable from paint."""
    import cv2
    import numpy as np
    import pandas as pd

    from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
    from nfl_gsplat.calibration.field_landmarks import HALF_WIDTH_M
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    from nfl_gsplat.utils.geometry import CameraIntrinsics

    pdir.mkdir(exist_ok=True)
    wh_sl = (640, 360)
    # sideline track (identity-ish) so the driver can read a yard prior
    K_sl = CameraIntrinsics(900.0, 900.0, wh_sl[0] / 2, wh_sl[1] / 2, *wh_sl).K()
    R_sl = np.eye(3); t_sl = np.array([0.0, 0.0, 30.0])
    sl = CameraTrack(K=np.repeat(K_sl[None], 12, 0), R=np.repeat(R_sl[None], 12, 0),
                     t=np.repeat(t_sl[None], 12, 0), conf=np.ones(12),
                     width=wh_sl[0], height=wh_sl[1])
    write_camera_track(pdir / "cameras.npz", {"sideline": sl}, fps=30.0)

    # tracks.parquet: sideline players spanning a known yard window
    rows = []
    for fr in range(12):
        for k in range(6):
            row = {"frame": fr, "cam": "sideline", "track_id": k,
                   "foot_u": 100 + 60 * k, "foot_v": 300, "bbox_x1": 0,
                   "bbox_y1": 0, "bbox_x2": 1, "bbox_y2": 1, "conf": 1}
            if include_player_uid:
                row["player_uid"] = f"u{k}"
            rows.append(row)
        if include_endzone_boxes:
            # A tiny placeholder box per frame -- not meant to mask anything
            # meaningful in this synthetic scene, just enough for a real
            # cam=="endzone" row to exist (finding I5's precondition).
            rows.append({"frame": fr, "cam": "endzone", "track_id": 0,
                        "foot_u": 0, "foot_v": 0, "bbox_x1": 0, "bbox_y1": 0,
                        "bbox_x2": 1, "bbox_y2": 1, "conf": 1})
    df = pd.DataFrame(rows)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c != "cam" else ""
    df.to_parquet(pdir / "tracks.parquet", index=False)

    # --- real perspective endzone camera on a tripod: fixed centre, zooms ---
    def look_at(C, target):
        fwd = np.asarray(target, float) - np.asarray(C, float); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd])

    N = 12
    wh = (1200, 675)
    C_true = np.array([-130.0, 0.0, 50.0])           # inside the prior box below
    R = look_at(C_true, [-27.0, 0.0, 0.0])
    t = -R @ C_true
    ref_frame = 6

    def fx_at(fr):
        return 1050.0 + (150.0 / (N - 1)) * fr        # gentle zoom, stays in prior.focal_range

    X_lines = [-45.72 + 4.572 * k for k in range(9)]  # 9 real yard lines, 5 yd/4.572m apart

    # World-plane texture: green field + noise + scattered blobs (SIFT needs
    # real corner features -- pure lines alone are an aperture-problem trap)
    # + the painted yard lines, at a fixed meters-per-pixel resolution. Each
    # frame is rendered by warping this ONE texture through that frame's true
    # world(z=0)->image homography, so every frame (including the background
    # texture, not just the lines) is exactly consistent with a single
    # tripod camera -- exactly what register_to_reference assumes.
    rng = np.random.default_rng(0)
    mpp = 0.1
    Xmin, Xmax, Ymin, Ymax = -60.0, 5.0, -32.0, 32.0
    Wpx, Hpx = int((Xmax - Xmin) / mpp), int((Ymax - Ymin) / mpp)
    # (40, 90, 60): B != R (deliberately asymmetric) -- lets a spy tell BGR
    # from RGB by reading a background pixel back (see test below). NOTE:
    # asymmetry alone does not make a test catch a dropped RGB->BGR
    # conversion: accumulate_field_paint's white threshold spans the FULL hue
    # range (0..180), and HSV's V/S are both permutation-invariant to
    # swapping two channels (they depend only on max/min of the channel
    # VALUES, never on which channel holds which value) -- provably no scene
    # color can make that threshold care. SIFT registration also turned out
    # empirically robust to the swap (measured, including an extreme
    # 110-value B/R split: reprojection error barely moved). A spy on the
    # frames dict is the only reliable proof (see the calling test).
    BASE_BGR = (40, 90, 60)
    world_tex = np.full((Hpx, Wpx, 3), BASE_BGR, np.uint8)
    noise = rng.normal(0, 12, size=world_tex.shape).astype(np.int16)
    world_tex = np.clip(world_tex.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(400):
        cx_, cy_ = int(rng.integers(0, Wpx)), int(rng.integers(0, Hpx))
        shade = int(rng.integers(-30, 30))
        color = tuple(int(np.clip(c + shade, 0, 255)) for c in BASE_BGR)
        cv2.circle(world_tex, (cx_, cy_), int(rng.integers(2, 6)), color, -1)

    def w2px(X, Y):
        return ((X - Xmin) / mpp, (Y - Ymin) / mpp)
    for X in X_lines:
        px, _ = w2px(X, 0.0)
        _, py0 = w2px(X, -HALF_WIDTH_M); _, py1 = w2px(X, HALF_WIDTH_M)
        cv2.line(world_tex, (int(px), int(py0)), (int(px), int(py1)), (255, 255, 255), 4)

    M_px_to_world = np.array([[mpp, 0, Xmin], [0, mpp, Ymin], [0, 0, 1]])

    vid = pdir / "endzone.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, wh)
    K_by_frame = {}                       # true per-frame K, kept for the propagate check below
    for fr in range(N):
        K = CameraIntrinsics(fx_at(fr), fx_at(fr), wh[0] / 2, wh[1] / 2, *wh).K()
        K_by_frame[fr] = K
        Hwi = K @ np.column_stack([R[:, 0], R[:, 1], t])       # world(z=0) -> frame fr
        vw.write(cv2.warpPerspective(world_tex, Hwi @ M_px_to_world, wh, flags=cv2.INTER_LINEAR))
    vw.release()

    from types import SimpleNamespace
    return SimpleNamespace(vid=vid, K_by_frame=K_by_frame, R=R, t=t, X_lines=X_lines,
                           ref_frame=ref_frame, wh=wh, N=N, BASE_BGR=BASE_BGR, C_true=C_true)


def test_build_endzone_mosaic_writes_endzone_track(tmp_path, monkeypatch):
    """Synthetic endzone clip of a textured, painted field (9 real yard lines,
    5-yd/4.572m spacing) viewed by a genuine pinhole camera on a fixed tripod
    that only zooms. The driver must register, accumulate, detect+label
    lines, solve a reference camera, propagate, and write an endzone track
    that preserves the sideline, recovering close to the true tripod centre
    AND reprojecting correctly per sampled frame (not just at the shared
    centre -- see the propagate note below)."""
    from types import SimpleNamespace

    import numpy as np

    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.utils.geometry import project_points

    pdir = tmp_path / "play_x"
    scene = _build_synthetic_endzone_clip(pdir)
    wh, N, ref_frame = scene.wh, scene.N, scene.ref_frame
    K_by_frame, R, t, X_lines = scene.K_by_frame, scene.R, scene.t, scene.X_lines
    K_ref = K_by_frame[ref_frame]

    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=wh[0], height=wh[1],
                                                  num_frames=N, fps=30.0))

    # Anchors: true pixel projection (at the reference frame) of the two
    # OUTERMOST yard lines' midpoints -- an operator would read the equivalent
    # off the accumulated mosaic PNG on a real first run.
    anchor_uv = project_points(
        np.array([[X_lines[0], 0.0, 0.0], [X_lines[-1], 0.0, 0.0]]), K_ref, R, t)
    anchors = (((float(anchor_uv[0][0]), float(anchor_uv[0][1])), X_lines[0]),
              ((float(anchor_uv[1][0]), float(anchor_uv[1][1])), X_lines[-1]))

    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(400, 2000))

    # Spy on the frames dict the driver hands to register_to_reference: this
    # is the actual load-bearing proof that iter_frames' RGB is converted to
    # BGR before use, not the scene's color asymmetry (see _build_synthetic_
    # endzone_clip's note). A dropped cv2.cvtColor swaps B<->R identically on
    # every frame; verified (temporarily dropping the conversion) that this
    # assertion then fails -- see task-5-report.md "Fix round 1" for the
    # RED/GREEN evidence.
    import nfl_gsplat.calibration.endzone_mosaic as em_mod
    captured_frames = {}
    _orig_register = em_mod.register_to_reference
    def _spy_register(frames, **kw):
        captured_frames.update(frames)
        return _orig_register(frames, **kw)
    monkeypatch.setattr(em_mod, "register_to_reference", _spy_register)

    out = build_endzone_mosaic(
        play_dir=pdir, tracks_path=pdir / "tracks.parquet",
        cameras_npz=pdir / "cameras.npz", endzone_video=scene.vid, fps=30.0,
        prior=prior, anchors=anchors, stride=2, ref_frame=ref_frame)

    sample = captured_frames[ref_frame]
    flat = sample.reshape(-1, 3).astype(np.float64)
    bg = flat[(flat.max(axis=1) < 200) & (flat.sum(axis=1) > 0)]   # exclude white lines + black border
    mean_bgr = bg.mean(axis=0)
    assert abs(mean_bgr[0] - scene.BASE_BGR[0]) < 15    # B channel really is B, not R
    assert abs(mean_bgr[2] - scene.BASE_BGR[2]) < 15    # R channel really is R, not B

    cams = load_camera_track(out)
    assert "sideline" in cams and "endzone" in cams          # merged, sideline preserved
    ez = cams["endzone"]
    assert (ez.conf > 0).sum() >= 1
    centers = np.array([ez.at(f)[1].center_world() for f in range(ez.num_frames)])
    valid = ez.conf > 0
    mean_center = centers[valid].mean(axis=0)
    # recovered close to the true fixed tripod centre -- proves the chain
    # (register -> accumulate -> detect -> label -> solve -> propagate)
    # actually recovered real geometry, not just "ran without crashing".
    #
    # NOTE: this centre check alone is structurally blind to propagate()'s
    # per-frame decomposition: propagate sets pose.t = -R @ C from ref_cam's
    # C, and CameraPose.center_world() is -R.T @ t, which recovers exactly C
    # for ANY valid rotation R -- so a broken per-frame R/K (e.g. composing
    # with H instead of inv(H)) would still leave this assertion green. The
    # reprojection check below is the one that actually exercises propagate's
    # per-frame product; verified (temporarily changing inv(H)->H inside
    # propagate) that it fails while this centre check alone would not --
    # see task-5-report.md "Fix round 1" for the RED/GREEN evidence.
    assert np.linalg.norm(mean_center - scene.C_true) < 5.0

    # Reproject a known world point through each SAMPLED (non-interpolated)
    # frame's recovered (K, R, t) and compare to where it was truly rendered
    # at that frame's true (K, R, t) -- this is sensitive to propagate's
    # per-frame homography decomposition, not just the shared centre. Picking
    # an interior line (not one of the two anchors) so the check is not
    # trivially satisfied by the anchor correspondences themselves.
    sampled_frames = list(range(0, N, 2))                     # matches stride=2 below
    test_pt = np.array([[X_lines[3], 0.0, 0.0]])
    errs = []
    for fr in sampled_frames:
        intr, pose = ez.at(fr)
        true_uv = project_points(test_pt, K_by_frame[fr], R, t)[0]
        rec_uv = project_points(test_pt, intr.K(), pose.R, pose.t)[0]
        errs.append(float(np.linalg.norm(true_uv - rec_uv)))
    assert np.mean(errs) < 1.5, f"propagated per-frame cameras reproject off-target: {errs}"


def test_build_endzone_mosaic_no_anchor_dumps_diag_and_fails_loud(tmp_path, monkeypatch):
    """First run for a game: no endzone_anchor yet. The driver must reach the
    diag-mosaic dump and raise SetupError naming it and the outermost-two-line
    instructions -- not an opaque KeyError from the identity-only yard-range
    validator (Fix round 1, finding 1: that call used to run BEFORE the
    anchors-None check, and field_positions_by_uid's groupby("player_uid")
    raises a bare KeyError on tracks.parquet without one). diag_dir is
    overridable (finding 3) so this never touches a real path on disk."""
    from types import SimpleNamespace

    import cv2
    import pytest

    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.errors import SetupError

    pdir = tmp_path / "play_y"
    scene = _build_synthetic_endzone_clip(pdir)
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=scene.wh[0], height=scene.wh[1],
                                                  num_frames=scene.N, fps=30.0))
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(400, 2000))
    diag_dir = tmp_path / "diag"
    with pytest.raises(SetupError, match="OUTERMOST TWO"):
        build_endzone_mosaic(
            play_dir=pdir, tracks_path=pdir / "tracks.parquet",
            cameras_npz=pdir / "cameras.npz", endzone_video=scene.vid, fps=30.0,
            prior=prior, anchors=None, stride=2, ref_frame=scene.ref_frame,
            diag_dir=diag_dir)

    png = diag_dir / f"{pdir.name}_mosaic.png"
    assert png.exists()
    img = cv2.imread(str(png))
    assert img is not None and img.ndim == 3          # BGR, detected lines drawn on it
    assert img.shape[:2] == (scene.wh[1], scene.wh[0])


def test_build_endzone_mosaic_works_without_player_uid(tmp_path, monkeypatch):
    """A bare tracks.parquet (scripts/03b_detect_players.py output, before
    scripts/03c_identity_tracks.py has ever run) has no player_uid column.
    field_positions_by_uid's proj.groupby("player_uid") raises a bare
    KeyError on that -- the mosaic driver must not let it leak: the sideline
    yard-range is a soft VALIDATOR only, and its absence must not block a run
    that otherwise has everything it needs (a real endzone_anchor). Fix round
    1, finding 1."""
    from types import SimpleNamespace

    import numpy as np

    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.utils.geometry import project_points

    pdir = tmp_path / "play_z"
    scene = _build_synthetic_endzone_clip(pdir, include_player_uid=False)
    wh, N, ref_frame = scene.wh, scene.N, scene.ref_frame
    K_by_frame, R, t, X_lines = scene.K_by_frame, scene.R, scene.t, scene.X_lines
    K_ref = K_by_frame[ref_frame]

    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=wh[0], height=wh[1],
                                                  num_frames=N, fps=30.0))

    anchor_uv = project_points(
        np.array([[X_lines[0], 0.0, 0.0], [X_lines[-1], 0.0, 0.0]]), K_ref, R, t)
    anchors = (((float(anchor_uv[0][0]), float(anchor_uv[0][1])), X_lines[0]),
              ((float(anchor_uv[1][0]), float(anchor_uv[1][1])), X_lines[-1]))
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(400, 2000))

    out = build_endzone_mosaic(
        play_dir=pdir, tracks_path=pdir / "tracks.parquet",
        cameras_npz=pdir / "cameras.npz", endzone_video=scene.vid, fps=30.0,
        prior=prior, anchors=anchors, stride=2, ref_frame=ref_frame)

    cams = load_camera_track(out)
    assert "endzone" in cams and (cams["endzone"].conf > 0).sum() >= 1


def test_build_endzone_mosaic_picks_largest_field_extent_as_reference(tmp_path, monkeypatch):
    """I3: the mosaic is clipped to the reference frame's FOV, so an
    arbitrary median-index pick can select a more-zoomed-in frame than
    necessary and clip yard lines at the image border. build_endzone_mosaic
    (ref_frame=None) must instead pick the SAMPLED frame with the largest
    field-extent score -- computed independently here from the decoded
    frames, not assumed."""
    from types import SimpleNamespace

    import cv2
    import numpy as np

    import nfl_gsplat.calibration.endzone_mosaic as em_mod
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.utils.geometry import project_points
    from nfl_gsplat.utils.video import iter_frames

    pdir = tmp_path / "play_ref"
    scene = _build_synthetic_endzone_clip(pdir)
    wh, N = scene.wh, scene.N
    K_by_frame, R, t, X_lines = scene.K_by_frame, scene.R, scene.t, scene.X_lines

    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=wh[0], height=wh[1],
                                                  num_frames=N, fps=30.0))

    # Independently compute the field-extent score for every stride=2 sampled
    # frame, exactly as the driver will decode them, to know in advance which
    # frame SHOULD be picked (and to build valid anchors for it -- anchors
    # are pixel coordinates in the reference frame).
    scores = {}
    for idx, rgb in iter_frames(scene.vid, stride=2):
        scores[idx] = em_mod.field_extent_score(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    expected_ref = max(scores, key=scores.get)
    median_idx = sorted(scores)[len(scores) // 2]
    assert expected_ref != median_idx, \
        "test scene must actually distinguish largest-extent from median-index"

    K_ref = K_by_frame[expected_ref]
    anchor_uv = project_points(
        np.array([[X_lines[0], 0.0, 0.0], [X_lines[-1], 0.0, 0.0]]), K_ref, R, t)
    anchors = (((float(anchor_uv[0][0]), float(anchor_uv[0][1])), X_lines[0]),
              ((float(anchor_uv[1][0]), float(anchor_uv[1][1])), X_lines[-1]))
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(400, 2000))

    captured = {}
    real_register = em_mod.register_to_reference

    def spy_register(frames, *, ref_idx, **kw):
        captured["ref_idx"] = ref_idx
        return real_register(frames, ref_idx=ref_idx, **kw)
    monkeypatch.setattr(em_mod, "register_to_reference", spy_register)

    build_endzone_mosaic(
        play_dir=pdir, tracks_path=pdir / "tracks.parquet",
        cameras_npz=pdir / "cameras.npz", endzone_video=scene.vid, fps=30.0,
        prior=prior, anchors=anchors, stride=2)          # ref_frame NOT forced

    assert captured["ref_idx"] == expected_ref


def test_build_endzone_mosaic_rejects_z_range_spanning_zero(tmp_path, monkeypatch):
    """I4: the both-orientations mirror search relies ENTIRELY on the mirror
    flipping C_z negative. A z_range spanning zero would silently accept a
    mirrored camera; the driver must fail loud before doing any work."""
    from types import SimpleNamespace

    import pytest

    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.errors import SetupError

    pdir = tmp_path / "play_zneg"
    scene = _build_synthetic_endzone_clip(pdir)
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=scene.wh[0], height=scene.wh[1],
                                                  num_frames=scene.N, fps=30.0))
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(-10, 60), focal_range=(400, 2000))   # spans zero
    with pytest.raises(SetupError, match="mirror"):
        build_endzone_mosaic(
            play_dir=pdir, tracks_path=pdir / "tracks.parquet",
            cameras_npz=pdir / "cameras.npz", endzone_video=scene.vid, fps=30.0,
            prior=prior, anchors=None, stride=2, ref_frame=scene.ref_frame)


def test_build_endzone_mosaic_no_endzone_boxes_fails_loud(tmp_path, monkeypatch):
    """I5: if tracks.parquet has no cam=="endzone" rows, every boxes[idx] is
    empty and the entire player-masking premise silently evaporates (white
    jerseys are bright + low-saturation, indistinguishable from paint to
    _white_mask). Must fail loud pointing at scripts/03b_detect_players.py."""
    from types import SimpleNamespace

    import pytest

    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.errors import SetupError

    pdir = tmp_path / "play_noboxes"
    scene = _build_synthetic_endzone_clip(pdir, include_endzone_boxes=False)
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=scene.wh[0], height=scene.wh[1],
                                                  num_frames=scene.N, fps=30.0))
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(400, 2000))
    with pytest.raises(SetupError, match="03b_detect_players"):
        build_endzone_mosaic(
            play_dir=pdir, tracks_path=pdir / "tracks.parquet",
            cameras_npz=pdir / "cameras.npz", endzone_video=scene.vid, fps=30.0,
            prior=prior, anchors=None, stride=2, ref_frame=scene.ref_frame)


def test_build_endzone_mosaic_diag_dir_defaults_to_play_dir(tmp_path, monkeypatch):
    """I6: diag_dir must default to play_dir, never a hardcoded personal
    path -- the first-run mosaic dump must land somewhere real without an
    operator having to pass --diag-dir."""
    from types import SimpleNamespace

    import pytest

    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    from nfl_gsplat.calibration.run_autocalib import build_endzone_mosaic
    from nfl_gsplat.errors import SetupError

    pdir = tmp_path / "play_diagdefault"
    scene = _build_synthetic_endzone_clip(pdir)
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta",
                        lambda p: SimpleNamespace(width=scene.wh[0], height=scene.wh[1],
                                                  num_frames=scene.N, fps=30.0))
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15),
                         z_range=(10, 60), focal_range=(400, 2000))
    with pytest.raises(SetupError, match="OUTERMOST TWO"):
        build_endzone_mosaic(
            play_dir=pdir, tracks_path=pdir / "tracks.parquet",
            cameras_npz=pdir / "cameras.npz", endzone_video=scene.vid, fps=30.0,
            prior=prior, anchors=None, stride=2, ref_frame=scene.ref_frame)

    png = pdir / f"{pdir.name}_mosaic.png"
    assert png.exists(), "diag PNG must land under play_dir by default"
