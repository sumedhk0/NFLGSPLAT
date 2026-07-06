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
    import numpy as np
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
        return ["JOINT0", "JOINT1"]
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
