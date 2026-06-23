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


def test_assemble_smooths_jitter():
    rng = np.random.default_rng(0)
    res = [_res(2600 + rng.normal(0, 30), 20 + rng.normal(0, 0.5)) for _ in range(20)]
    tr = assemble_track_from_results(res, width=1920, height=1080, max_gap=2)
    fx = tr.K[:, 0, 0]
    # Smoothed focal length should vary less frame-to-frame than the raw injected jitter (~30 std).
    assert np.mean(np.abs(np.diff(fx))) < 30.0
    assert np.isfinite(tr.K).all()


def test_sweep_tolerates_none_frames(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import identify_correspondences
    from nfl_gsplat.utils.meta import CalibHint

    def feats_for(i):
        base = 400 + 5 * i
        xs = [base, base + 400, base + 800]
        return DetectedFeatures(
            yard_lines=[YardLineSeg((float(x), 0.0), (float(x), 1080.0)) for x in xs],
            sidelines=[YardLineSeg((0, 100), (1920, 110)), YardLineSeg((0, 980), (1920, 990))],
            hashes=[(float(x), 540.0) for x in xs], numbers=[], image_size=(1920, 1080))

    def fake_register(feats, prior, image_size, **kw):
        corrs, state = identify_correspondences(feats, prior)
        return (object() if corrs else None), state
    monkeypatch.setattr(ra, "register_frame", fake_register)

    frames = [feats_for(0), None, feats_for(2), feats_for(3), None]
    hint = CalibHint(ref_frame=2, ref_x=410.0, yard=30, side="away", increasing="right")
    results = ra._register_sequence(frames, hint, (1920, 1080))
    assert len(results) == 5
    assert results[1] is None and results[4] is None         # None frames -> gaps, no crash
    assert results[2] is not None                            # ref frame registered


def test_sweep_seeds_and_propagates(monkeypatch):
    from nfl_gsplat.calibration import run_autocalib as ra
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import identify_correspondences
    from nfl_gsplat.utils.meta import CalibHint

    def feats_for(i):
        base = 400 + 5 * i                         # lines pan +5px/frame
        xs = [base, base + 400, base + 800]
        return DetectedFeatures(
            yard_lines=[YardLineSeg((float(x), 0.0), (float(x), 1080.0)) for x in xs],
            sidelines=[YardLineSeg((0, 100), (1920, 110)), YardLineSeg((0, 980), (1920, 990))],
            hashes=[(float(x), 540.0) for x in xs],
            numbers=[], image_size=(1920, 1080),
        )

    # Mock PnP: succeed iff identify produced correspondences; return the
    # propagated identity state so the sweep can carry labels frame-to-frame.
    def fake_register(feats, prior, image_size, **kw):
        corrs, state = identify_correspondences(feats, prior)
        return (object() if corrs else None), state

    monkeypatch.setattr(ra, "register_frame", fake_register)
    hint = CalibHint(ref_frame=2, ref_x=410.0, yard=30, side="away", increasing="right")
    results = ra._register_sequence([feats_for(i) for i in range(5)], hint, (1920, 1080))
    assert len(results) == 5
    assert all(r is not None for r in results)   # seeded at ref, propagated both ways
