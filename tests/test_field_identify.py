from __future__ import annotations

from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
from nfl_gsplat.calibration.field_identify import (
    identify_correspondences, seed_state_from_hint,
)
from nfl_gsplat.utils.meta import CalibHint


def _vline(x, H=1080):
    return YardLineSeg(p0=(float(x), 0.0), p1=(float(x), float(H)))


def _feats(xs, hashes=None, sidelines=None):
    return DetectedFeatures(
        yard_lines=[_vline(x) for x in xs],
        sidelines=sidelines or [YardLineSeg((0, 200), (1920, 220)),
                                YardLineSeg((0, 900), (1920, 950))],
        hashes=hashes or [],
        numbers=[],
        image_size=(1920, 1080),
    )


def test_seed_from_hint_labels_by_spacing_and_direction():
    feats = _feats([400, 800, 1200])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    labels = set(state.line_yardage.values())
    assert ("away", 30) in labels
    assert ("away", 25) in labels and ("away", 35) in labels


def test_seed_crosses_50_to_home_when_increasing():
    feats = _feats([400, 800, 1200])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=45, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    labels = set(state.line_yardage.values())
    assert ("away", 45) in labels
    assert ("mid", 50) in labels


def test_identify_propagates_from_prior_with_hashes():
    feats = _feats([400, 800, 1200], hashes=[(400, 560), (800, 560), (1200, 560)])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state0 = seed_state_from_hint(feats, hint)
    corrs, state = identify_correspondences(feats, state0)
    names = {c[0] for c in corrs}
    assert any(n.startswith("away_30") for n in names)
    shifted = _feats([410, 810, 1210], hashes=[(410, 560), (810, 560), (1210, 560)])
    corrs2, _ = identify_correspondences(shifted, state)
    assert any(n.startswith("away_30") for n in {c[0] for c in corrs2})


def test_identify_without_prior_returns_empty():
    feats = _feats([400])
    corrs, state = identify_correspondences(feats, None)
    assert corrs == [] and not state.line_yardage


def test_line_x_at_vertical_is_constant():
    from nfl_gsplat.calibration.field_identify import line_x_at
    from nfl_gsplat.calibration.field_features import YardLineSeg
    seg = YardLineSeg((500.0, 0.0), (500.0, 1080.0))
    assert abs(line_x_at(seg, 0) - 500.0) < 1e-6
    assert abs(line_x_at(seg, 540) - 500.0) < 1e-6


def test_line_x_at_diagonal_interpolates():
    from nfl_gsplat.calibration.field_identify import line_x_at
    from nfl_gsplat.calibration.field_features import YardLineSeg
    seg = YardLineSeg((400.0, 0.0), (600.0, 1000.0))
    assert abs(line_x_at(seg, 0) - 400.0) < 1e-6
    assert abs(line_x_at(seg, 500) - 500.0) < 1e-6
    assert abs(line_x_at(seg, 1000) - 600.0) < 1e-6
