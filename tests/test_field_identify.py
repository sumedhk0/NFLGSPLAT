from __future__ import annotations

from nfl_gsplat.calibration.field_features import (
    DetectedFeatures, OCRNumber, YardLineSeg,
)
from nfl_gsplat.calibration.field_identify import IdentityState, identify_correspondences


def _vertical_line(x, H=1080):
    return YardLineSeg(p0=(float(x), 0.0), p1=(float(x), float(H)))


def _features_with_numbers():
    return DetectedFeatures(
        yard_lines=[_vertical_line(400), _vertical_line(800), _vertical_line(1200)],
        sidelines=[YardLineSeg((0, 200), (1920, 220)), YardLineSeg((0, 900), (1920, 950))],
        hashes=[(400, 560), (800, 560), (1200, 560)],
        numbers=[OCRNumber(40, (400, 560)), OCRNumber(50, (800, 560))],
        image_size=(1920, 1080),
    )


def test_identify_assigns_yardage_from_numbers():
    feats = _features_with_numbers()
    corrs, state = identify_correspondences(feats, prior=None)
    names = {c[0] for c in corrs}
    assert any(n.startswith("mid_50") for n in names)
    assert isinstance(state, IdentityState)
    assert state.line_yardage


def test_identity_propagates_when_no_number_visible():
    feats = _features_with_numbers()
    _, state = identify_correspondences(feats, prior=None)
    shifted = DetectedFeatures(
        yard_lines=[_vertical_line(410), _vertical_line(810), _vertical_line(1210)],
        sidelines=feats.sidelines,
        hashes=[(410, 560), (810, 560), (1210, 560)],
        numbers=[],
        image_size=(1920, 1080),
    )
    corrs2, state2 = identify_correspondences(shifted, prior=state)
    names2 = {c[0] for c in corrs2}
    assert any(n.startswith("mid_50") for n in names2)


def test_no_number_and_no_prior_returns_empty():
    feats = DetectedFeatures(
        yard_lines=[_vertical_line(400)], sidelines=[], hashes=[], numbers=[],
        image_size=(1920, 1080),
    )
    corrs, state = identify_correspondences(feats, prior=None)
    assert corrs == []
    assert not state.line_yardage
