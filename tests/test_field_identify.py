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


def test_fold_direction_when_decreasing():
    from nfl_gsplat.calibration.field_features import DetectedFeatures, OCRNumber, YardLineSeg
    from nfl_gsplat.calibration.field_identify import identify_correspondences
    def vline(x): return YardLineSeg((float(x), 0.0), (float(x), 1080.0))
    # numbers decrease left->right: 50 at x=400, 40 at x=800 => inc<0.
    # line at x=0 (index 0) has yd_signed=60 -> folds to 40 -> must be AWAY, not home.
    feats = DetectedFeatures(
        yard_lines=[vline(0), vline(400), vline(800)],
        sidelines=[YardLineSeg((0, 200), (1920, 220)), YardLineSeg((0, 900), (1920, 950))],
        hashes=[(0, 560), (400, 560), (800, 560)],
        numbers=[OCRNumber(50, (400, 560)), OCRNumber(40, (800, 560))],
        image_size=(1920, 1080),
    )
    corrs, _ = identify_correspondences(feats, prior=None)
    names = {c[0] for c in corrs}
    # x=0 (yd_signed=60) must fold to away_40, not home_40.
    # x=800 (yd_signed=40, inc<0) legitimately becomes home_40 — that's correct.
    assert any(n.startswith("away_40") for n in names)
