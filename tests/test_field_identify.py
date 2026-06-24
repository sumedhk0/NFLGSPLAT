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


def test_merge_lines_dedupes_same_diagonal_line():
    from nfl_gsplat.calibration.field_identify import _merge_lines, line_x_at
    from nfl_gsplat.calibration.field_features import YardLineSeg
    a = YardLineSeg((500.0, 0.0), (520.0, 1080.0))      # x@540 ≈ 510
    b = YardLineSeg((505.0, 200.0), (515.0, 760.0))     # x@540 ≈ 510
    far = YardLineSeg((800.0, 0.0), (820.0, 1080.0))    # x@540 ≈ 810
    merged = _merge_lines([a, b, far], tol=25.0, ref_y=540.0)
    xs = sorted(round(line_x_at(s, 540.0)) for s in merged)
    assert len(merged) == 2
    assert any(abs(x - 510) < 6 for x in xs)
    assert any(abs(x - 810) < 6 for x in xs)


def test_fit_hash_rows_finds_two_rows():
    import numpy as np
    from nfl_gsplat.calibration.field_identify import fit_hash_rows
    rng = np.random.default_rng(0)
    pts = []
    for x in range(200, 1400, 20):
        pts.append((float(x), 360.0 + rng.normal(0, 1.0)))
        pts.append((float(x), 620.0 + rng.normal(0, 1.0)))
    pts += [(700.0, 150.0), (300.0, 900.0)]
    rows = fit_hash_rows(pts, image_width=1920)
    assert len(rows) == 2
    ys = sorted(0.5 * (r.p0[1] + r.p1[1]) for r in rows)
    assert abs(ys[0] - 360) < 10 and abs(ys[1] - 620) < 10
    assert min(r.p0[0] for r in rows) <= 1 and max(r.p1[0] for r in rows) >= 1919


def test_fit_hash_rows_too_few_returns_empty():
    from nfl_gsplat.calibration.field_identify import fit_hash_rows
    assert fit_hash_rows([(10.0, 20.0), (30.0, 40.0)], image_width=1920) == []
