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


def test_seed_records_anchor_and_direction():
    feats = _feats([400, 800, 1200])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    assert state.anchor_label == ("away", 30)
    assert state.anchor_x == 800.0
    assert state.direction == 1
    assert state.homography is None


def test_seed_direction_left_and_mid_normalization():
    feats = _feats([400, 800, 1200])
    hint = CalibHint(ref_frame=0, ref_x=800, yard=50, side="home", increasing="left")
    state = seed_state_from_hint(feats, hint)
    assert state.anchor_label == ("mid", 50)        # yardline_label normalizes 50 → mid
    assert state.direction == -1


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


def test_identify_emits_hash_correspondences_via_consensus():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, _yardline_x_m
    from nfl_gsplat.utils.meta import CalibHint
    world = np.array([[-25.0, 3.0], [25.0, 3.0], [25.0, -3.0], [-25.0, -3.0]], np.float32)
    image = np.array([[260, 180], [1660, 210], [1520, 900], [380, 930]], np.float32)
    H = cv2.getPerspectiveTransform(world, image)

    def proj(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
        return (float(p[0]), float(p[1]))
    yards = [20, 25, 30, 35, 40]
    lines = [YardLineSeg(proj(_yardline_x_m(f"away_{y}"), +HASH_OFFSET_M),
                         proj(_yardline_x_m(f"away_{y}"), -HASH_OFFSET_M)) for y in yards]
    hashes = []                                  # dense 1-yard ticks (≥6/row)
    for X in np.linspace(-28.0, -9.0, 24):
        hashes += [proj(X, +HASH_OFFSET_M), proj(X, -HASH_OFFSET_M)]
    feats = DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                             numbers=[], image_size=(1920, 1080))
    ref_x = 0.5 * (lines[2].p0[0] + lines[2].p1[0])
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    state0 = seed_state_from_hint(feats, hint)
    corrs, state = identify_correspondences(feats, state0)
    names = {n for n, _ in corrs}
    assert "away_30_left_hash" in names and "away_30_right_hash" in names
    assert state.homography is not None


def test_identify_recovers_all_lines_rejects_noise_homography_under_2px():
    # This cycle's deliverable: consistent correspondences + a verified field
    # homography. (Per-frame focal/K,R,t from that homography is the NEXT cycle —
    # the near-affine telephoto focal solve is out of scope here, so we validate
    # via homography residual, not PnP.)
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, NFL_LANDMARKS, _yardline_x_m
    from nfl_gsplat.utils.meta import CalibHint
    # World rect spans the away_15..away_45 X range so every line projects in-frame.
    world = np.array([[-33.0, 3.0], [-4.0, 3.0], [-4.0, -3.0], [-33.0, -3.0]], np.float32)
    image = np.array([[220, 200], [1700, 230], [1620, 880], [260, 850]], np.float32)
    H = cv2.getPerspectiveTransform(world, image)

    def proj(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
        return (float(p[0]), float(p[1]))
    yards = [15, 20, 25, 30, 35, 40, 45]
    lines = [YardLineSeg(proj(_yardline_x_m(f"away_{y}"), +HASH_OFFSET_M),
                         proj(_yardline_x_m(f"away_{y}"), -HASH_OFFSET_M)) for y in yards]
    hashes = []                                  # dense 1-yard ticks (≥6/row)
    for X in np.linspace(-32.0, -4.5, 32):
        hashes += [proj(X, +HASH_OFFSET_M), proj(X, -HASH_OFFSET_M)]
    # spurious lines placed BETWEEN real yard lines (x@mid ≈ 289/512/.../1634)
    # so they test consensus rejection, not merge eviction of a nearby real line.
    for sx in (400.0, 850.0, 1300.0):            # no hash support → must be rejected
        lines.append(YardLineSeg((sx, 150.0), (sx + 2.0, 950.0)))
    feats = DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                             numbers=[], image_size=(1920, 1080))
    ref_x = 0.5 * (lines[3].p0[0] + lines[3].p1[0])
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    state0 = seed_state_from_hint(feats, hint)
    corrs, state = identify_correspondences(feats, state0)
    names = {n for n, _ in corrs}
    # all 7 real yard lines recovered with correct yardage; no spurious labels
    for y in yards:
        assert f"away_{y}_left_hash" in names and f"away_{y}_right_hash" in names
    assert len(corrs) == 14
    # the recovered homography reprojects every labeled world point onto its image pt
    assert state.homography is not None
    resid = []
    for name, (u, v) in corrs:
        X, Y = NFL_LANDMARKS[name][:2]
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), state.homography).reshape(2)
        resid.append(float(np.hypot(p[0] - u, p[1] - v)))
    assert max(resid) < 2.0


def test_identify_propagates_via_prior_homography():
    import cv2
    import numpy as np
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, _yardline_x_m
    from nfl_gsplat.utils.meta import CalibHint
    world = np.array([[-25.0, 3.0], [25.0, 3.0], [25.0, -3.0], [-25.0, -3.0]], np.float32)

    def make(image):
        H = cv2.getPerspectiveTransform(world, image.astype(np.float32))

        def pr(X, Y):
            p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
            return (float(p[0]), float(p[1]))
        lines = [YardLineSeg(pr(_yardline_x_m(f"away_{y}"), +HASH_OFFSET_M),
                             pr(_yardline_x_m(f"away_{y}"), -HASH_OFFSET_M))
                 for y in [20, 25, 30, 35, 40]]
        hashes = []                              # dense 1-yard ticks (≥6/row)
        for X in np.linspace(-28.0, -9.0, 24):
            hashes += [pr(X, +HASH_OFFSET_M), pr(X, -HASH_OFFSET_M)]
        return DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                                numbers=[], image_size=(1920, 1080))
    f0 = make(np.array([[260, 180], [1660, 210], [1520, 900], [380, 930]]))
    f1 = make(np.array([[280, 180], [1680, 210], [1540, 900], [400, 930]]))
    ref_x = 0.5 * (f0.yard_lines[2].p0[0] + f0.yard_lines[2].p1[0])
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    _, prior = identify_correspondences(f0, seed_state_from_hint(f0, hint))
    assert prior.homography is not None
    corrs, _ = identify_correspondences(f1, prior)
    names = {n for n, _ in corrs}
    assert "away_30_left_hash" in names


def test_identify_skips_hashes_with_single_row():
    # Only one hash band visible → can't disambiguate left/right → no hash corrs.
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.utils.meta import CalibHint
    xs = [400, 800, 1200]
    hashes = [(float(x), 360.0) for x in range(200, 1400, 20)]   # one row only
    feats = DetectedFeatures(
        yard_lines=[YardLineSeg((float(x), 0.0), (float(x), 1080.0)) for x in xs],
        sidelines=[], hashes=hashes, numbers=[], image_size=(1920, 1080))
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    corrs, _ = identify_correspondences(feats, state)
    assert not any(n.endswith("_hash") for n, _ in corrs)
