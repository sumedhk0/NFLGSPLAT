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


def test_identify_emits_two_hash_correspondences_per_line():
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.utils.meta import CalibHint
    xs = [400, 800, 1200]
    hashes = []
    for x in range(200, 1400, 20):
        hashes += [(float(x), 360.0), (float(x), 620.0)]
    feats = DetectedFeatures(
        yard_lines=[YardLineSeg((float(x), 0.0), (float(x), 1080.0)) for x in xs],
        sidelines=[], hashes=hashes, numbers=[], image_size=(1920, 1080))
    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    corrs, _ = identify_correspondences(feats, state)
    names = [c[0] for c in corrs]
    pt_by_name = dict(corrs)
    assert "away_30_left_hash" in names and "away_30_right_hash" in names
    lu = pt_by_name["away_30_left_hash"]; ld = pt_by_name["away_30_right_hash"]
    assert abs(lu[0] - 800) < 2 and abs(ld[0] - 800) < 2
    assert {round(lu[1]), round(ld[1])} == {360, 620}
    assert len([n for n in names if n.endswith("_hash")]) == 6


def test_identify_propagates_identity_across_shifted_frame():
    # Seed on frame 0, then run identify on a panned frame (lines shifted +10px);
    # identity must carry over via the nearest-x(@mid) <60px match (the 30 line
    # stays the 30 even though every line moved).
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, seed_state_from_hint,
    )
    from nfl_gsplat.utils.meta import CalibHint

    def feats(shift):
        xs = [400 + shift, 800 + shift, 1200 + shift]
        hashes = []
        for x in range(200, 1400, 20):
            hashes += [(float(x + shift), 360.0), (float(x + shift), 620.0)]
        return DetectedFeatures(
            yard_lines=[YardLineSeg((float(x), 0.0), (float(x), 1080.0)) for x in xs],
            sidelines=[], hashes=hashes, numbers=[], image_size=(1920, 1080))

    hint = CalibHint(ref_frame=0, ref_x=800, yard=30, side="away", increasing="right")
    state0 = seed_state_from_hint(feats(0), hint)
    _, prior = identify_correspondences(feats(0), state0)
    corrs, _ = identify_correspondences(feats(10), prior)        # panned +10px
    names = {c[0] for c in corrs}
    assert "away_30_left_hash" in names and "away_30_right_hash" in names


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


def test_identify_pnp_roundtrip_under_2px():
    import numpy as np
    from nfl_gsplat.calibration.field_features import DetectedFeatures, YardLineSeg
    from nfl_gsplat.calibration.field_identify import (
        identify_correspondences, line_x_at, seed_state_from_hint,
    )
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    from nfl_gsplat.calibration.solve_pnp import solve_pnp_from_correspondences
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points
    from nfl_gsplat.utils.meta import CalibHint

    # Top-down camera: looks in -Z (down onto field), camera X = world X, camera Y = -world Y.
    # Yard lines (constant world X) appear as exactly vertical image lines.
    # Hash rows (constant world Y) appear as exactly horizontal image rows.
    intr = CameraIntrinsics(1100.0, 1100.0, 960, 540, 1920, 1080)
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    pose = CameraPose(R=R, t=np.array([0.0, 0.0, 100.0]))

    def proj(name):
        return project_points(NFL_LANDMARKS[name][None], intr.K(), pose.R, pose.t)[0]

    # 7 yards gives 7 hash points per row, exceeding fit_hash_rows min_inliers=6.
    yards = [15, 20, 25, 30, 35, 40, 45]
    lines, hashes = [], []
    for y in yards:
        lh = proj(f"away_{y}_left_hash"); rh = proj(f"away_{y}_right_hash")
        lines.append(YardLineSeg((float(lh[0]), float(lh[1])), (float(rh[0]), float(rh[1]))))
        hashes += [(float(lh[0]), float(lh[1])), (float(rh[0]), float(rh[1]))]
    feats = DetectedFeatures(yard_lines=lines, sidelines=[], hashes=hashes,
                             numbers=[], image_size=(1920, 1080))
    # lines[3] is away_30 (centre of the 7-yard span)
    ref_x = line_x_at(lines[3], 540.0)
    hint = CalibHint(ref_frame=0, ref_x=ref_x, yard=30, side="away", increasing="right")
    state = seed_state_from_hint(feats, hint)
    corrs, _ = identify_correspondences(feats, state)
    assert len(corrs) >= 6
    res = solve_pnp_from_correspondences(corrs, image_size=(1920, 1080), max_reproj_px=1e9)
    assert res.rms_px < 2.0
