"""Fusion tests on a synthetic slanted view.

Geometry: yard lines rendered as slanted segments x = X_world*40 + 800 + 0.15*y
(40 px per meter, leaning right). Hash rows at v=300 (upper/left/+Y) and v=700
(lower/right/-Y). Model keypoints derived from the same geometry + noise.
"""
import numpy as np
import pytest

from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.calibration.field_landmarks import _yardline_x_m
from nfl_gsplat.calibration.fuse_pretrained import fuse_frame, identify_lines

W, H = 1920, 1080


def _seg_for(x_world):
    def u(y):
        return x_world * 40.0 + 800.0 + 0.15 * y
    return YardLineSeg((u(0.0), 0.0), (u(float(H)), float(H)))


def _u_at(x_world, y):
    return x_world * 40.0 + 800.0 + 0.15 * y


LINES = [_seg_for(_yardline_x_m(n)) for n in ("away_40", "away_35", "away_30")]


def _hashes():
    # dense ticks along both rows so fit_hash_rows locks on
    return ([(float(x), 300.0) for x in range(100, 1900, 40)]
            + [(float(x), 700.0) for x in range(100, 1900, 40)])


def test_identify_lines_votes_nearest_with_noise():
    # model kp for away_30's top number, 25px off the line -> still votes right line
    u30 = _u_at(_yardline_x_m("away_30"), 200.0) + 25.0
    ident = identify_lines(LINES, [("30", u30, 200.0, 0.8)], territory="away")
    assert ident == {2: "away_30"}


def test_identify_lines_fills_unvoted_neighbors_consistently():
    # votes on away_40 and away_30; the middle line must become away_35 —
    # via world-X interpolation + snap + yardline_name_from_x_m round-trip.
    kps = [("40", _u_at(_yardline_x_m("away_40"), 100.0) + 10.0, 100.0, 0.8),
           ("30", _u_at(_yardline_x_m("away_30"), 500.0) - 15.0, 500.0, 0.8)]
    ident = identify_lines(LINES, kps, territory="away")
    assert ident == {0: "away_40", 1: "away_35", 2: "away_30"}


def test_identify_lines_drops_frame_on_conflict():
    # two kps voting DIFFERENT yards for the SAME line, 1-1 tie -> that line
    # is dropped (spec: majority per line, tie drops the line, not the
    # frame). With only one line ever voted here, dropping it empties ident.
    u = _u_at(_yardline_x_m("away_30"), 400.0)
    kps = [("30", u + 5.0, 400.0, 0.8), ("40", u - 5.0, 400.0, 0.8)]
    assert identify_lines(LINES, kps, territory="away") == {}


def test_identify_lines_majority_beats_single_bad_vote():
    # 3 correct votes for away_30 (number + both hash variants) outvote one
    # hallucinated "10" keypoint landing on the same line -> majority wins,
    # frame is NOT dropped.
    v = 400.0
    u30 = _u_at(_yardline_x_m("away_30"), v)
    kps = [
        ("30", u30 + 5.0, v, 0.8),
        ("30-top-hash", u30 - 3.0, v, 0.8),
        ("30-bottom-hash", u30 + 2.0, v, 0.8),
        ("10", u30 + 4.0, v, 0.95),   # spurious hallucinated vote, same line
    ]
    ident = identify_lines(LINES, kps, territory="away")
    assert ident == {2: "away_30"}


def test_identify_lines_tie_drops_line_not_frame():
    # 1-1 tie on the away_30 line drops that line only; a clean, unambiguous
    # vote on the away_40 line survives.
    v30 = 400.0
    u30 = _u_at(_yardline_x_m("away_30"), v30)
    v40 = 100.0
    u40 = _u_at(_yardline_x_m("away_40"), v40)
    kps = [
        ("30", u30 + 3.0, v30, 0.8),
        ("10", u30 - 3.0, v30, 0.8),   # tie: 1 vs 1 on the away_30 line
        ("40", u40 + 2.0, v40, 0.8),
    ]
    ident = identify_lines(LINES, kps, territory="away")
    assert ident == {0: "away_40"}


def test_identify_lines_no_clamped_extrapolation():
    # votes only on away_35 (middle) and away_30 (one end); away_40 lies
    # outside the voted image-x range and must NOT be filled by clamped
    # np.interp extrapolation.
    v35 = 300.0
    u35 = _u_at(_yardline_x_m("away_35"), v35)
    v30 = 500.0
    u30 = _u_at(_yardline_x_m("away_30"), v30)
    kps = [
        ("35", u35 + 2.0, v35, 0.8),
        ("30", u30 - 2.0, v30, 0.8),
    ]
    ident = identify_lines(LINES, kps, territory="away")
    assert 0 not in ident              # away_40 (index 0) left unidentified
    assert ident.get(1) == "away_35"
    assert ident.get(2) == "away_30"
    vals = list(ident.values())
    assert len(vals) == len(set(vals))


def test_identify_lines_duplicate_base_dropped():
    # A ghost classical line ~50px from the real away_25 line: both are
    # unvoted and both interpolate/snap to "away_25" -> ambiguous -> both
    # lines carrying that base are dropped, voted lines survive untouched.
    x30 = _yardline_x_m("away_30")
    x25 = _yardline_x_m("away_25")
    x20 = _yardline_x_m("away_20")
    ghost_lines = [
        _seg_for(x30),                 # index 0: voted away_30
        _seg_for(x25 + 50.0 / 40.0),   # index 1: ghost, ~50px from away_25
        _seg_for(x25),                 # index 2: real away_25 (unvoted)
        _seg_for(x20),                 # index 3: voted away_20
    ]
    kps = [
        ("30", _u_at(x30, 400.0) + 3.0, 400.0, 0.8),
        ("20", _u_at(x20, 100.0) - 3.0, 100.0, 0.8),
    ]
    ident = identify_lines(ghost_lines, kps, territory="away")
    vals = list(ident.values())
    assert len(vals) == len(set(vals))
    assert ident.get(0) == "away_30"
    assert ident.get(3) == "away_20"


def test_identify_lines_rejects_far_assignment():
    # kp ~220px from the only line (line x at v=50 is ~76) -> no vote -> {}
    kps = [("30", 300.0, 50.0, 0.8)]
    ident = identify_lines([_seg_for(_yardline_x_m("away_30"))], kps,
                           territory="away")
    assert ident == {}


def test_fuse_frame_emits_intersections_and_numbers():
    # two voted lines (40 and 30) so the unvoted away_35 gets neighbor-filled
    u30 = _u_at(_yardline_x_m("away_30"), 200.0) + 20.0
    u40 = _u_at(_yardline_x_m("away_40"), 100.0) - 10.0
    corrs = fuse_frame(LINES, _hashes(),
                       [("30", u30, 200.0, 0.8), ("40", u40, 100.0, 0.8)],
                       territory="away", image_size=(W, H))
    names = {n for (n, _uv) in corrs}
    # every identified line x both hash rows, named by the left=+Y=upper convention
    for base in ("away_40", "away_35", "away_30"):
        assert f"{base}_left_hash" in names       # upper row v=300
        assert f"{base}_right_hash" in names      # lower row v=700
    assert "away_30_left_number" in names          # the model number kp rides along
    # intersection precision: away_30 x upper row lands on the true line at v=300
    uv = dict(corrs)["away_30_left_hash"]
    assert abs(uv[0] - _u_at(_yardline_x_m("away_30"), 300.0)) < 1.0
    assert abs(uv[1] - 300.0) < 1.0


def test_fuse_frame_no_model_kps_returns_empty():
    assert fuse_frame(LINES, _hashes(), [], territory="away",
                      image_size=(W, H)) == []
