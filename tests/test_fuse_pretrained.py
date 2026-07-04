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
    # two kps voting DIFFERENT yards for the same line -> ambiguous -> {}
    u = _u_at(_yardline_x_m("away_30"), 400.0)
    kps = [("30", u + 5.0, 400.0, 0.8), ("40", u - 5.0, 400.0, 0.8)]
    assert identify_lines(LINES, kps, territory="away") == {}


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
