"""Fusion tests on a synthetic slanted view.

Geometry: yard lines rendered as slanted segments x = X_world*40 + 800 + 0.15*y
(40 px per meter, leaning right). Hash rows at v=300 (upper/left/+Y) and v=700
(lower/right/-Y). Model keypoints derived from the same geometry + noise.
"""
import numpy as np
import pytest

from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.calibration.field_landmarks import _yardline_x_m
from nfl_gsplat.calibration.fuse_pretrained import (
    fuse_frame, identify_lines, predict_identities,
)

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


def test_identify_lines_does_not_fill_unvoted_neighbors():
    # votes on away_40 and away_30 only; identify_lines no longer fills the
    # unvoted middle line (away_35) -- that completion now happens in
    # fuse_frame via the homography, not here.
    kps = [("40", _u_at(_yardline_x_m("away_40"), 100.0) + 10.0, 100.0, 0.8),
           ("30", _u_at(_yardline_x_m("away_30"), 500.0) - 15.0, 500.0, 0.8)]
    ident = identify_lines(LINES, kps, territory="away")
    assert ident == {0: "away_40", 2: "away_30"}


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


def test_identify_lines_no_duplicate_among_voted_lines():
    # votes only on away_35 (middle) and away_30 (one end); away_40 (index 0)
    # is unvoted and identify_lines no longer fills it -- it simply stays
    # out of the result. What this test guards -- no duplicate base names
    # among the lines that DO get identified -- still holds.
    v35 = 300.0
    u35 = _u_at(_yardline_x_m("away_35"), v35)
    v30 = 500.0
    u30 = _u_at(_yardline_x_m("away_30"), v30)
    kps = [
        ("35", u35 + 2.0, v35, 0.8),
        ("30", u30 - 2.0, v30, 0.8),
    ]
    ident = identify_lines(LINES, kps, territory="away")
    assert ident == {1: "away_35", 2: "away_30"}
    vals = list(ident.values())
    assert len(vals) == len(set(vals))


def test_identify_lines_leaves_unvoted_lines_unidentified():
    # A ghost classical line ~50px from the real away_25 line: both are
    # unvoted, and identify_lines no longer fills/snaps unvoted lines to a
    # nearby name -- they simply stay out of the result. Only the voted
    # lines (away_30, away_20) get an identity, so no duplicate/ambiguous
    # base name can arise here.
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
    assert ident == {0: "away_30", 3: "away_20"}


def test_identify_lines_rejects_far_assignment():
    # kp ~220px from the only line (line x at v=50 is ~76) -> no vote -> {}
    kps = [("30", 300.0, 50.0, 0.8)]
    ident = identify_lines([_seg_for(_yardline_x_m("away_30"))], kps,
                           territory="away")
    assert ident == {}


def test_fuse_frame_emits_intersections_and_numbers():
    # two voted lines (40 and 30) so the unvoted away_35 gets completed via
    # the homography implied by the voted lines' hash intersections
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
    # the model number kp rides along at its raw (coarse) pixel position,
    # which is not consistent with the single-plane homography fit from the
    # hash intersections -> the RANSAC gate at the end of fuse_frame drops it.
    assert "away_30_left_number" not in names
    # intersection precision: away_30 x upper row lands on the true line at v=300
    uv = dict(corrs)["away_30_left_hash"]
    assert abs(uv[0] - _u_at(_yardline_x_m("away_30"), 300.0)) < 1.0
    assert abs(uv[1] - 300.0) < 1.0


def test_fuse_frame_drops_coarse_outlier_numbers():
    # Same setup as above, but the model number keypoints are coarsened to
    # ~30px off the true line position -- representative of real-footage
    # model error. All 6 hash intersections must survive the RANSAC gate;
    # the coarse, plane-inconsistent numbers must not.
    u30 = _u_at(_yardline_x_m("away_30"), 200.0) + 30.0
    u40 = _u_at(_yardline_x_m("away_40"), 100.0) - 30.0
    corrs = fuse_frame(LINES, _hashes(),
                       [("30", u30, 200.0, 0.8), ("40", u40, 100.0, 0.8)],
                       territory="away", image_size=(W, H))
    names = {n for (n, _uv) in corrs}
    for base in ("away_40", "away_35", "away_30"):
        assert f"{base}_left_hash" in names
        assert f"{base}_right_hash" in names
    assert "away_30_left_number" not in names
    assert "away_40_left_number" not in names


def test_fuse_frame_keeps_consistent_points():
    # Small, realistic voting-only offsets (well within max_assign_px) so
    # line identification succeeds cleanly. The 6 hash intersections are
    # exactly plane-consistent by construction (same synthetic homography)
    # and must always survive the RANSAC gate.
    u30 = _u_at(_yardline_x_m("away_30"), 200.0) + 5.0
    u40 = _u_at(_yardline_x_m("away_40"), 100.0) - 5.0
    corrs = fuse_frame(LINES, _hashes(),
                       [("30", u30, 200.0, 0.8), ("40", u40, 100.0, 0.8)],
                       territory="away", image_size=(W, H))
    names = {n for (n, _uv) in corrs}
    for base in ("away_40", "away_35", "away_30"):
        assert f"{base}_left_hash" in names
        assert f"{base}_right_hash" in names
    assert len(corrs) >= 6


def test_fuse_frame_completes_identities_via_homography():
    # votes only on away_35 and away_30; the unvoted away_40 must be
    # identified by mapping its hash-row intersections through the
    # homography implied by the two voted lines -- not by linear world-X
    # interpolation (real footage: that misassigned the 40 as away_35).
    # u = X*40 + 800 + 0.15*y IS a valid (affine, hence also projective)
    # homography, so completion is exact on this synthetic geometry.
    kps = [("35-top-hash", _u_at(_yardline_x_m("away_35"), 300.0) + 5.0, 300.0, 0.9),
           ("30", _u_at(_yardline_x_m("away_30"), 200.0) - 5.0, 200.0, 0.8)]
    corrs = fuse_frame(LINES, _hashes(), kps, territory="away", image_size=(W, H))
    names = {n for (n, _uv) in corrs}
    assert "away_40_left_hash" in names and "away_40_right_hash" in names


def test_fuse_frame_completion_rejects_offgrid_line():
    # a spurious classical line 2m off the away_35 line (i.e. not on any
    # painted yard line) is unvoted; when completion maps its hash-row
    # intersections through the homography, the resulting world-X is >1.6m
    # (the snap tolerance) from every painted line, so it is correctly left
    # unidentified and contributes no correspondences.
    ghost = _seg_for(_yardline_x_m("away_35") + 2.0)
    lines = LINES + [ghost]
    kps = [("35-top-hash", _u_at(_yardline_x_m("away_35"), 300.0), 300.0, 0.9),
           ("30", _u_at(_yardline_x_m("away_30"), 200.0), 200.0, 0.8)]
    corrs = fuse_frame(lines, _hashes(), kps, territory="away", image_size=(W, H))
    names = [n for (n, _uv) in corrs]
    # no duplicate names -- the ghost contributes nothing
    assert len(names) == len(set(names))
    # exactly the three real lines' hash correspondences are present
    for base in ("away_40", "away_35", "away_30"):
        assert f"{base}_left_hash" in names
        assert f"{base}_right_hash" in names
    assert len(names) == 6


def test_fuse_frame_no_model_kps_returns_empty():
    assert fuse_frame(LINES, _hashes(), [], territory="away",
                      image_size=(W, H)) == []


# --- predict_identities: identification from a KNOWN neighbor plane homography ----
#
# Geometry: a real perspective camera (not the affine synthetic rig above), reusing
# the exact K/R/t pattern from test_learned_register_sequence_with_stub_detector in
# test_run_autocalib.py. yard lines and both hash rows are rendered by projecting
# world points through H = K @ [R[:,0], R[:,1], t] -- the same plane homography a
# solved neighboring frame would hand to predict_identities.

def _plane_H_for_rig():
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M  # noqa: F401 (re-export check)
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
    intr = CameraIntrinsics(1400.0, 1400.0, 960, 540, 1920, 1080)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    t = np.array([0.0, 6.0, 55.0])
    pose = CameraPose(R=R, t=t)
    K = intr.K()
    Rt = np.column_stack([pose.R[:, 0], pose.R[:, 1], pose.t])
    return K @ Rt


def _proj(H, X, Y):
    w = H @ np.array([X, Y, 1.0])
    return (float(w[0] / w[2]), float(w[1] / w[2]))


def _rig_line(H, X0):
    return YardLineSeg(_proj(H, X0, -25.0), _proj(H, X0, 25.0))


def _rig_row(H, Y0):
    return YardLineSeg(_proj(H, -55.0, Y0), _proj(H, 55.0, Y0))


def test_predict_identities_from_plane_homography():
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M

    H = _plane_H_for_rig()
    names = ["away_30", "away_35", "away_40"]
    yard_lines = [_rig_line(H, _yardline_x_m(n)) for n in names]
    # upper row = +Y = *_left_hash convention (validated in fuse_pretrained.py)
    rows = [_rig_row(H, HASH_OFFSET_M), _rig_row(H, -HASH_OFFSET_M)]

    ident = predict_identities(yard_lines, rows, H)
    assert set(ident.values()) == set(names)
    assert len(ident) == 3


def test_predict_identities_rejects_offgrid():
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M

    H = _plane_H_for_rig()
    names = ["away_30", "away_35", "away_40"]
    yard_lines = [_rig_line(H, _yardline_x_m(n)) for n in names]
    # ghost line 2m off the away_35 line -- well outside predict_identities'
    # default 1m snap tolerance -> must get no identity, and must not corrupt
    # the real lines' identities (no duplicates).
    ghost = _rig_line(H, _yardline_x_m("away_35") + 2.0)
    yard_lines_with_ghost = yard_lines + [ghost]
    rows = [_rig_row(H, HASH_OFFSET_M), _rig_row(H, -HASH_OFFSET_M)]

    ident = predict_identities(yard_lines_with_ghost, rows, H)
    assert 3 not in ident                      # ghost (index 3) gets no identity
    assert set(ident.values()) == set(names)   # the three real lines still resolve
    vals = list(ident.values())
    assert len(vals) == len(set(vals))         # no duplicate base names
