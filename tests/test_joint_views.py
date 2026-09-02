"""Two views must agree about where the players are; one view cannot check that."""
import numpy as np
import pytest

from nfl_gsplat.calibration.joint_views import choose_pair, match_count
from nfl_gsplat.errors import CalibrationError

W, H = 1920, 1080


def look_at(centre, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return R, -R @ centre


def K_of(f):
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def cam(centre, focal):
    R, t = look_at(np.asarray(centre, float))
    return K_of(focal), R, t


def project(c, pts):
    K, R, t = c
    q = (np.asarray(pts, float) @ R.T + t) @ K.T
    return q[:, :2] / q[:, 2:3]


SIDE = cam([0.0, -90.0, 40.0], 3000.0)
ENDZ = cam([95.0, 0.0, 25.0], 2200.0)


def feet(n=16, seed=0):
    """Where players stand -- on the turf, which is what makes them comparable."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform([-22, -16], [22, 16], size=(n, 2))
    return np.c_[xy, np.zeros(n)]


def test_the_right_pair_reconciles_the_players():
    pts = feet()
    n, placed = match_count(SIDE, project(SIDE, pts), ENDZ, project(ENDZ, pts))
    assert n >= 14
    # And it puts them where they really stood, not merely in agreement.
    gap = np.linalg.norm(placed[:, None] - pts[None, :, :2], axis=2).min(1)
    assert gap.max() < 0.5


def test_a_stretched_second_camera_reconciles_almost_nothing():
    """The measured failure: two cameras each fine alone, disagreeing on width.

    On a real play the sideline put the players across 15 m of field and the
    endzone across 45 m; only three of twenty-two matched, and neither view
    could see anything wrong with itself.
    """
    pts = feet()
    seen_a = project(SIDE, pts)
    # The endzone view of a world whose Y is stretched threefold.
    stretched = pts.copy()
    stretched[:, 1] *= 3.0
    seen_b = project(ENDZ, stretched)
    n, _placed = match_count(SIDE, seen_a, ENDZ, seen_b)
    assert n < 6


def test_the_pair_search_prefers_the_consistent_camera():
    pts = feet()
    dets_a = {0: project(SIDE, pts), 1: project(SIDE, pts)}
    dets_b = {0: project(ENDZ, pts), 1: project(ENDZ, pts)}
    # A labelling that took the hash rows for the sidelines: the same camera
    # believing the field is three times as wide. This is the measured failure,
    # and unlike a distance error it survives every single-view check.
    wrong = cam([95.0 / 1.0, 0.0, 25.0], 2200.0 / 3.0)
    cands_a = [{0: SIDE, 1: SIDE}]
    cands_b = [{0: wrong, 1: wrong}, {0: ENDZ, 1: ENDZ}]
    _a, _b, matched, info = choose_pair(cands_a, cands_b, dets_a, dets_b)
    assert info["b"] == 1                         # the consistent one
    assert matched >= 14


def test_it_refuses_when_the_views_describe_different_worlds():
    pts = feet()
    stretched = pts.copy()
    stretched[:, 1] *= 3.0
    dets_a = {0: project(SIDE, pts)}
    dets_b = {0: project(ENDZ, stretched)}
    with pytest.raises(CalibrationError, match="different worlds"):
        choose_pair([{0: SIDE}], [{0: ENDZ}], dets_a, dets_b)
