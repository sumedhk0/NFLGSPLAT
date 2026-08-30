"""Cameras from labelled helmets, and 3D from the two of them together.

The rig is synthetic so that the answer is known exactly. Two cameras roughly
where a sideline and an endzone camera sit, players on the helmet plane, and
the question is whether the pipeline gets the cameras and then the positions
back out.
"""
import numpy as np
import pytest

from nfl_gsplat.calibration.decompose_homography import (
    camera_centre,
    krt_to_homography,
    projection_matrix,
)
from nfl_gsplat.calibration.from_helmets import (
    cameras_for_view,
    plane_homography,
    pooled_focal,
    triangulate_matched,
)

W, H = 1280, 720


def look_at(centre, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """OpenCV-convention (R, t): rows are right/down/forward, ``t = -R C``."""
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return R, -R @ centre


def intrinsics(f):
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


SIDELINE_C = np.array([-3.0, 78.0, 33.0])
ENDZONE_C = np.array([98.0, 2.0, 22.0])
FOCAL = 1600.0


def project(K, R, t, xyz):
    P = projection_matrix(K, R, t)
    h = np.c_[xyz, np.ones(len(xyz))] @ P.T
    return h[:, :2] / h[:, 2:3]


def players(n=22, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([-25, -20], [25, 20], size=(n, 2))


def test_plane_homography_reproduces_the_true_one():
    K, (R, t) = intrinsics(FOCAL), look_at(SIDELINE_C)
    xy = players()
    uv = project(K, R, t, np.c_[xy, np.zeros(len(xy))])
    got, inliers = plane_homography(xy, uv)
    assert inliers.sum() == len(xy)
    truth = krt_to_homography(K, R, t)
    assert np.allclose(got / got[2, 2], truth, atol=1e-6)


def test_pooled_focal_recovers_the_focal_length():
    """Pooled over frames, because one frame's estimate is noisy by itself."""
    K = intrinsics(FOCAL)
    homographies = []
    for dx in np.linspace(-12, 12, 9):
        R, t = look_at(SIDELINE_C, target=(dx, 0.0, 0.0))
        homographies.append(krt_to_homography(K, R, t))
    f, spread = pooled_focal(homographies, W, H)
    assert abs(f - FOCAL) < 1.0
    assert spread < 1.0


def test_cameras_for_view_recovers_the_camera_position():
    K = intrinsics(FOCAL)
    xy = players()
    byf, truth = {}, {}
    for frame, dx in enumerate(np.linspace(-12, 12, 9), start=1):
        R, t = look_at(SIDELINE_C, target=(dx, 0.0, 0.0))
        byf[frame] = (project(K, R, t, np.c_[xy, np.zeros(len(xy))]),
                      np.arange(len(xy)))
        truth[frame] = (R, t)
    cams, focal = cameras_for_view(byf, lambda _f: xy, W, H)
    assert abs(focal - FOCAL) < 1.0
    for frame, (_K, R, t) in cams.items():
        assert np.allclose(camera_centre(R, t), SIDELINE_C, atol=0.05)
        assert np.allclose(R, truth[frame][0], atol=1e-3)


def test_triangulation_recovers_positions_from_two_views():
    K = intrinsics(FOCAL)
    xy = players()
    # Give the helmets real height spread; recovering it is the point.
    rng = np.random.default_rng(1)
    z = rng.uniform(-0.3, 0.3, size=len(xy))
    xyz = np.c_[xy, z]

    Rs, ts = look_at(SIDELINE_C)
    Re, te = look_at(ENDZONE_C)
    uv_s = project(K, Rs, ts, xyz)
    uv_e = project(K, Re, te, xyz)
    cols = np.arange(len(xy))

    got_cols, got_xyz = triangulate_matched(
        projection_matrix(K, Rs, ts), uv_s, cols,
        projection_matrix(K, Re, te), uv_e, cols)
    assert np.array_equal(got_cols, cols)
    assert np.allclose(got_xyz, xyz, atol=1e-6)


def test_triangulation_uses_only_players_seen_in_both_views():
    K = intrinsics(FOCAL)
    xy = players()
    xyz = np.c_[xy, np.zeros(len(xy))]
    Rs, ts = look_at(SIDELINE_C)
    Re, te = look_at(ENDZONE_C)
    keep_s = np.array([0, 1, 2, 3, 4, 5, 6])
    keep_e = np.array([4, 5, 6, 7, 8, 9])
    got_cols, got_xyz = triangulate_matched(
        projection_matrix(K, Rs, ts), project(K, Rs, ts, xyz)[keep_s], keep_s,
        projection_matrix(K, Re, te), project(K, Re, te, xyz)[keep_e], keep_e)
    assert np.array_equal(got_cols, np.array([4, 5, 6]))
    assert np.allclose(got_xyz, xyz[[4, 5, 6]], atol=1e-6)


def test_pooled_focal_refuses_when_every_view_is_degenerate():
    """A camera that never tilts cannot reveal its focal; say so, don't guess."""
    with pytest.raises(Exception):
        pooled_focal([np.eye(3), np.eye(3), np.eye(3)], W, H)
