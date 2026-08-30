"""Cameras from labelled helmets, and 3D from the two of them together.

The rig is synthetic so that the answer is known exactly. Two cameras roughly
where a sideline and an endzone camera sit, players on the helmet plane, and
the question is whether the pipeline gets the cameras and then the positions
back out.
"""
import numpy as np
import pytest

from nfl_gsplat.errors import CalibrationError

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
    with pytest.raises(CalibrationError):
        pooled_focal([np.eye(3), np.eye(3), np.eye(3)], W, H)


def test_fixed_centre_beats_per_frame_under_realistic_noise():
    """Why the fixed-centre solve exists, with the real mechanism.

    It is not that a telephoto view has no focal in it -- on noiseless points
    the per-frame estimate is exact. It is that the focal is weakly
    CONDITIONED at this geometry, so the ~8 px that real helmet-height
    variation puts on every correspondence is amplified enormously. On real
    footage that took the per-frame focal from 2125 to 18430 px across one
    play while every homography still fitted to 6.3 px.

    One shared camera centre over the play constrains what one frame cannot.
    """
    from nfl_gsplat.calibration.from_helmets import cameras_fixed_centre

    centre = np.array([0.0, -80.0, 35.0])
    focal = 3000.0
    K = intrinsics(focal)
    rng = np.random.default_rng(3)
    base = players(seed=3)
    byf, world = {}, {}
    for frame, dx in enumerate(np.linspace(-8, 8, 14), start=1):
        xy = base + rng.normal(0.0, 0.4, size=base.shape)
        R, t = look_at(centre, target=(dx, 0.0, 0.0))
        uv = (project(K, R, t, np.c_[xy, np.zeros(len(xy))])
              + rng.normal(0.0, 8.0, size=(len(xy), 2)))
        byf[frame] = (uv, np.arange(len(xy)))
        world[frame] = xy

    homographies = []
    for frame, (uv, _cols) in byf.items():
        h, _inliers = plane_homography(world[frame], uv)
        # The FIT is still fine -- that is the whole point. It is the focal
        # extracted from the fit that is not, not the fit itself.
        import cv2
        proj = cv2.perspectiveTransform(
            world[frame].reshape(-1, 1, 2).astype(float), h).reshape(-1, 2)
        assert np.median(np.linalg.norm(proj - uv, axis=1)) < 15.0
        homographies.append(h)
    _f_pf, spread = pooled_focal(homographies, W, H)

    cams, got_centre, mirrored = cameras_fixed_centre(
        byf, lambda f: world[f], W, H, audit_px=25.0)
    got_focal = np.median([Ki[0, 0] for Ki, _R, _t in cams.values()])

    assert not mirrored
    assert len(cams) >= 10
    assert spread > 0.05 * focal                   # per-frame: scattered
    assert abs(got_focal - focal) < 0.05 * focal   # jointly: pinned down
    assert np.linalg.norm(got_centre - centre) < 2.0
