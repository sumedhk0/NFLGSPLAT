"""Paint-only calibration: does it recover a camera from lines alone?"""
from dataclasses import dataclass

import numpy as np
import pytest

from nfl_gsplat.calibration.field_landmarks import (
    HALF_WIDTH_M,
    HASH_OFFSET_M,
    YARD_LINE_SPACING_M,
)
from nfl_gsplat.calibration.from_paint import (
    CROSS_Y_M,
    cameras_from_paint,
    fit_rows,
    frame_homography,
    homography_from_lines,
    line_at_x,
    ray_to_plane,
    seg_line,
)

W, H = 1280, 720


@dataclass(frozen=True)
class Seg:
    p0: tuple
    p1: tuple


@dataclass
class Feats:
    yard_lines: list
    sidelines: list
    hashes: list
    numbers: tuple = ()
    image_size: tuple = (W, H)


def look_at(centre, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return R, -R @ centre


FOCAL = 2600.0
CENTRE = np.array([0.0, -70.0, 30.0])


def ground_homography(K, R, t):
    """Ground plane (z=0) to image."""
    M = np.column_stack([R[:, 0], R[:, 1], t])
    return K @ M


def K_of(f):
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def project(Hm, xy):
    q = np.c_[np.asarray(xy, float), np.ones(len(xy))] @ Hm.T
    return q[:, :2] / q[:, 2:3]


def synth_features(x_origin=0.0, n_yard=5, K=None, R=None, t=None):
    """A frame's worth of paint, exactly as a camera would see it."""
    K = K_of(FOCAL) if K is None else K
    if R is None:
        R, t = look_at(CENTRE)
    Hm = ground_homography(K, R, t)

    yard = []
    for k in range(n_yard):
        x = x_origin + k * YARD_LINE_SPACING_M
        pts = project(Hm, [[x, -HALF_WIDTH_M], [x, HALF_WIDTH_M]])
        yard.append(Seg(tuple(pts[0]), tuple(pts[1])))

    hashes = []
    for y in (+HASH_OFFSET_M, -HASH_OFFSET_M):
        xs = x_origin + np.linspace(-4.0, 4.0 + n_yard * YARD_LINE_SPACING_M, 30)
        for p in project(Hm, np.c_[xs, np.full_like(xs, y)]):
            hashes.append((float(p[0]), float(p[1])))

    xs = x_origin + np.linspace(-10.0, 10.0 + n_yard * YARD_LINE_SPACING_M, 2)
    side = project(Hm, np.c_[xs, np.full_like(xs, HALF_WIDTH_M)])
    sidelines = [Seg(tuple(side[0]), tuple(side[1]))]
    return Feats(yard, sidelines, hashes), Hm


def test_homography_from_lines_recovers_the_true_homography():
    _feats, Hm = synth_features()
    world, image = [], []
    for x in (0.0, YARD_LINE_SPACING_M, 2 * YARD_LINE_SPACING_M):
        world.append([1.0, 0.0, -x])
        pts = project(Hm, [[x, -10.0], [x, 10.0]])
        image.append(seg_line(Seg(tuple(pts[0]), tuple(pts[1]))))
    for y in (HALF_WIDTH_M, HASH_OFFSET_M, -HASH_OFFSET_M):
        world.append([0.0, 1.0, -y])
        pts = project(Hm, [[-5.0, y], [15.0, y]])
        image.append(seg_line(Seg(tuple(pts[0]), tuple(pts[1]))))
    got = homography_from_lines(world, image)
    assert got is not None
    a = got / got[2, 2]
    b = Hm / Hm[2, 2]
    assert np.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_fit_rows_finds_the_hash_rows_among_noise():
    feats, _Hm = synth_features()
    rng = np.random.default_rng(0)
    noisy = list(feats.hashes) + [
        (float(x), float(y)) for x, y in rng.uniform([0, 0], [W, 60], size=(40, 2))]
    rows = fit_rows(noisy)
    assert len(rows) >= 2
    # the two strongest rows should be the planted ones, ~30 marks each
    assert sorted(n for _l, n in rows)[-2] >= 20


def test_frame_homography_labels_the_long_lines_correctly():
    feats, _Hm = synth_features()
    got = frame_homography(feats, W, H, gap=1)
    assert got is not None
    _H, world_y, residual = got
    assert residual < 2.0
    assert set(np.round(world_y, 3)) <= set(np.round(CROSS_Y_M, 3))
    # sideline plus both hash rows were planted
    assert len(world_y) == 3


def test_cameras_from_paint_recovers_focal_and_position_up_to_x_shift():
    """X ORIGIN is unrecoverable from paint, so only the shift is allowed to move."""
    feats, _Hm = synth_features(x_origin=-3 * YARD_LINE_SPACING_M)
    by_frame = {i: feats for i in range(1, 9)}
    cams, focal, residual = cameras_from_paint(by_frame, W, H, gap=1)
    assert abs(focal - FOCAL) < 0.05 * FOCAL
    assert residual < 2.0
    K, R, t = next(iter(cams.values()))
    got_centre = -R.T @ t
    # y and z must be right; x carries the unknowable whole-yard shift
    assert abs(got_centre[1] - CENTRE[1]) < 3.0
    assert abs(got_centre[2] - CENTRE[2]) < 3.0


def test_ray_to_plane_inverts_the_projection():
    K = K_of(FOCAL)
    R, t = look_at(CENTRE)
    Hm = ground_homography(K, R, t)
    truth = np.array([[4.0, 6.0], [-8.0, -3.0], [12.0, 1.5]])
    for xy, uv in zip(truth, project(Hm, truth)):
        got = ray_to_plane(K, R, t, uv, 0.0)
        assert got is not None
        assert np.allclose(got, xy, atol=1e-6)


def test_ray_to_plane_refuses_rays_going_the_wrong_way():
    K = K_of(FOCAL)
    R, t = look_at(CENTRE)
    # far above the horizon: the ray never reaches the ground in front
    assert ray_to_plane(K, R, t, (W / 2.0, -5000.0), 0.0) is None


def test_line_at_x_handles_a_vertical_line():
    assert line_at_x(np.array([1.0, 0.0, -5.0]), 0.0) is None


def test_cameras_from_paint_says_so_when_the_field_is_not_visible():
    from nfl_gsplat.errors import CalibrationError

    empty = Feats([], [], [])
    with pytest.raises(CalibrationError):
        cameras_from_paint({i: empty for i in range(6)}, W, H)
