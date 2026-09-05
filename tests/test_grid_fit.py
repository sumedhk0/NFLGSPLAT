"""The calibrated grid against the painted lines, in pixels (calibration.grid_fit)."""
import numpy as np

from nfl_gsplat.calibration import grid_fit as gf
from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

W, H = 1920, 1080


def _camera(roll_deg=0.0):
    K = intrinsics(W, H, fov_deg=14.0)
    R, t = look_at(np.array([0.0, -100.0, 45.0]), np.array([0.0, 0.0, 0.0]))
    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return K, R, t


def _painted_frame(K, R, t):
    """White 5-yard lines drawn on dark turf through the TRUE camera."""
    import cv2

    img = np.full((H, W, 3), (40, 90, 40), np.uint8)
    P = K @ np.column_stack([R[:, :2], t])
    for x in np.arange(-45.72, 45.73, 4.572):
        a = P @ np.array([x, -24.0, 1.0])
        b = P @ np.array([x, 24.0, 1.0])
        if a[2] > 0 and b[2] > 0:
            cv2.line(img, tuple((a[:2] / a[2]).astype(int)), tuple((b[:2] / b[2]).astype(int)),
                     (240, 240, 240), 4)
    return img


def test_right_camera_reads_a_few_pixels_and_a_rolled_one_reads_many():
    K, R, t = _camera()
    frame = _painted_frame(K, R, t)
    d_ok, n_ok = gf.grid_distance_px(frame, K, R, t)
    assert n_ok >= 5
    assert d_ok < 2.5, d_ok
    K2, R2, t2 = _camera(roll_deg=3.0)
    d_bad, _n = gf.grid_distance_px(frame, K2, R2, t2)
    assert d_bad > 4 * max(d_ok, 1.0), (d_ok, d_bad)


def test_segment_distances_use_the_nearest_line():
    lines = np.array([[1.0, 0.0, -100.0], [1.0, 0.0, -300.0]])     # x = 100 and x = 300
    segs = [YardLineSeg((104.0, 0.0), (104.0, 50.0)), YardLineSeg((290.0, 10.0), (290.0, 90.0))]
    d = gf.segment_distances_px(segs, lines)
    assert np.allclose(d, [4.0, 10.0])


def test_endzone_view_scores_its_horizontal_yard_lines():
    """An endzone-like camera sees the yard lines running across the image;
    the sideline's vertical filter finds nothing there, the orientation
    gate against the projected lines scores them."""
    import cv2

    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at
    from nfl_gsplat.field import footage_texture as ft
    from nfl_gsplat.field.procedural_field import render_field_texture, texture_extent

    res = 0.25
    field = render_field_texture(res, numbers=False)              # BGR, lines and hashes only
    R, t = look_at(np.array([-75.0, 0.0, 18.0]), np.array([-20.0, 0.0, 0.0]))   # behind an end zone
    K = intrinsics(960, 540, fov_deg=30.0)
    M = ft.ground_homography(K, R, t) @ ft.texel_to_ground(res, texture_extent())
    image = cv2.warpPerspective(field, M, (960, 540), flags=cv2.INTER_LINEAR)
    d_side, n_side = gf.grid_distance_px(image, K, R, t)
    d_end, n_end = gf.grid_distance_px(image, K, R, t, orient_tol_deg=25.0)
    assert n_end >= gf.MIN_SEGMENTS and np.isfinite(d_end) and d_end < 3.0, (d_end, n_end)
    assert n_end > n_side

