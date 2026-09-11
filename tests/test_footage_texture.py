"""field.footage_texture: the ground comes back where the camera saw it."""
import numpy as np

from nfl_gsplat.field import footage_texture as ft
from nfl_gsplat.field.procedural_field import render_field_texture, texture_extent


def _camera():
    """A sideline-like camera 40 m out, 20 m up, looking at the field centre."""
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

    R, t = look_at(np.array([0.0, -40.0, 20.0]), np.array([0.0, 0.0, 0.0]))
    K = intrinsics(640, 360, fov_deg=60.0)
    return K, R, t


def test_ground_homography_matches_projection():
    from nfl_gsplat.compositing.appearance import project

    K, R, t = _camera()
    H = ft.ground_homography(K, R, t)
    pts = np.array([[3.0, -2.0, 0.0], [-10.0, 5.0, 0.0]])
    uv, _ = project(pts, K, R, t)
    p = np.column_stack([pts[:, 0], pts[:, 1], np.ones(2)]) @ H.T
    assert np.allclose(p[:, :2] / p[:, 2:3], uv, atol=1e-6)


def test_warp_recovers_the_field_the_camera_saw():
    import cv2

    res = 0.5
    extent = texture_extent()
    field = render_field_texture(res)
    K, R, t = _camera()
    # Render the field into the camera by the inverse mapping (image -> ground).
    M = ft.ground_homography(K, R, t) @ ft.texel_to_ground(res, extent)
    image = cv2.warpPerspective(field, M, (640, 360), flags=cv2.INTER_LINEAR)
    tex, valid = ft.warp_frame(image, K, R, t, res_m=res, extent=extent)
    assert tex.shape == field.shape and valid.shape == field.shape[:2]
    assert 0.05 < valid.mean() < 0.9                       # part of the field, not all
    err = np.abs(tex[valid].astype(float) - field[valid].astype(float)).mean()
    assert err < 12.0, err                                  # resampling blur on the paint only
    # Behind the camera or off-frame is not valid: the far end zone is out of a 60 deg view.
    assert not valid[:, :5].any() or not valid[:, -5:].any()


def test_median_ignores_a_passing_player_and_counts_frames():
    h, w = 8, 10
    base = np.full((h, w, 3), 100, np.uint8)
    frames = []
    for i in range(9):
        img = base.copy()
        img[2, i] = 255                                      # a bright body moving along row 2
        valid = np.ones((h, w), bool)
        valid[7, :] = i < 3                                  # row 7 seen by only 3 frames
        frames.append((img, valid))
    med, count = ft.median_texture(frames, min_count=8)
    assert (med[2] == 100).all() and (med[:7] == 100).all()
    assert (count[7] == 3).all() and (med[7] == 0).all()


def test_composite_uses_footage_only_where_seen():
    proc = np.full((20, 20, 3), 50, np.uint8)
    foot = np.full((20, 20, 3), 200, np.uint8)
    count = np.zeros((20, 20), int)
    count[:, 10:] = 20
    out = ft.composite(proc, foot, count, min_count=8, feather_px=2)
    assert (out[:, :6] == 50).all() and (out[:, 15:] == 200).all()
    assert 50 < out[10, 11].mean() < 200                      # the seam is soft


def test_footage_colours_split_turf_from_paint():
    foot = np.full((40, 40, 3), (70, 130, 60), np.uint8)
    foot[:, ::10] = (230, 235, 230)                            # 10 % lines
    count = np.full((40, 40), 20)
    turf, paint = ft.footage_colours(foot, count, min_count=8)
    assert turf == (70, 130, 60) and paint == (230, 235, 230)
    assert ft.footage_colours(foot, np.zeros((40, 40), int), min_count=8) == (None, None)


def test_person_boxes_are_not_sampled():
    import cv2

    res = 0.5
    extent = texture_extent()
    field = render_field_texture(res)
    K, R, t = _camera()
    M = ft.ground_homography(K, R, t) @ ft.texel_to_ground(res, extent)
    image = cv2.warpPerspective(field, M, (640, 360), flags=cv2.INTER_LINEAR)
    _tex, valid_all = ft.warp_frame(image, K, R, t, res_m=res, extent=extent)
    box = (300.0, 150.0, 340.0, 250.0)                       # a person mid-frame
    _tex, valid_box = ft.warp_frame(image, K, R, t, res_m=res, extent=extent, boxes=[box])
    lost = valid_all & ~valid_box
    assert 0 < lost.sum() < valid_all.sum()
    mask = ft.people_mask((360, 640, 3), [box])
    assert mask[200, 320] and not mask[50, 50]

