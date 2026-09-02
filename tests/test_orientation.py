"""The quarter turn must be exact: a rotated solve is the same camera."""
import numpy as np

from nfl_gsplat.calibration.orientation import (
    camera_from_rotated,
    rotate_images_90,
    rotate_points_90,
)

W, H = 1920, 1080


def synth_camera(f=2400.0, centre=(0.0, -90.0, 40.0)):
    centre = np.asarray(centre, float)
    fwd = -centre / np.linalg.norm(centre)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K, R, -R @ centre


def project(cam, pts):
    K, R, t = cam
    q = (np.asarray(pts, float) @ R.T + t) @ K.T
    return q[:, :2] / q[:, 2:3]


def test_the_point_map_matches_what_cv2_actually_does():
    import cv2

    img = np.zeros((H, W), np.uint8)
    u, v = 300, 700
    img[v, u] = 255
    rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    got = np.argwhere(rot == 255)[0][::-1]      # (u, v)
    assert np.allclose(got, rotate_points_90([[u, v]], W)[0])


def test_a_camera_solved_from_rotated_frames_maps_back_exactly():
    cam = synth_camera()
    pts = np.c_[np.random.default_rng(0).uniform(-20, 20, (40, 2)), np.zeros(40)]
    seen = project(cam, pts)
    turned = rotate_points_90(seen, W)

    # The camera that explains the TURNED picture, built the way the solver
    # would see it: same world, image axes rotated.
    K, R, t = cam
    K_rot = np.array([[K[1, 1], 0, K[1, 2]],
                      [0, K[0, 0], (W - 1) - K[0, 2]],
                      [0, 0, 1.0]])
    R90 = np.array([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1.0]])
    rot_cam = (K_rot, R90.T @ R, R90.T @ t)
    assert np.allclose(project(rot_cam, pts), turned, atol=1e-6)

    back = camera_from_rotated(*rot_cam, W)
    assert np.allclose(project(back, pts), seen, atol=1e-6)
    assert np.isclose(np.linalg.det(back[1]), 1.0)      # a real camera


def test_rotating_frames_gives_a_portrait_picture():
    imgs = {0: np.zeros((H, W, 3), np.uint8)}
    out, w, h = rotate_images_90(imgs)
    assert (w, h) == (H, W)
    assert out[0].shape[:2] == (W, H)


def test_the_length_gates_do_not_depend_on_which_way_up_the_frame_is():
    """A turned frame must not face a taller bar than an upright one.

    Both detectors measured their minimum line length against the image height
    or width. Identical for every landscape frame the project has ever seen --
    and on a quarter-turned frame the bar jumped 78%, so the endzone solve found
    no sidelines at all and produced no camera.
    """
    import cv2

    from nfl_gsplat.calibration import field_detect
    from nfl_gsplat.calibration.orientation import rotate_images_90

    # A synthetic pitch: long lines one way, cross lines the other.
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 90, 40)
    for x in range(200, W - 200, 260):
        cv2.line(img, (x, 60), (x, H - 60), (255, 255, 255), 5)
    for y in (200, H - 200):
        cv2.line(img, (60, y), (W - 60, y), (255, 255, 255), 5)

    cfg = field_detect.FieldDetectConfig()
    up = field_detect.detect_field_features(img, cfg=cfg)
    turned, _w, _h = rotate_images_90({0: img})
    rot = field_detect.detect_field_features(turned[0], cfg=cfg)

    # The families swap, and neither may come back empty.
    assert len(up.yard_lines) >= 3 and len(up.sidelines) >= 1
    assert len(rot.sidelines) >= 3 and len(rot.yard_lines) >= 1
