"""All-22 footage shows the whole field; a camera that cannot is not the camera."""
import numpy as np

from nfl_gsplat.calibration.coverage import (
    field_coverage,
    sees_enough_of_the_field,
)

W, H = 1920, 1080


def cam(centre, fov_deg, target=(0.0, 0.0, 0.0)):
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    f = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K, R, -R @ centre


def test_a_real_all22_sideline_camera_sees_most_of_the_field():
    K, R, t = cam([0.0, -80.0, 40.0], 65.0)
    assert field_coverage(K, R, t, W, H) > 0.6
    assert sees_enough_of_the_field(K, R, t, W, H)


def test_the_telephoto_the_solve_kept_choosing_is_rejected():
    """The measured failure: 95 m out behind a 12.2 degree lens.

    It fits the paint, and it puts players at a believable height, because a
    camera can be the right distance away behind quite the wrong lens. It sees
    about a fifth of the field, which is what gives it away.
    """
    K, R, t = cam([29.5, -95.3, 47.1], 12.2)
    assert field_coverage(K, R, t, W, H) < 0.25
    assert not sees_enough_of_the_field(K, R, t, W, H)


def test_an_endzone_camera_also_passes():
    K, R, t = cam([-75.0, 0.0, 25.0], 40.0)
    assert sees_enough_of_the_field(K, R, t, W, H)


def test_a_camera_pointed_away_from_the_field_sees_none_of_it():
    K, R, t = cam([0.0, -80.0, 40.0], 65.0, target=(0.0, -400.0, 40.0))
    assert field_coverage(K, R, t, W, H) == 0.0


def test_coverage_counts_only_what_is_in_front():
    """A camera in the middle of the field sees half of it, not all of it."""
    K, R, t = cam([0.0, 0.0, 3.0], 90.0, target=(60.0, 0.0, 3.0))
    assert 0.05 < field_coverage(K, R, t, W, H) < 0.6
