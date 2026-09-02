"""Field coverage is a diagnostic. It must never decide which camera is right."""
import numpy as np

from nfl_gsplat.calibration.coverage import field_coverage

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


def test_a_wide_lens_high_over_the_field_sees_most_of_it():
    K, R, t = cam([0.0, -80.0, 40.0], 65.0)
    assert field_coverage(K, R, t, W, H) > 0.6


def test_the_right_all22_camera_sees_only_a_fifth_of_the_field():
    """The measured fact that made this a diagnostic and not a gate.

    The camera the paint solve kept choosing -- 95 m out behind a 12.2 degree
    lens -- IS the All-22 sideline camera: its person boxes are 140 px tall,
    which at that range only a lens this long produces, and it puts every
    player at 1.85 m on the turf. It sees about 13% of the field, because
    All-22 film frames the formation and pans. A gate at 35% rejected it in
    favour of a 62.5 degree camera that made the players five metres tall.
    """
    K, R, t = cam([29.5, -95.3, 47.1], 12.2)
    cover = field_coverage(K, R, t, W, H)
    assert cover < 0.25
    # And there is no threshold in this module to trip over.
    import nfl_gsplat.calibration.coverage as mod

    assert not [n for n in dir(mod) if "MIN" in n or "enough" in n]


def test_a_camera_pointed_away_from_the_field_sees_none_of_it():
    K, R, t = cam([0.0, -80.0, 40.0], 65.0, target=(0.0, -400.0, 40.0))
    assert field_coverage(K, R, t, W, H) == 0.0


def test_coverage_counts_only_what_is_in_front():
    """A camera in the middle of the field sees half of it, not all of it."""
    K, R, t = cam([0.0, 0.0, 3.0], 90.0, target=(60.0, 0.0, 3.0))
    assert 0.05 < field_coverage(K, R, t, W, H) < 0.6
