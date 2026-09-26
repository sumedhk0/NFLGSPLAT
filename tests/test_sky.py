"""render.sky: a night-stadium backdrop behind the splats -- turf, stands and crowd where a ray meets the ground,
sky where it does not."""
import numpy as np

from nfl_gsplat.render import sky


def _camera(pitch_deg=0.0, eye=(0.0, -30.0, 10.0), w=320, h=180, f=300.0):
    """A camera at ``eye`` looking along world +y, z up, pitched down by ``pitch_deg``; ``(K, R, t, w, h)``."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # world -> camera: x right, y down (-z), z fwd (+y)
    p = np.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    R = Rx @ base
    t = -R @ np.asarray(eye, float)
    return K, R, t, w, h


def test_horizon_of_a_level_camera_is_the_principal_row_and_moves_up_when_pitched_down():
    K, R, _t, w, h = _camera(0.0)
    left, right = sky.horizon_rows(K, R, w)
    assert abs(left - h / 2) < 1e-6 and abs(right - h / 2) < 1e-6
    K, R, _t, w, h = _camera(10.0)
    left, right = sky.horizon_rows(K, R, w)
    assert left < h / 2 - 20 and abs(left - right) < 1e-6          # looking down: the horizon rises in the image


def test_backdrop_sky_above_the_horizon_turf_near_the_field_crowd_beyond():
    K, R, t, w, h = _camera(5.0)
    img = sky.backdrop(K, R, t, w, h)
    assert img.shape == (h, w, 3) and np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
    top = img[0].mean(axis=0)
    assert np.allclose(top, sky.SKY_TOP, atol=0.02)                # far above the horizon: the night sky
    # the bottom rows land near the camera's own feet, on the field or its apron: turf
    assert np.allclose(img[-1, w // 2], sky.TURF, atol=1e-6)
    # a row just under the horizon lands far beyond the far sideline: the stands, not turf and not sky
    row = int(np.ceil(sky.horizon_rows(K, R, w)[0])) + 2
    far = img[row]
    assert not np.allclose(far, sky.TURF) and far.std() > 0.0     # a crowd, not a flat colour


def test_the_crowd_is_fixed_on_the_ground_not_in_the_image():
    """The crowd pattern belongs to the stands: the same ground cell keeps its colour as the camera moves."""
    xs = np.array([10.0, 10.2, 40.0]); ys = np.array([40.0, 40.1, 45.0])
    a = sky._crowd(xs, ys, seed=7); b = sky._crowd(xs, ys, seed=7)
    assert np.array_equal(a, b)
    assert np.array_equal(a[0], a[1])                              # one cell (0.55 m): one colour
