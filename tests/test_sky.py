"""render.sky: a night-stadium backdrop behind the splats, split at the camera's own horizon line."""
import numpy as np

from nfl_gsplat.render import sky


def _level_camera(pitch_deg=0.0, w=320, h=180, f=300.0):
    """A camera at the origin looking along world +y, z up, pitched down by ``pitch_deg``."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    # camera axes in world: x right = +x, y down = -z, z forward = +y; then pitch down about camera x
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # world -> camera
    p = np.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return K, Rx @ base, w, h


def test_horizon_of_a_level_camera_is_the_principal_row_and_moves_up_when_pitched_down():
    K, R, w, h = _level_camera(0.0)
    left, right = sky.horizon_rows(K, R, w)
    assert abs(left - h / 2) < 1e-6 and abs(right - h / 2) < 1e-6
    K, R, w, h = _level_camera(10.0)
    left, right = sky.horizon_rows(K, R, w)
    assert left < h / 2 - 20 and abs(left - right) < 1e-6          # looking down: the horizon rises in the image


def test_backdrop_is_dark_at_the_top_brighter_at_the_horizon_and_finite():
    K, R, w, h = _level_camera(5.0)
    img = sky.backdrop(K, R, w, h)
    assert img.shape == (h, w, 3) and np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
    top = img[0].mean()
    row = int(round(sky.horizon_rows(K, R, w)[0]))
    near = img[max(0, row - 8)].mean()
    assert near > top                                               # the glow of the stadium lights at the horizon
    below = img[min(h - 1, row + 20)].mean()
    assert below < near                                             # the stands under the rim
