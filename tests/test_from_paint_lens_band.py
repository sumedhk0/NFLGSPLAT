"""from_paint.assignment_is_possible: a lens prior narrows the plausible band."""
import numpy as np

from nfl_gsplat.calibration import from_paint as fp
from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at

W, H = 1920, 1080


def _H(fov_deg):
    K = intrinsics(W, H, fov_deg=fov_deg)
    R, t = look_at(np.array([-4.0, -100.0, 42.0]), np.array([0.0, 0.0, 0.0]))
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def test_lens_band_rejects_the_stretched_labelling_and_keeps_the_right_one():
    wide, right = _H(41.0), _H(13.0)
    assert abs(fp.implied_fov_deg(wide, W, H) - 41.0) < 1.0
    assert abs(fp.implied_fov_deg(right, W, H) - 13.0) < 0.5
    assert fp.assignment_is_possible(wide, W, H)                         # the old band lets it through
    band = (11.7 / 1.6, 11.7 * 1.6)
    assert not fp.assignment_is_possible(wide, W, H, fov_band=band)
    assert fp.assignment_is_possible(right, W, H, fov_band=band)
