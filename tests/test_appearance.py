"""Colours must come from the pixels the calibration says a vertex is under."""
import numpy as np

from nfl_gsplat.compositing import appearance as ap
from nfl_gsplat.compositing.appearance import (
    median_colours,
    project,
    sample_bilinear,
    vertex_colours_from_view,
)

W, H = 640, 360


def cam(centre, f=800.0, target=(0.0, 0.0, 1.0)):
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K, R, -R @ centre


def cube(size=1.0, centre=(0.0, 0.0, 1.0)):
    """A closed cube with outward normals."""
    c = np.asarray(centre, float)
    s = size / 2.0
    v = np.array([[-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
                  [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]]) + c
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    return v, f


def test_projection_puts_the_camera_target_at_the_principal_point():
    K, R, t = cam([0.0, -5.0, 1.0])
    uv, depth = project(np.array([[0.0, 0.0, 1.0]]), K, R, t)
    assert np.allclose(uv[0], [W / 2.0, H / 2.0])
    assert depth[0] > 0


def test_bilinear_sampling_interpolates_and_masks_outside():
    img = np.zeros((4, 4, 3))
    img[:, :, 0] = np.arange(4)[None, :]           # red ramps with u
    s = sample_bilinear(img, np.array([[1.5, 2.0], [-1.0, 0.0], [3.0, 3.0]]))
    assert np.isclose(s[0, 0], 1.5)
    assert np.isnan(s[1]).all()
    assert np.isclose(s[2, 0], 3.0)


def test_a_vertex_takes_the_pixel_under_it_and_back_faces_stay_unseen():
    """A cube seen from -y: the near face is sampled from a red image, the
    far face (facing away) is left NaN for another view to fill."""
    K, R, t = cam([0.0, -5.0, 1.0])
    v, f = cube()
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :, 0] = 255                             # red everywhere
    col = vertex_colours_from_view(v, f, K, R, t, img)
    near = v[:, 1] < 0                             # the -y face vertices
    assert np.allclose(col[near], [1.0, 0.0, 0.0])
    assert np.isnan(col[~near]).all()


def test_two_views_together_colour_the_whole_cube():
    front = cam([0.0, -5.0, 1.0])
    back = cam([0.0, 5.0, 1.0])
    v, f = cube()
    red = np.zeros((H, W, 3), np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros((H, W, 3), np.uint8)
    blue[:, :, 2] = 255
    a = vertex_colours_from_view(v, f, *front, red)
    b = vertex_colours_from_view(v, f, *back, blue)
    col, unseen = median_colours(np.stack([a, b]), fallback=(0.5, 0.5, 0.5))
    assert not unseen.any()
    assert np.allclose(col[v[:, 1] < 0], [1.0, 0.0, 0.0])
    assert np.allclose(col[v[:, 1] > 0], [0.0, 0.0, 1.0])


def test_the_fallback_fills_what_no_view_saw():
    v, f = cube()
    a = np.full((len(v), 3), np.nan)
    col, unseen = median_colours(a[None], fallback=(0.2, 0.3, 0.4))
    assert unseen.all()
    assert np.allclose(col, [0.2, 0.3, 0.4])



def test_mask_turf_drops_grass_samples_and_keeps_the_jersey():
    img = np.full((64, 64, 3), (0.30, 0.55, 0.25), np.float32)       # a turf frame
    img[20:40, 20:40] = (0.95, 0.95, 0.95)                             # a white jersey
    turf = ap.turf_colour(img, step=4)
    assert np.allclose(turf, (0.30, 0.55, 0.25), atol=0.01)
    sample = np.array([[0.31, 0.54, 0.26],       # grass
                       [0.95, 0.95, 0.95],       # jersey
                       [0.45, 0.55, 0.35],       # half grass, half jersey: still turf-like
                       [np.nan, np.nan, np.nan]])
    out = ap.mask_turf(sample, turf, dist=0.12)
    assert np.isnan(out[0]).all() and np.allclose(out[1], sample[1]) and np.isnan(out[3]).all()
    assert np.isnan(out[2]).all() is False or np.isfinite(out[2]).all()   # 0.19 away: kept
    assert np.isfinite(sample[0]).all()                                    # input untouched

