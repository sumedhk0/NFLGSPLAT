import numpy as np


def test_render_gaussian_peaks_at_uv():
    from nfl_gsplat.landmarks.heatmap import render_gaussian
    h = render_gaussian((40, 60), (30.0, 20.0), sigma=2.0)
    assert h.shape == (40, 60) and h.dtype == np.float32
    iy, ix = np.unravel_index(int(np.argmax(h)), h.shape)
    assert (ix, iy) == (30, 20)
    assert abs(h.max() - 1.0) < 1e-5


def test_extract_peak_subpixel_and_threshold():
    from nfl_gsplat.landmarks.heatmap import extract_peak, render_gaussian
    h = render_gaussian((40, 60), (30.4, 20.0), sigma=2.0)
    got = extract_peak(h, thresh=0.3)
    assert got is not None
    (u, v), conf = got
    assert abs(u - 30.4) < 0.5 and abs(v - 20.0) < 0.5 and conf > 0.9
    assert extract_peak(np.zeros((40, 60), np.float32), thresh=0.3) is None
