"""render.camera_path: a smoothed dolly on the play's centroid."""
import numpy as np
import pytest

from nfl_gsplat.render import camera_path as cp


def test_constant_centroid_gives_a_constant_path():
    frames = list(range(0, 100, 2))
    path = cp.follow_path(frames, {f: (10.0, -3.0) for f in frames})
    eyes = np.array([path[f][0] for f in frames])
    assert np.allclose(eyes, eyes[0])
    assert np.allclose(path[0][1], [10.0, -3.0, cp.TARGET_HEIGHT_M])
    assert np.allclose(path[0][0] - path[0][1], cp.EYE_OFFSET_M)


def test_smoothing_removes_jitter_and_keeps_the_drift():
    rng = np.random.default_rng(0)
    frames = list(range(300))
    drift = {f: (0.05 * f + rng.normal(0, 1.0), rng.normal(0, 1.0)) for f in frames}
    path = cp.smooth_track(frames, drift, sigma_frames=30)
    xs = np.array([path[f][0] for f in frames])
    assert np.std(np.diff(xs)) < 0.05                 # smooth frame to frame
    assert xs[250] - xs[50] > 8.0                     # but it follows the 10 m drift


def test_missing_frames_borrow_from_neighbours():
    frames = [0, 1, 2, 3, 4]
    path = cp.smooth_track(frames, {0: (0.0, 0.0), 4: (4.0, 0.0)}, sigma_frames=2)
    assert np.isfinite(path[2]).all() and 0.5 < path[2][0] < 3.5
    with pytest.raises(ValueError):
        cp.smooth_track(frames, {})
