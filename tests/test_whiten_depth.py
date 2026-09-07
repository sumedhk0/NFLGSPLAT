"""link3d.whiten_depth: the depth component shrinks, the across component holds; a depth-jittering
track links whole where the round gate broke it."""
import numpy as np

from nfl_gsplat.tracking import link3d


def test_whiten_scales_only_along_the_direction():
    pts = {0: np.array([[3.0, 4.0], [-1.0, 2.0]])}
    w = link3d.whiten_depth(pts, np.array([0.0, 1.0]), 1.0 / 3.0)
    assert np.allclose(w[0], [[3.0, 4.0 / 3.0], [-1.0, 2.0 / 3.0]])
    w2 = link3d.whiten_depth(pts, np.array([1.0, 1.0]), 0.5)      # a diagonal direction
    d = np.array([1.0, 1.0]) / np.sqrt(2)
    for a, b in zip(pts[0], w2[0]):
        assert np.isclose(b @ d, 0.5 * (a @ d)) and np.isclose(b @ [1, -1], a @ [1, -1])


def test_depth_direction_is_the_optical_axis_on_the_ground():
    import cv2

    R = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))[0]                   # camera z = world z: no ground component
    assert np.allclose(link3d.depth_direction(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)), [0.0, 1.0])


def test_depth_jitter_breaks_the_round_gate_and_not_the_whitened_one():
    rng = np.random.default_rng(0)
    fps = 60.0
    frames = range(120)
    placements = {}
    for f in frames:
        x = 0.05 * f
        y = 0.0 + rng.normal(0, 0.9)                                   # 0.9 m of depth jitter along y
        placements[f] = np.array([[x, y]])
    tracks = link3d.link(placements, fps=fps, min_frames=3)
    whitened = link3d.link(link3d.whiten_depth(placements, [0.0, 1.0], 1.0 / 3.0), fps=fps, min_frames=3)
    assert len(whitened) < len(tracks) or max(len(t.frames) for t in whitened) > max(len(t.frames) for t in tracks)
    assert max(len(t.frames) for t in whitened) >= 100
