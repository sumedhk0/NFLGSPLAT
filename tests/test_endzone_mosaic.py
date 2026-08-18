import cv2
import numpy as np

from nfl_gsplat.calibration import endzone_mosaic as em


def _textured_field(w=640, h=480, seed=0):
    """Deterministic textured image with strong corners so SIFT has features."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 40, np.uint8)
    for i in range(9):                       # bright 'yard lines'
        y = 40 + i * 45
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 2)
    for _ in range(300):                     # texture for feature matching
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), 2, (200, 200, 200), -1)
    return img


def _warp(img, H):
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))


def test_register_to_reference_recovers_known_homographies():
    base = _textured_field()
    truth = {0: np.eye(3)}
    frames = {0: base}
    for i, (dx, s) in enumerate([(12.0, 1.00), (25.0, 1.04), (-18.0, 0.97)], start=1):
        H = np.array([[s, 0.0, dx], [0.0, s, 0.5 * dx], [0.0, 0.0, 1.0]])
        truth[i] = H
        frames[i] = _warp(base, H)           # frame i = base warped by H

    H_by, inl = em.register_to_reference(frames, ref_idx=0)
    assert set(H_by) == {0, 1, 2, 3}
    assert np.allclose(H_by[0], np.eye(3), atol=1e-6)
    # H_by[i] maps frame i back INTO the reference, so it must invert truth[i]
    pts = np.float32([[100, 100], [500, 120], [300, 400]]).reshape(-1, 1, 2)
    for i in (1, 2, 3):
        back = cv2.perspectiveTransform(
            cv2.perspectiveTransform(pts, truth[i]), H_by[i])
        assert np.abs(back - pts).max() < 2.0, f"frame {i} round-trip off"
        assert inl[i] >= 25


def test_register_fails_loud_on_unregisterable_frame():
    import pytest

    from nfl_gsplat.errors import CalibrationError
    frames = {0: _textured_field(), 1: np.zeros((480, 640, 3), np.uint8)}
    with pytest.raises(CalibrationError, match="could not be registered"):
        em.register_to_reference(frames, ref_idx=0)


def test_keep_mask_zeroes_player_boxes_with_padding():
    m = em.keep_mask((100, 200, 3), [(50, 40, 70, 60)], pad=5)
    assert m[50, 60] == 0          # inside the box
    assert m[36, 46] == 0          # inside the pad
    assert m[10, 10] == 255        # untouched field
