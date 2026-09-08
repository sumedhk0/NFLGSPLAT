"""yard_numbers.read_line_strips: a degenerate strip or a reader error is no reading, not a crash."""
import numpy as np

from nfl_gsplat.field import yard_numbers as yn


class _Reader:
    def __init__(self):
        self.calls = 0

    def readtext(self, img, **kw):
        self.calls += 1
        if min(img.shape[:2]) < 8:
            import cv2
            raise cv2.error("empty")
        return []


def test_empty_or_tiny_strips_are_skipped(monkeypatch):
    img = np.zeros((1080, 1920, 3), np.uint8)
    K = np.array([[9000.0, 0, 960], [0, 9000.0, 540], [0, 0, 1]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 100.0])
    monkeypatch.setattr(yn, "lines_in_view", lambda *a, **k: [0.0, 4.572])
    monkeypatch.setattr(yn, "rectify", lambda *a, **k: np.zeros((0, 40, 3), np.uint8))
    reader = _Reader()
    assert yn.read_line_strips(img, K, R, t, reader) == []
    assert reader.calls == 0


def test_a_reader_error_on_one_strip_is_swallowed(monkeypatch):
    import cv2

    img = np.zeros((1080, 1920, 3), np.uint8)
    K = np.array([[9000.0, 0, 960], [0, 9000.0, 540], [0, 0, 1]])
    monkeypatch.setattr(yn, "lines_in_view", lambda *a, **k: [0.0])
    monkeypatch.setattr(yn, "rectify", lambda *a, **k: np.zeros((40, 400, 3), np.uint8))

    class Boom:
        def readtext(self, img, **kw):
            raise cv2.error("resize")

    assert yn.read_line_strips(img, K, np.eye(3), np.array([0.0, 0.0, 100.0]), Boom()) == []
