import cv2
import numpy as np
import pytest

from nfl_gsplat.calibration import field_model_fit as fmf
from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
from nfl_gsplat.errors import CalibrationError


def _horiz_lines(ys, w=400):
    """Near-horizontal yard lines — the real endzone orientation."""
    return [YardLineSeg((0.0, float(y)), (float(w), float(y) + 1.0)) for y in ys]


def test_anchor_determines_labels():
    lines = _horiz_lines([100, 150, 200, 250, 300])
    xs = fmf.label_yard_lines(lines, anchor=((200.0, 200.0), -9.144))
    assert np.isclose(xs[2], -9.144, atol=1e-6)
    assert np.allclose(np.abs(np.diff(xs)), YARD_LINE_SPACING_M, atol=1e-6)


def test_shifted_prior_cannot_move_the_labels():
    """The bug this task exists to prevent: the prior must not relabel."""
    lines = _horiz_lines([100, 150, 200, 250, 300])
    a = fmf.label_yard_lines(lines, anchor=((200.0, 200.0), -9.144))
    b = fmf.label_yard_lines(lines, anchor=((200.0, 200.0), -9.144),
                             yard_range_m=(-30.0, 10.0))
    assert np.allclose(a, b), "labels must come from the anchor, not the prior"


def test_missing_anchor_fails_loud():
    with pytest.raises(CalibrationError, match="anchor"):
        fmf.label_yard_lines(_horiz_lines([100, 150, 200]), anchor=None)


def test_prior_contradiction_fails_loud():
    """The prior is a validator: if it disagrees with the anchor, raise."""
    with pytest.raises(CalibrationError, match="contradict"):
        fmf.label_yard_lines(_horiz_lines([100, 150, 200]),
                             anchor=((200.0, 100.0), -9.144),
                             yard_range_m=(30.0, 45.0))


def test_labels_outside_painted_field_fail_loud():
    lines = _horiz_lines([100 + 50 * k for k in range(8)])
    with pytest.raises(CalibrationError, match="painted field"):
        fmf.label_yard_lines(lines, anchor=((200.0, 100.0), 44.0))


def test_nonfinite_prior_fails_loud_and_does_not_hang():
    with pytest.raises(CalibrationError, match="finite"):
        fmf.label_yard_lines(_horiz_lines([100, 150, 200]),
                             anchor=((200.0, 100.0), -9.144),
                             yard_range_m=(float("nan"), 10.0))


def test_detect_merges_multi_segment_hough_output():
    """One physical line must yield ONE segment, not many fragments."""
    votes = np.zeros((300, 400), np.float32)
    for y in (80, 160, 240):
        cv2.line(votes, (10, y), (390, y), 1.0, 3)
        cv2.line(votes, (200, y), (240, y), 0.0, 5)   # cable gap mid-span
    lines = fmf.detect_accumulated_lines(votes)
    assert len(lines) == 3, f"expected 3 merged lines, got {len(lines)}"
