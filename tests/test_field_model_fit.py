import numpy as np
import pytest

from nfl_gsplat.calibration import field_model_fit as fmf
from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
from nfl_gsplat.errors import CalibrationError


def test_label_yard_lines_assigns_world_x_from_prior():
    # 5 equally spaced lines; the sideline says the action sits near X=-20..0
    lines = [(0.0, 100.0 + k * 50.0) for k in range(5)]
    xs = fmf.label_yard_lines(lines, yard_range_m=(-20.0, 0.0))
    assert len(xs) == 5
    d = np.diff(xs)
    assert np.allclose(np.abs(d), YARD_LINE_SPACING_M, atol=1e-6)
    assert min(xs) >= -25.0 and max(xs) <= 5.0     # inside the prior window


def test_label_yard_lines_fails_loud_when_ambiguous():
    # A prior window far wider than the observed span admits several offsets
    lines = [(0.0, 100.0 + k * 50.0) for k in range(3)]
    with pytest.raises(CalibrationError, match="ambiguous"):
        fmf.label_yard_lines(lines, yard_range_m=(-50.0, 50.0))


def test_label_yard_lines_rejects_inconsistent_spacing():
    lines = [(0.0, 100.0), (0.0, 150.0), (0.0, 260.0)]      # 50, 110 - not uniform
    with pytest.raises(CalibrationError, match="spacing"):
        fmf.label_yard_lines(lines, yard_range_m=(-20.0, 0.0))
