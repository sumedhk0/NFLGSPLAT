import cv2
import numpy as np
import pytest

from nfl_gsplat.calibration import field_model_fit as fmf
from nfl_gsplat.calibration.field_features import YardLineSeg
from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
from nfl_gsplat.errors import CalibrationError

S = YARD_LINE_SPACING_M


def _horiz(ys, w=400, flip=()):
    """Near-horizontal yard lines. `flip` reverses endpoint order for those
    indices, mimicking the arbitrary ordering real detections come back with."""
    out = []
    for i, y in enumerate(ys):
        p0, p1 = (0.0, float(y)), (float(w), float(y) + 1.0)
        out.append(YardLineSeg(p1, p0) if i in flip else YardLineSeg(p0, p1))
    return out


def _anchors(y_a, x_a, y_b, x_b, w=400):
    return (((w / 2, float(y_a)), float(x_a)), ((w / 2, float(y_b)), float(x_b)))


def test_two_anchors_give_signed_labels():
    lines = _horiz([100, 150, 200, 250, 300])
    xs = fmf.label_yard_lines(lines, anchors=_anchors(100, -18.288, 300, 0.0))
    # SIGNED equality — not abs(); the sign is what v2 got wrong
    assert np.allclose(xs, [-18.288, -13.716, -9.144, -4.572, 0.0], atol=1e-6)


def test_labels_are_invariant_to_endpoint_order():
    """The v2 mirror bug: flipping some segments' endpoint order must not
    mirror the labeling."""
    ys = [100, 150, 200, 250, 300]
    a = fmf.label_yard_lines(_horiz(ys), anchors=_anchors(100, -18.288, 300, 0.0))
    b = fmf.label_yard_lines(_horiz(ys, flip=(0, 3)),
                             anchors=_anchors(100, -18.288, 300, 0.0))
    assert np.allclose(a, b), "endpoint order must not change the labeling"


def test_missing_line_fails_loud():
    """A dropped line would shift every later label by a full spacing."""
    lines = _horiz([100, 150, 200, 300])          # the y=250 line is absent
    with pytest.raises(CalibrationError, match="spacing|missing"):
        fmf.label_yard_lines(lines, anchors=_anchors(100, -18.288, 300, 0.0))


def test_outermost_rule_catches_a_missing_line_outside_the_anchor_span():
    """Reviewer's exact repro: with the 4th of 5 lines absent, anchoring the
    two adjacent INNER lines used to silently mislabel the last line by a full
    spacing, because the step BETWEEN those two anchors is locally correct
    (they really are adjacent) — only the span outside them is wrong. Must
    now raise instead of returning a wrong labeling."""
    lines = _horiz([100, 150, 200, 300])          # the y=250 line is absent
    with pytest.raises(CalibrationError, match="(?i)outermost"):
        fmf.label_yard_lines(lines, anchors=_anchors(150, -13.716, 200, -9.144))


def test_anchor_far_from_every_line_fails_loud():
    lines = _horiz([100, 150, 200])
    with pytest.raises(CalibrationError, match="anchor"):
        fmf.label_yard_lines(lines, anchors=_anchors(9e5, -9.144, 150, -4.572))


def test_both_anchors_on_same_line_fails_loud():
    lines = _horiz([100, 150, 200])
    with pytest.raises(CalibrationError, match="distinct"):
        fmf.label_yard_lines(lines, anchors=_anchors(100, -9.144, 102, -4.572))


def test_anchor_off_the_yard_grid_fails_loud():
    lines = _horiz([100, 150, 200])
    with pytest.raises(CalibrationError, match="grid"):
        fmf.label_yard_lines(lines, anchors=_anchors(100, -7.3, 200, -7.3 + 2 * S))


def test_missing_anchors_fails_loud():
    with pytest.raises(CalibrationError, match="anchor"):
        fmf.label_yard_lines(_horiz([100, 150]), anchors=None)


def test_coincident_offsets_fail_loud():
    """Two lines at an identical offset means merging upstream failed."""
    lines = _horiz([100, 100, 200])        # first two lines exactly coincide
    with pytest.raises(CalibrationError, match="coincident"):
        fmf.label_yard_lines(lines, anchors=_anchors(100, -9.144, 200, 0.0))


def test_labels_outside_painted_field_fail_loud():
    lines = _horiz([100 + 50 * k for k in range(8)])
    # Outermost anchors, internally consistent spacing, but centred far from
    # the field (multiples 9..16 of the yard-line spacing).
    anchors = _anchors(100, 9 * S, 450, 16 * S)
    with pytest.raises(CalibrationError, match="painted field"):
        fmf.label_yard_lines(lines, anchors=anchors)


def test_prior_is_validator_only():
    lines = _horiz([100, 150, 200, 250, 300])
    a = fmf.label_yard_lines(lines, anchors=_anchors(100, -18.288, 300, 0.0))
    b = fmf.label_yard_lines(lines, anchors=_anchors(100, -18.288, 300, 0.0),
                             yard_range_m=(-30.0, 10.0))
    assert np.allclose(a, b)
    with pytest.raises(CalibrationError, match="contradict"):
        fmf.label_yard_lines(lines, anchors=_anchors(100, -18.288, 300, 0.0),
                             yard_range_m=(30.0, 45.0))


def test_nonfinite_prior_fails_loud_either_slot():
    """min()/max() silently drop a NaN that is not first — check both slots."""
    lines = _horiz([100, 150, 200])
    for rng in ((float("nan"), 10.0), (10.0, float("nan"))):
        with pytest.raises(CalibrationError, match="finite"):
            fmf.label_yard_lines(lines, anchors=_anchors(100, -9.144, 200, 0.0),
                                 yard_range_m=rng)


def test_near_vertical_lines_also_label():
    lines = [YardLineSeg((float(x), 0.0), (float(x) + 1.0, 300.0))
             for x in (100, 150, 200)]
    xs = fmf.label_yard_lines(
        lines, anchors=(((100.0, 150.0), -9.144), ((200.0, 150.0), 0.0)))
    assert np.allclose(xs, [-9.144, -4.572, 0.0], atol=1e-6)


def test_detect_merges_fragments_across_a_cable_gap():
    """One physical line, painted with a cable gap in it, must yield ONE
    segment, not many fragments. detect_accumulated_lines deliberately carries
    no over-merge guard (both tried variants were measured unsound — see the
    module docstring); over-merging is instead caught in label_yard_lines as
    a line-count violation via the outermost-anchor rule, exercised above."""
    votes = np.zeros((300, 400), np.float32)
    for y in (80, 160, 240):
        cv2.line(votes, (10, y), (390, y), 1.0, 3)
        cv2.line(votes, (200, y), (240, y), 0.0, 5)      # cable gap
    lines = fmf.detect_accumulated_lines(votes)
    assert len(lines) == 3, f"expected 3 merged lines, got {len(lines)}"
