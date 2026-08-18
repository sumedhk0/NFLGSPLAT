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


def test_step_check_error_names_line_count_and_offsets():
    """I2: the step-check failure must be diagnosable -- name the detected
    line count and offsets, not just the wrong step."""
    lines = _horiz([100, 150, 200, 300])          # the y=250 line is absent
    with pytest.raises(CalibrationError, match=r"Detected 4 lines at offsets"):
        fmf.label_yard_lines(lines, anchors=_anchors(100, -18.288, 300, 0.0))


def test_anchor_margin_uses_local_gap_not_global_min():
    """I2: the anchor click tolerance is gated on the LOCAL gap at the
    matched line, not the GLOBAL minimum gap across every detected line. A
    single close/compressed pair of lines elsewhere in the mosaic (far lines
    fall under merge_tol_px on a full-field render) must not force an
    unreasonably tight click tolerance on a widely-separated OUTERMOST line
    the operator is actually told to anchor (reviewer measured: 3.4-5.4 px
    tolerance even at the outermost lines on a 1920-px mosaic)."""
    # Outermost lines each have a single 50px-away neighbour; an unrelated
    # close pair sits in the middle (5px apart) and drags the GLOBAL minimum
    # gap down to 5px -- but must not affect the tolerance at y=100.
    lines = _horiz([100, 150, 200, 205, 400])
    anchors = _anchors(103, -2 * S, 400, 2 * S)     # 3px off the true y=100 line
    # This fixture is deliberately NOT on a yard grid (gaps 50/50/5/195) -- it
    # exists to exercise the anchor click tolerance, so the ladder is supplied
    # rather than fitted. fit_yard_ladder has its own tests; left to itself it
    # would (correctly) reject these offsets before the anchor logic ran.
    xs = fmf.label_yard_lines(lines, anchors=anchors, indices=[0, 1, 2, 3, 4])
    assert np.isclose(xs[0], -2 * S)                # matched despite the 3px click


def test_anchor_margin_still_rejects_a_bad_click_near_a_tight_local_gap():
    """The local-gap relaxation must not become globally permissive: a click
    near a line whose OWN local gap is genuinely tight must still fail."""
    lines = _horiz([100, 150, 200, 205, 400])
    # nearest line is y=200 (local gap = min(gaps to 150, to 205) = 5px);
    # 3px off -> 3 > 0.4*5=2.0px, must still fail.
    anchors = _anchors(197, 0.0, 400, 2 * S)
    with pytest.raises(CalibrationError, match="anchor"):
        # ladder supplied for the same reason as the test above
        fmf.label_yard_lines(lines, anchors=anchors, indices=[0, 1, 2, 3, 4])


def test_outermost_rule_refuses_inner_anchors():
    """Reviewer's repro, restated for v4: anchoring two INNER lines used to
    silently mislabel lines outside their span, because the step BETWEEN those
    two anchors is locally correct (they really are adjacent). Must refuse.

    The set here is evenly spaced, so the LADDER is unambiguous -- otherwise
    this would fail on the ambiguity check instead and stop testing the
    outermost rule at all (the original 4-line fixture, with the 4th of 5 lines
    absent, is now caught earlier by test_missing_line_fails_loud)."""
    lines = _horiz([100, 150, 200, 250, 300])
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


def test_detect_accumulated_lines_drops_non_parallel_sidelines():
    """C2 RED/GREEN: real endzone footage always shows both sidelines. Before
    the fix, detect_accumulated_lines returned every merged group regardless
    of orientation, so the reviewer's render (11 merged lines including two
    sidelines at +-43.5deg, symmetric about the image centre so their
    midpoint offsets are IDENTICAL) drove gaps.min() to ~0 and broke the
    anchor-margin check downstream. Only the largest cluster of MUTUALLY
    near-parallel lines (the yard-line pencil) must survive; the two
    non-parallel sidelines here must be dropped entirely, before any offset/
    rank/gap math runs."""
    h, w = 400, 600
    votes = np.zeros((h, w), np.float32)
    for y in (80, 140, 200, 260, 320):
        cv2.line(votes, (10, y), (w - 10, y), 1.0, 3)
    # Two sidelines converging toward a vanishing point, at a steep angle to
    # the near-horizontal yard-line pencil -- NOT part of it.
    cv2.line(votes, (0, 0), (w // 2, h - 1), 1.0, 3)
    cv2.line(votes, (w - 1, 0), (w // 2, h - 1), 1.0, 3)
    lines = fmf.detect_accumulated_lines(votes)
    assert len(lines) == 5, f"expected only the 5 yard lines, got {len(lines)}"
    for seg in lines:
        n = fmf._line_normal(seg)
        assert abs(n[1]) > 0.9, f"non-yard-line orientation slipped through: {n}"


def test_largest_parallel_cluster_rejects_too_few_survivors():
    """A vote image with no clean pencil of >= 2 mutually-parallel lines
    (e.g. only crossing sidelines) must fail loud, not silently hand a
    single stray line to the rest of the pipeline."""
    h, w = 300, 400
    votes = np.zeros((h, w), np.float32)
    cv2.line(votes, (0, 0), (w // 2, h - 1), 1.0, 3)
    cv2.line(votes, (w - 1, 0), (w // 2, h - 1), 1.0, 3)
    with pytest.raises(CalibrationError, match="mutually-parallel"):
        fmf.detect_accumulated_lines(votes)


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
