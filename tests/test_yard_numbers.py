"""Absolute field position from painted yard numbers (field.yard_numbers)."""
import numpy as np
import pytest

from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M, YARD_TO_M
from nfl_gsplat.field import yard_numbers as yn


def _camera(width=1920, height=1080):
    """A broadcast-like camera: high on the -y side, looking across the field."""
    from nfl_gsplat.compositing.preview_cpu import look_at

    K = np.array([[3000.0, 0, width / 2], [0, 3000.0, height / 2], [0, 0, 1]])
    R, t = look_at(np.array([0.0, -80.0, 30.0]), np.array([0.0, 0.0, 0.0]))
    return K, R, t


def _synthetic_frame(K, R, t, centre_xy, label="20", width=1920, height=1080):
    """A dark frame with one numeral painted flat on the turf at ``centre_xy``."""
    import cv2

    ppm = yn.PATCH_PX_PER_M
    w_px, h_px = int(yn.PATCH_W_M * ppm), int(yn.PATCH_H_M * ppm)
    patch = np.zeros((h_px, w_px, 3), np.uint8)
    cv2.putText(patch, label, (10, h_px - 20), cv2.FONT_HERSHEY_DUPLEX, 2.4, (255, 255, 255), 5)
    M = yn.ground_homography(K, R, t) @ yn.patch_to_world(centre_xy, yn.PATCH_W_M,
                                                          yn.PATCH_H_M, ppm)
    frame = cv2.warpPerspective(patch, M, (width, height))
    return frame, patch


def test_rectify_round_trips_a_numeral_painted_on_the_turf():
    K, R, t = _camera()
    centre = (yn.yard_line_xs()[9], -yn.number_row_y())
    frame, patch = _synthetic_frame(K, R, t, centre)
    back = yn.rectify(frame, K, R, t, centre)
    a = back[..., 0].astype(float).ravel()
    b = patch[..., 0].astype(float).ravel()
    corr = np.corrcoef(a, b)[0, 1]
    assert back.shape == patch.shape
    assert corr > 0.9, corr


def test_candidate_patches_are_in_view_and_readable():
    K, R, t = _camera()
    cands = yn.candidate_patches(K, R, t, 1920, 1080)
    assert cands, "the camera sees the number rows"
    for x, side in cands:
        corners = yn.project_corners(K, R, t, (x, side * yn.number_row_y()),
                                     yn.PATCH_W_M, yn.PATCH_H_M)
        assert (corners >= 0).all() and (corners[:, 0] < 1920).all() and (corners[:, 1] < 1080).all()
        assert corners[:, 1].max() - corners[:, 1].min() >= yn.MIN_PATCH_HEIGHT_PX


def _readings_from_truth(turn, shift_m, lines_and_sides, conf=0.9, noise=0.0, seed=0):
    """What the reader would return if the solved frame were the rule-book
    frame turned/shifted: numerals at absolute lines, x reported in solved units."""
    rng = np.random.default_rng(seed)
    out = []
    for x_abs, side in lines_and_sides:
        numeral = yn.numeral_at(x_abs)
        if numeral is None:
            continue
        x_solved = (x_abs - shift_m) * (-1 if turn else 1)
        out.append(yn.Reading(x_solved + rng.normal(scale=noise), side, numeral, conf, False))
    return out


def _lands_on_its_numeral(tf, readings):
    return all(yn.numeral_at(tf.apply_points([r.x_m, 0.0])[0]) == r.numeral for r in readings)


@pytest.mark.parametrize("shift", [0.0, -3 * YARD_LINE_SPACING_M, 5 * YARD_LINE_SPACING_M])
def test_two_numerals_pin_the_shift(shift):
    lines = [(-30 * YARD_TO_M, 1), (-20 * YARD_TO_M, -1)]        # the 20 and the 30 on one half
    readings = _readings_from_truth(False, shift, lines, noise=0.3)
    tf = yn.solve_transform(readings)
    assert tf is not None
    assert tf.turn is False and abs(tf.shift_m - shift) < 1e-9
    assert tf.votes > tf.runner_up and _lands_on_its_numeral(tf, readings)


def test_a_turned_frame_still_lands_every_numeral():
    """The field is invariant under a half turn: a turned solved frame gets a
    shift that puts every reading on a line with its numeral, on the other half."""
    lines = [(-30 * YARD_TO_M, 1), (-20 * YARD_TO_M, -1), (-40 * YARD_TO_M, 1)]
    readings = _readings_from_truth(True, 2 * YARD_LINE_SPACING_M, lines)
    tf = yn.solve_transform(readings)
    assert tf is not None and _lands_on_its_numeral(tf, readings)


def test_a_lone_fifty_is_unique():
    tf = yn.solve_transform(_readings_from_truth(False, YARD_LINE_SPACING_M, [(0.0, 1)]))
    assert tf is not None and abs(tf.shift_m - YARD_LINE_SPACING_M) < 1e-9


def test_one_numeral_is_ambiguous():
    tf = yn.solve_transform(_readings_from_truth(False, 0.0, [(-20 * YARD_TO_M, 1),
                                                              (-20 * YARD_TO_M, -1)]))
    assert tf is None


def test_a_reading_off_the_grid_is_ignored():
    good = _readings_from_truth(False, YARD_LINE_SPACING_M, [(-30 * YARD_TO_M, 1),
                                                             (-20 * YARD_TO_M, -1)])
    stray = [yn.Reading(2.2, 1, 40, 0.9, False)]              # 2.2 m from any line
    tf = yn.solve_transform(good + stray)
    assert tf is not None and tf.turn is False and abs(tf.shift_m - YARD_LINE_SPACING_M) < 1e-9


def test_transformed_camera_sees_the_same_pixels():
    K, R, t = _camera()
    tf = yn.FieldTransform(turn=True, shift_m=-13.716, votes=2, runner_up=0, n_readings=2)
    X = np.array([[3.0, -12.0, 0.0], [10.0, 5.0, 1.8], [-20.0, 8.0, 0.0]])
    Xp = X.copy()
    Xp[:, :2] = tf.apply_points(X[:, :2])
    R2, t2 = yn.transform_camera(R, t, tf)
    before = (K @ (R @ X.T + t.reshape(3, 1))).T
    after = (K @ (R2 @ Xp.T + t2.reshape(3, 1))).T
    assert np.allclose(before[:, :2] / before[:, 2:], after[:, :2] / after[:, 2:], atol=1e-6)
    assert np.isclose(np.linalg.det(R2), 1.0)


class _FakeReader:
    """Reads a white numeral painted by _synthetic_frame: finds the bright blob."""

    def __init__(self, text="20", conf=0.9):
        self.text, self.conf = text, conf

    def readtext(self, img, **_kw):
        ys, xs = np.nonzero(img[..., 0] > 128)
        if len(ys) < 50:
            return []
        box = [[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]]
        return [(box, self.text, self.conf)]


def test_line_strip_finds_the_numeral_and_measures_its_y():
    K, R, t = _camera()
    x = yn.yard_line_xs()[9]
    y_true = 0.82 * yn.number_row_y()            # NOT where the rule book says: measured
    frame, _patch = _synthetic_frame(K, R, t, (x, y_true))
    readings = yn.read_line_strips(frame, K, R, t, _FakeReader("20"))
    hits = [r for r in readings if r.numeral == 20]
    assert len(hits) == 1
    r = hits[0]
    assert abs(r.x_m - x) < 1e-9 and r.side == 1
    assert abs(r.y_m - y_true) < 0.3, (r.y_m, y_true)


class _RowReader:
    """Scripted reader: per call index returns scripted (text, conf) boxes at a row."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def readtext(self, img, **_kw):
        i = self.calls
        self.calls += 1
        boxes = []
        # Odd calls are the turned strip: the same physical row sits at the
        # mirrored height there, as it would in a real turned image.
        frac = 0.75 if i % 2 else 0.25
        for text, conf in self.script.get(i, []):
            v = img.shape[0] * frac
            boxes.append(([[10, v - 10], [60, v - 10], [60, v + 10], [10, v + 10]], text, conf))
        return boxes


def test_row_orientation_is_decided_once_per_row(monkeypatch):
    """Three lines: turned reads '1','2','3' at 1.0; the third line ALSO reads
    an upright '20' at 0.9. Per-line scoring would take the '20' (two-digit
    bonus); the row faces one way, so the third line must read 30."""
    K, R, t = _camera()
    monkeypatch.setattr(yn, "Y_STRETCHES", (1.0,))
    monkeypatch.setattr(yn, "lines_in_view", lambda *a, **k: [-4.572, 0.0, 4.572])
    # Calls go line by line, (upright, turned) per line.
    script = {1: [("1", 1.0)], 3: [("2", 1.0)], 4: [("20", 0.9)], 5: [("3", 1.0)]}
    img = np.zeros((1080, 1920, 3), np.uint8)
    readings = yn.read_line_strips(img, K, R, t, _RowReader(script))
    by_x = {round(r.x_m, 3): r for r in readings}
    assert [by_x[k].numeral for k in sorted(by_x)] == [10, 20, 30]
    assert all(r.turned for r in readings)

