"""Tests for per-frame refinement and bundle adjustment.

Several of these are regression guards for bugs that produced CONFIDENT WRONG
answers during bring-up -- a solve that reported a small residual while sitting a
yard line out, or a metric that read 20 px on a camera known correct to 2.7 px.
Those are the failures worth pinning, because none of them announce themselves.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from nfl_gsplat.calibration.endzone_refine import (associate, bundle_adjust,
                                                   project_line, projected_shift,
                                                   refine_frame, verify_frame)
from nfl_gsplat.errors import CalibrationError

CX, CY = 960.0, 540.0
CENTRE = np.array([100.0, 1.5, 23.0])
WORLD_X = [-45.720 + 4.5720 * k for k in range(9)]


def _look_at(centre, target):
    fwd = np.asarray(target, float) - np.asarray(centre, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _truth(focal=19000.0, target=(-27.0, 0.0, 0.0)):
    return float(focal), _look_at(CENTRE, target)


def _lines_from(focal, rot, world_xs=WORLD_X):
    """The image lines a perfect detector would report for this camera."""
    out = []
    for x in world_xs:
        uv, vis = project_line(focal, rot, CENTRE, x, cx=CX, cy=CY)
        if vis.sum() < 8:
            continue
        p = uv[vis]
        ln = np.cross([p[0][0], p[0][1], 1.0], [p[-1][0], p[-1][1], 1.0])
        out.append(ln / np.hypot(ln[0], ln[1]))
    return out


def _perturb(rot, deg):
    """Tilt the camera, which moves yard lines ACROSS themselves.

    Rotating about the camera's Z axis is image roll: near-horizontal lines spin
    in place and their perpendicular offset barely changes, so a "wrong" camera
    built that way is not wrong in the direction these tests measure.
    """
    axis = np.array([1.0, 0.0, 0.0]) * np.radians(deg)
    dr, _ = cv2.Rodrigues(axis)
    return dr @ rot


def test_refine_recovers_a_perturbed_camera():
    """A camera nudged off truth should be pulled back by its own lines."""
    focal, rot = _truth()
    dets = _lines_from(focal, rot)
    bad_rot = _perturb(rot, 0.05)
    pairs = associate(focal, bad_rot, CENTRE, WORLD_X, dets, cx=CX, cy=CY)
    assert len(pairs) >= 3
    got = refine_frame(focal, bad_rot, CENTRE, pairs, cx=CX, cy=CY)
    assert got is not None
    before = projected_shift(focal, bad_rot, focal, rot, CENTRE, WORLD_X, cx=CX, cy=CY)
    after = projected_shift(got[0], got[1], focal, rot, CENTRE, WORLD_X, cx=CX, cy=CY)
    assert after < before / 3.0, f"refine did not converge: {before:.2f} -> {after:.2f}"


def test_offsensor_samples_cost_rather_than_vanish():
    """Regression: zeroing off-sensor samples let the optimiser escape.

    Scoring an out-of-frame sample as 0 makes "push everything off the sensor" a
    global minimum with zero residual, and the solve destroyed the camera --
    focal ran away and the error came back nan. The cost must be positive.
    """
    from nfl_gsplat.calibration import endzone_refine as er
    assert er._OFF_SENSOR_COST > 0, "off-sensor samples must be penalised"

    focal, rot = _truth()
    dets = _lines_from(focal, rot)
    pairs = associate(focal, rot, CENTRE, WORLD_X, dets, cx=CX, cy=CY)
    got = refine_frame(focal, _perturb(rot, 0.03), CENTRE, pairs, cx=CX, cy=CY)
    assert got is not None
    assert np.isfinite(got[0]) and got[0] > 1000.0, "focal ran away"


def test_ambiguous_association_is_refused():
    """A match that could equally be the neighbouring line must be dropped.

    This is the failure that let an earlier sweep slide onto the wrong yard line
    while reporting 1.59 px: the fit stayed self-consistent, so nothing
    downstream could tell.
    """
    focal, rot = _truth()
    dets = _lines_from(focal, rot)
    # halfway between two lines: nearest and runner-up are equally plausible
    half = (WORLD_X[3] + WORLD_X[4]) / 2.0
    strict = associate(focal, rot, CENTRE, [half], dets, cx=CX, cy=CY,
                       require_unambiguous=True)
    loose = associate(focal, rot, CENTRE, [half], dets, cx=CX, cy=CY,
                      require_unambiguous=False)
    assert strict == [], "an ambiguous match was accepted"
    assert len(loose) == 1, "fixture is wrong: nothing matched even loosely"


def test_slip_guard_rejects_a_whole_line_swap():
    """Moving onto the adjacent yard line is a swap, not a correction."""
    focal, rot = _truth()
    dets = _lines_from(focal, rot)
    # Offset far enough that each projected line sits on its NEIGHBOUR, then
    # associate from there -- so the pairs themselves are the wrong lines and
    # refinement converges happily onto a swapped, self-consistent solution.
    swapped = _perturb(rot, 0.33)
    shift = projected_shift(focal, swapped, focal, rot, CENTRE, WORLD_X, cx=CX, cy=CY)
    assert shift > 55.0, f"fixture too gentle to test the guard ({shift:.1f} px)"
    pairs = associate(focal, swapped, CENTRE, WORLD_X, dets, cx=CX, cy=CY,
                      require_unambiguous=False)
    assert len(pairs) >= 3, "fixture matched nothing"
    got = refine_frame(focal, swapped, CENTRE, pairs, cx=CX, cy=CY,
                       max_shift_px=55.0, anchor=(focal, rot))
    assert got is None, "a yard-line swap passed the guard"


def test_projected_shift_ignores_off_sensor_points():
    """Regression: including off-sensor samples inflated the guard.

    A line projecting far outside the frame moves enormously for a tiny camera
    change. Counting those made the slip guard reject 926 of 1004 honest
    refinements, so the measure must look only at what lands on the sensor.
    """
    focal, rot = _truth()
    near = projected_shift(focal, _perturb(rot, 0.01), focal, rot, CENTRE,
                           WORLD_X, cx=CX, cy=CY)
    # a line far behind the camera's field of view must not dominate
    with_far = projected_shift(focal, _perturb(rot, 0.01), focal, rot, CENTRE,
                               WORLD_X + [400.0], cx=CX, cy=CY)
    assert with_far == pytest.approx(near, rel=0.25)


def test_verify_refuses_a_frame_whose_control_does_not_discriminate():
    """A metric that cannot tell right from wrong must not certify anything.

    Four per-frame metrics during bring-up reported ~20 px on a camera known
    correct to 2.7 px. Small numbers are not evidence; separation from a
    deliberately wrong model is.
    """
    focal, rot = _truth()
    dets = _lines_from(focal, rot)
    offset, control, ok = verify_frame(focal, rot, CENTRE, dets, WORLD_X,
                                       cx=CX, cy=CY)
    assert ok and offset < 1.0 and control > 3.0 * offset

    # a wrong camera must fail, even though it still matches SOME line nearby
    bad = _perturb(rot, 0.12)
    _off_b, _ctl_b, ok_b = verify_frame(focal, bad, CENTRE, dets, WORLD_X,
                                        cx=CX, cy=CY)
    assert not ok_b, "a visibly wrong camera was certified"


def test_bundle_refuses_to_run_without_anchors():
    """Chain residuals alone admit a rigidly wrong solution."""
    focal, rot = _truth()
    initial = {0: (focal, rot), 5: (focal, _perturb(rot, 0.02))}
    with pytest.raises(CalibrationError, match="anchor"):
        bundle_adjust({}, {}, CENTRE, initial, cx=CX, cy=CY)


def test_bundle_pulls_a_drifted_chain_back():
    """Chain + anchor together should recover nodes that drifted apart."""
    focal, rot = _truth()
    nodes = [0, 5, 10, 15]
    truth = {}
    for i, n in enumerate(nodes):
        truth[n] = (focal, _perturb(rot, 0.02 * i))

    def homography(a, b):
        f_a, r_a = truth[a]
        f_b, r_b = truth[b]
        k_a = np.array([[f_a, 0, CX], [0, f_a, CY], [0, 0, 1.0]])
        k_b = np.array([[f_b, 0, CX], [0, f_b, CY], [0, 0, 1.0]])
        return k_b @ r_b @ r_a.T @ np.linalg.inv(k_a)

    pair_h = {(nodes[i], nodes[i + 1]): homography(nodes[i], nodes[i + 1])
              for i in range(len(nodes) - 1)}
    anchors = {nodes[0]: [(x, ln) for x, ln in
                          zip(WORLD_X, _lines_from(*truth[nodes[0]]))]}
    # start every node at the first node's pose: the later ones are wrong
    initial = {n: truth[nodes[0]] for n in nodes}
    out = bundle_adjust(pair_h, anchors, CENTRE, initial, cx=CX, cy=CY,
                        max_nfev=200)

    worst_before = max(
        projected_shift(*initial[n], *truth[n], CENTRE, WORLD_X, cx=CX, cy=CY)
        for n in nodes)
    worst_after = max(
        projected_shift(*out[n], *truth[n], CENTRE, WORLD_X, cx=CX, cy=CY)
        for n in nodes)
    assert worst_after < worst_before / 3.0, (
        f"bundle did not converge: {worst_before:.2f} -> {worst_after:.2f} px")
