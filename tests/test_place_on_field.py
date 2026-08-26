"""Tests for placing a SMPL-X skeleton into field coordinates.

The failure this guards is silent: a skeleton placed with the wrong scale or the
wrong plane still renders, still looks like a person, and only reveals itself as
players sunk into the turf or standing in the stands.
"""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.pose.place_on_field import (ANKLE_HEIGHT_M, ground_point,
                                            place_skeleton, stature)


def _camera(eye=(0.0, -80.0, 30.0), target=(0.0, 0.0, 0.0), focal=2000.0):
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.stack([right, down, fwd])
    k = np.array([[focal, 0, 960.0], [0, focal, 540.0], [0, 0, 1.0]])
    return k, rot, -rot @ eye


def _project(k, rot, tvec, world):
    cam = rot @ np.asarray(world, float) + tvec
    uvw = k @ cam
    return uvw[:2] / uvw[2]


def _upright_skeleton(rot, height=1.8, n=30):
    """Root-relative skeleton of a body standing VERTICAL IN THE WORLD,
    expressed in camera axes -- which is what the network returns.

    Building it along the camera's own -Y instead models a body perpendicular to
    the view axis, which is not what standing means: with the camera tilted down
    21 degrees that shortens the recovered stature by cos(21) and looks like a
    scale bug in the code under test.
    """
    zs = np.linspace(0.0, height, n) - height / 2.0
    world = np.zeros((n, 3))
    world[:, 2] = zs                      # vertical in WORLD
    return (np.asarray(rot, float) @ world.T).T


def test_ground_point_lands_where_it_was_projected_from():
    """Round trip: a known field point projects to a pixel, which must ground
    back to the same field point."""
    k, rot, tvec = _camera()
    truth = np.array([12.0, -5.0, 0.0])
    uv = _project(k, rot, tvec, truth)
    got = ground_point(uv, k, rot, tvec)
    assert np.allclose(got, truth, atol=1e-6), f"{got} != {truth}"


def test_ground_point_refuses_a_ray_that_misses_the_plane():
    """Silently returning a point behind the camera would put a player in the
    stands with no other symptom."""
    k, rot, tvec = _camera()
    # a pixel far above the horizon: the ray rises away from the field
    with pytest.raises(CalibrationError):
        ground_point(np.array([960.0, -5000.0]), k, rot, tvec)


def test_feet_land_on_the_turf_not_in_it():
    """The foot REFERENCE joint is pinned, not the absolute lowest one.

    Pinning the minimum would let a single badly estimated toe drag the whole
    body into the ground, so a low quantile is used instead; the lowest joint may
    then sit a few cm under, which is physically unremarkable.
    """
    from nfl_gsplat.pose.place_on_field import _FOOT_QUANTILE

    k, rot, tvec = _camera()
    foot_world = np.array([5.0, 3.0, 0.0])
    uv = _project(k, rot, tvec, foot_world)
    placed = place_skeleton(_upright_skeleton(rot), uv, k, rot, tvec)
    ref = float(np.quantile(placed[:, 2], _FOOT_QUANTILE))
    assert ref == pytest.approx(ANKLE_HEIGHT_M, abs=1e-6)
    assert placed[:, 2].min() > -0.10, "body sank into the turf"


def test_skeleton_stands_where_the_foot_pixel_says():
    k, rot, tvec = _camera()
    foot_world = np.array([-20.0, 8.0, 0.0])
    uv = _project(k, rot, tvec, foot_world)
    placed = place_skeleton(_upright_skeleton(rot), uv, k, rot, tvec)
    lowest = placed[placed[:, 2] <= placed[:, 2].min() + 1e-6]
    assert np.allclose(lowest[:, :2].mean(axis=0), foot_world[:2], atol=1e-6)


def test_stature_survives_the_placement():
    """Placement moves and rotates; it must not rescale. A height that drifts
    means the rotation is not orthonormal or the skeleton was distorted."""
    k, rot, tvec = _camera()
    uv = _project(k, rot, tvec, np.array([0.0, 0.0, 0.0]))
    placed = place_skeleton(_upright_skeleton(rot, height=1.83), uv, k, rot, tvec)
    assert stature(placed) == pytest.approx(1.83, abs=1e-6)


def test_ankle_height_is_actually_applied():
    """Ignoring it sinks every player by 8 cm, which reads as a stature error
    rather than a placement one -- measured ~5% before it was accounted for."""
    k, rot, tvec = _camera()
    uv = _project(k, rot, tvec, np.array([0.0, 0.0, 0.0]))
    on_turf = place_skeleton(_upright_skeleton(rot), uv, k, rot, tvec,
                             ankle_height_m=0.0)
    lifted = place_skeleton(_upright_skeleton(rot), uv, k, rot, tvec,
                            ankle_height_m=0.25)
    assert lifted[:, 2].min() - on_turf[:, 2].min() == pytest.approx(0.25, abs=1e-6)


def test_bad_shape_is_rejected():
    k, rot, tvec = _camera()
    with pytest.raises(ValueError, match=r"\[J, 3\]"):
        place_skeleton(np.zeros((10, 2)), np.array([960.0, 700.0]), k, rot, tvec)
