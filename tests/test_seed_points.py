"""Tests for the splatfacto seed point cloud.

The regression these guard is expensive and silent: splatfacto with no seed
points random-initialises a +-10 unit cube at the origin, which in a metric
field scene (110 x 49 m, cameras 100+ m out) is a small box in the wrong place.
On play_001 that culled 38148 of 50000 gaussians at the FIRST densification
step and finished 30000 iterations with 11580 gaussians and a grey slab -- while
exiting 0 and writing a valid PLY, so nothing downstream noticed.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_landmarks import (GOAL_LINE_X_M,
                                                    HALF_LENGTH_M,
                                                    HALF_WIDTH_M)
from nfl_gsplat.field.seed_points import field_seed_points, write_seed_ply


def test_seed_covers_the_whole_field():
    """Points must span the playing surface, not a cube near the origin."""
    xyz, _rgb = field_seed_points(0.5)
    assert xyz[:, 0].min() <= -HALF_LENGTH_M
    assert xyz[:, 0].max() >= HALF_LENGTH_M
    assert xyz[:, 1].min() <= -HALF_WIDTH_M
    assert xyz[:, 1].max() >= HALF_WIDTH_M
    # the failure mode being guarded: everything inside a small origin cube
    assert np.abs(xyz[:, 0]).max() > 10.0, "seed cloud is smaller than splatfacto's random cube"


def test_seed_lies_on_the_playing_surface():
    xyz, _rgb = field_seed_points(0.5)
    assert np.allclose(xyz[:, 2], 0.0), "field is flat; seed points belong at z=0"


def test_yard_lines_are_painted():
    """Paint carries the structure the cameras key on.

    A uniform green field gives the optimiser almost no high-frequency signal,
    which matters more than usual here: two tripod centres mean very little
    parallax, so appearance is doing most of the work.
    """
    xyz, rgb = field_seed_points(0.1)
    bright = rgb.max(axis=1) > 200
    assert bright.sum() > 0, "no painted points at all"
    # a point right on the 50 yard line (x=0) inside the field must be paint
    on_fifty = (np.abs(xyz[:, 0]) < 0.05) & (np.abs(xyz[:, 1]) < HALF_WIDTH_M / 2)
    assert bright[on_fifty].any(), "the 50 yard line is not painted"
    # midway between two yard lines must be turf, or 'paint' means nothing
    between = (np.abs(xyz[:, 0] - 2.286) < 0.05) & (np.abs(xyz[:, 1]) < 5.0)
    assert not bright[between].all(), "everything is painted; the mask is meaningless"


def test_goal_lines_are_within_the_painted_set():
    xyz, rgb = field_seed_points(0.1)
    bright = rgb.max(axis=1) > 200
    on_goal = (np.abs(np.abs(xyz[:, 0]) - GOAL_LINE_X_M) < 0.05) & \
              (np.abs(xyz[:, 1]) < HALF_WIDTH_M / 2)
    assert bright[on_goal].any(), "goal line not painted"


def test_ply_is_readable_and_matches_the_point_count(tmp_path):
    """The header must declare exactly what the payload holds.

    nerfstudio reads this file; a header/payload mismatch is a truncated-read
    bug that would surface as a confusing dataparser error much later.
    """
    out = write_seed_ply(tmp_path / "field_seed.ply", 1.0)
    raw = out.read_bytes()
    head, _, payload = raw.partition(b"end_header\n")
    declared = int([line.split()[-1] for line in head.split(b"\n")
                    if line.startswith(b"element vertex")][0])
    assert declared > 0
    assert len(payload) == declared * (3 * 4 + 3), "payload size disagrees with the header"
    assert b"property uchar red" in head, "nerfstudio expects uint8 colour"


def test_jitter_breaks_the_lattice():
    """A perfectly regular lattice gives every gaussian an identical
    neighbourhood, so the split/duplicate heuristics fire on all of them at
    once."""
    xyz, _rgb = field_seed_points(0.5, jitter_m=0.05)
    xs = np.sort(np.unique(np.round(xyz[:, 0], 6)))
    assert len(xs) > 1000, "grid collapsed onto a few exact columns"
