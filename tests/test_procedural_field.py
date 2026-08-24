"""Tests for the spec-drawn field.

These check METRIC correctness, not looks. The markings double as the geometric
reference the calibration is validated against, so a yard line in the wrong
place is a silent error that would propagate into every pose downstream.
"""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration.field_landmarks import (GOAL_LINE_X_M,
                                                    HALF_LENGTH_M,
                                                    HALF_WIDTH_M,
                                                    HASH_OFFSET_M,
                                                    YARD_LINE_SPACING_M)
from nfl_gsplat.compositing.merge_ply import load_gaussian_ply
from nfl_gsplat.errors import SetupError
from nfl_gsplat.field.procedural_field import (HASH_MARK_PITCH_M,
                                               render_field_texture,
                                               texture_extent,
                                               texture_to_gaussians,
                                               write_field_gaussian_ply)

RES = 0.04


def _sample(img, x_m, y_m, res_m=RES):
    """Brightest pixel within a paint-width of a world point."""
    x_min, _x_max, _y_min, y_max = texture_extent()
    col = int(round((x_m - x_min) / res_m))
    row = int(round((y_max - y_m) / res_m))
    patch = img[max(0, row - 2):row + 3, max(0, col - 2):col + 3]
    return int(patch.max()) if patch.size else 0


def test_yard_lines_sit_at_their_metric_positions():
    img = render_field_texture(RES, numbers=False)
    for i in range(int(2 * GOAL_LINE_X_M / YARD_LINE_SPACING_M) + 1):
        x = -GOAL_LINE_X_M + i * YARD_LINE_SPACING_M
        assert _sample(img, x, 0.0) > 200, f"no paint on the yard line at x={x:.3f}"


def test_between_yard_lines_is_turf():
    """If everything reads as paint the position test proves nothing."""
    img = render_field_texture(RES, numbers=False)
    x = -GOAL_LINE_X_M + 0.5 * YARD_LINE_SPACING_M
    assert _sample(img, x, 0.0) < 200, "midway between yard lines is painted"


def test_hash_rows_are_at_the_calibration_offset():
    """The endzone calibration takes its cross-field constraint from these."""
    img = render_field_texture(RES, numbers=False)
    # Hash marks sit every YARD, so sample at one that is not also a yard line
    # (x=0 is the 50 and is painted full width, which would pass for the wrong
    # reason). One yard off the 50 is a hash mark and nothing else.
    x_hash = HASH_MARK_PITCH_M
    for sign in (-1.0, 1.0):
        assert _sample(img, x_hash, sign * HASH_OFFSET_M) > 200, "hash row missing"
    assert _sample(img, x_hash, 1.5 * HASH_OFFSET_M) < 200, "hash rows are not localised"


def test_sidelines_and_end_lines_bound_the_field():
    img = render_field_texture(RES, numbers=False)
    assert _sample(img, 0.0, HALF_WIDTH_M) > 200
    assert _sample(img, 0.0, -HALF_WIDTH_M) > 200
    assert _sample(img, HALF_LENGTH_M, 0.0) > 200
    assert _sample(img, -HALF_LENGTH_M, 0.0) > 200


def test_gaussians_are_flat_and_lie_in_the_surface():
    """A plane needs no volume; discs in the surface, not blobs above it."""
    img = render_field_texture(0.2, numbers=False)
    xyz, rot, scale, opac, f_dc = texture_to_gaussians(img, 0.2)
    assert np.allclose(xyz[:, 2], 0.0)
    sigma = np.exp(scale)
    assert (sigma[:, 2] < sigma[:, 0] / 5).all(), "gaussians are not flattened"
    assert np.allclose(rot, np.array([1.0, 0.0, 0.0, 0.0])), "expected identity rotation"
    assert np.isfinite(f_dc).all() and np.isfinite(opac).all()


def test_ply_round_trips_through_the_repo_loader(tmp_path):
    """The field must be the SAME primitive as the players.

    render_gsplat merges the field batch with the avatar batches, so if this
    file does not load with the project's own reader the whole compositing path
    needs a special case.
    """
    out = write_field_gaussian_ply(tmp_path / "field.ply", 0.3, numbers=False)
    batch = load_gaussian_ply(out)
    batch.assert_no_nans()
    assert batch.sh_degree == 0
    assert batch.num_gaussians > 1000
    assert np.abs(batch.xyz[:, 0]).max() >= HALF_LENGTH_M
    assert np.allclose(batch.xyz[:, 2], 0.0)


def test_colour_survives_the_sh_encoding(tmp_path):
    """f_dc is a spherical-harmonic coefficient, not a colour.

    Getting the C0 factor wrong yields a field that loads fine and renders the
    wrong brightness, which is exactly the kind of error that survives review.
    """
    out = write_field_gaussian_ply(tmp_path / "field.ply", 0.3, numbers=False)
    batch = load_gaussian_ply(out)
    rgb = 0.5 + 0.28209479177387814 * batch.sh[:, :, 0]
    assert rgb.min() > -0.05 and rgb.max() < 1.05, "decoded colour out of range"
    green = rgb[:, 1]
    assert green.max() > 0.8, "no bright paint decoded"
    assert np.median(green) < 0.7, "everything decoded as paint"


def test_absurd_resolution_fails_loud():
    with pytest.raises(SetupError, match="coarser"):
        render_field_texture(0.0005)
