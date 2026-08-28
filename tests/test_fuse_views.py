"""Fusing two cameras' independent placements of one player."""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.pose.fuse_views import (SIGMA_ACROSS_M, SIGMA_ALONG_M,
                                        disagreement, fuse_skeletons,
                                        ray_directions, summarise)

# Roughly play_001's arrangement: sideline off to -Y, endzone beyond +X.
SIDELINE_C = np.array([-3.6, 80.5, 35.9])
ENDZONE_C = np.array([100.8, 1.6, 23.2])


def _skeleton(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(-20, 0.3, n), rng.normal(0, 0.3, n),
                            rng.uniform(0.1, 1.8, n)])


def test_rays_are_unit_length_and_point_away_from_the_camera():
    pts = _skeleton()
    rays = ray_directions(pts, SIDELINE_C)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
    # each ray, scaled back out, returns the point
    dist = np.linalg.norm(pts - SIDELINE_C, axis=1, keepdims=True)
    assert np.allclose(SIDELINE_C + rays * dist, pts)


def test_a_point_at_the_camera_centre_is_refused():
    with pytest.raises(ValueError, match="coincides with the camera"):
        ray_directions(SIDELINE_C[None, :], SIDELINE_C)


def test_identical_inputs_fuse_to_themselves():
    truth = _skeleton()
    got = fuse_skeletons(truth, truth, SIDELINE_C, ENDZONE_C)
    assert np.allclose(got, truth, atol=1e-9)


def test_fusion_trusts_each_camera_across_its_own_ray():
    """The point of the whole module: each camera's error along its viewing ray
    is the other camera's strong direction, so the fused answer should beat a
    plain average."""
    truth = _skeleton()
    # Push each estimate off the truth ALONG that camera's ray -- the direction
    # a monocular estimate is genuinely bad at.
    err = 1.2
    a = truth + err * ray_directions(truth, SIDELINE_C)
    b = truth - err * ray_directions(truth, ENDZONE_C)
    fused = fuse_skeletons(a, b, SIDELINE_C, ENDZONE_C)
    mean = 0.5 * (a + b)
    fused_err = float(np.median(np.linalg.norm(fused - truth, axis=1)))
    mean_err = float(np.median(np.linalg.norm(mean - truth, axis=1)))
    assert fused_err < mean_err, f"fused {fused_err:.3f} vs mean {mean_err:.3f}"


def test_fusion_falls_back_to_the_average_when_geometry_cannot_help():
    """Two views from the SAME point carry no complementary information, so the
    weighting must not invent any -- it lands exactly on the plain mean.

    The centres have to coincide, not merely share a bearing. Two cameras at
    different distances along one line still see a joint from slightly different
    angles, and because confidence across a ray is about 50x that along it, even
    a fraction of a degree moves the result measurably -- 0.13 m for centres
    200 m apart on one bearing. That is the fusion working, not failing.
    """
    truth = _skeleton()
    centre = np.array([-20.0, 300.0, 3.0])
    a, b = truth + 0.4, truth - 0.4
    fused = fuse_skeletons(a, b, centre, centre)
    assert np.allclose(fused, 0.5 * (a + b), atol=1e-9)


def test_mismatched_skeletons_are_refused():
    with pytest.raises(ValueError, match="differ in shape"):
        fuse_skeletons(_skeleton(20), _skeleton(19), SIDELINE_C, ENDZONE_C)
    with pytest.raises(ValueError, match="differ in shape"):
        disagreement(_skeleton(20), _skeleton(19))


def test_summarise_separates_a_whole_body_offset_from_a_pose_error():
    """A large root with tiny shape spread means the two cameras PLACED the
    player differently while agreeing on the pose -- a calibration problem, not
    a pose one, and the summary has to tell them apart."""
    truth = _skeleton()
    shifted = truth + np.array([2.0, 0.0, 0.0])
    out = summarise(truth, shifted)
    assert out["root_m"] == pytest.approx(2.0, abs=1e-9)
    assert out["shape_median_m"] < 1e-9
    assert out["median_m"] == pytest.approx(2.0, abs=1e-9)


def test_sigma_ratio_is_what_drives_the_weighting():
    """Documented behaviour: the fusion depends on the RATIO, not the absolutes."""
    truth = _skeleton()
    a = truth + 0.9 * ray_directions(truth, SIDELINE_C)
    b = truth - 0.9 * ray_directions(truth, ENDZONE_C)
    one = fuse_skeletons(a, b, SIDELINE_C, ENDZONE_C,
                         sigma_along_m=SIGMA_ALONG_M, sigma_across_m=SIGMA_ACROSS_M)
    two = fuse_skeletons(a, b, SIDELINE_C, ENDZONE_C,
                         sigma_along_m=10 * SIGMA_ALONG_M,
                         sigma_across_m=10 * SIGMA_ACROSS_M)
    assert np.allclose(one, two, atol=1e-9)


def test_fusion_is_symmetric_in_its_two_cameras():
    """Swapping which camera is 'a' must not change the answer. It did while the
    covariance was evaluated at each camera's own estimate rather than at one
    shared reference point."""
    truth = _skeleton()
    a = truth + 0.9 * ray_directions(truth, SIDELINE_C)
    b = truth - 0.9 * ray_directions(truth, ENDZONE_C)
    one = fuse_skeletons(a, b, SIDELINE_C, ENDZONE_C)
    two = fuse_skeletons(b, a, ENDZONE_C, SIDELINE_C)
    assert np.allclose(one, two, atol=1e-12)
