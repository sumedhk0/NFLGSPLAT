import numpy as np

from nfl_gsplat.calibration.field_homography import LabelResult, fit_plane_homography


def test_fit_plane_homography_recovers_known_H():
    import cv2
    world = np.array([[-20.0, 3.0], [20.0, 3.0], [20.0, -3.0], [-20.0, -3.0]])
    image = np.array([[300.0, 200.0], [1600.0, 220.0], [1500.0, 850.0], [400.0, 870.0]])
    H = fit_plane_homography(world, image)
    assert H is not None
    proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.allclose(proj, image, atol=1e-6)


def test_fit_plane_homography_too_few_points_returns_none():
    assert fit_plane_homography(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_label_result_is_frozen_dataclass():
    r = LabelResult(correspondences=[], homography=None, inlier_count=0)
    assert r.inlier_count == 0 and r.correspondences == [] and r.homography is None
