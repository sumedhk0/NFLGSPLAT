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


def _known_H():
    import cv2
    world = np.array([[-25.0, 3.0], [25.0, 3.0], [25.0, -3.0], [-25.0, -3.0]])
    image = np.array([[260.0, 180.0], [1660.0, 210.0], [1520.0, 900.0], [380.0, 930.0]])
    return cv2.getPerspectiveTransform(world.astype(np.float32), image.astype(np.float32))


def _project(H, X, Y):
    import cv2
    p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
    return (float(p[0]), float(p[1]))


def test_label_consensus_recovers_real_lines_rejects_noise():
    from nfl_gsplat.calibration.field_features import YardLineSeg
    from nfl_gsplat.calibration.field_homography import label_lines_by_consensus
    from nfl_gsplat.calibration.field_landmarks import HASH_OFFSET_M, _yardline_x_m

    H = _known_H()
    yards = [15, 20, 25, 30, 35, 40, 45]
    lines = []
    for y in yards:
        X = _yardline_x_m(f"away_{y}")
        top = _project(H, X, +HASH_OFFSET_M); bot = _project(H, X, -HASH_OFFSET_M)
        lines.append(YardLineSeg(top, bot))
    for sx in (520.0, 905.0, 1210.0, 1602.0):
        lines.append(YardLineSeg((sx, 150.0), (sx + 3.0, 950.0)))
    xL, xR = _yardline_x_m("away_15"), _yardline_x_m("away_45")
    row_top = YardLineSeg(_project(H, xL, +HASH_OFFSET_M), _project(H, xR, +HASH_OFFSET_M))
    row_bot = YardLineSeg(_project(H, xL, -HASH_OFFSET_M), _project(H, xR, -HASH_OFFSET_M))

    anchor_idx = yards.index(30)
    res = label_lines_by_consensus(
        lines, [row_top, row_bot], anchor_idx=anchor_idx,
        anchor_world_x=_yardline_x_m("away_30"), anchor_side="away", anchor_yard=30,
        direction=+1, image_size=(1920, 1080))
    assert res.inlier_count == 7
    names = {n for n, _ in res.correspondences}
    for y in yards:
        assert f"away_{y}_left_hash" in names and f"away_{y}_right_hash" in names
    import cv2
    pj = cv2.perspectiveTransform(
        np.array([[[_yardline_x_m("away_30"), +HASH_OFFSET_M]]], np.float64),
        res.homography).reshape(2)
    true_top = _project(H, _yardline_x_m("away_30"), +HASH_OFFSET_M)
    assert np.linalg.norm(pj - np.array(true_top)) < 2.0


def test_label_consensus_too_few_lines_empty():
    from nfl_gsplat.calibration.field_features import YardLineSeg
    from nfl_gsplat.calibration.field_homography import label_lines_by_consensus
    one = [YardLineSeg((800.0, 0.0), (800.0, 1080.0))]
    rows = [YardLineSeg((0, 300), (1920, 300)), YardLineSeg((0, 600), (1920, 600))]
    res = label_lines_by_consensus(one, rows, anchor_idx=0, anchor_world_x=0.0,
                                   anchor_side="mid", anchor_yard=50, direction=+1,
                                   image_size=(1920, 1080))
    assert res.inlier_count == 0 and res.correspondences == []
