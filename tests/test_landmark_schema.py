import numpy as np


def test_number_anchor_landmarks_exist_with_correct_Y():
    from nfl_gsplat.calibration.field_landmarks import (
        NFL_LANDMARKS, NUMBER_BOTTOM_Y_M, NUMBER_TOP_Y_M, _yardline_x_m,
    )
    assert abs(NUMBER_BOTTOM_Y_M - 13.4112) < 1e-6
    assert abs(NUMBER_TOP_Y_M - 15.24) < 1e-6
    p = NFL_LANDMARKS["away_30_left_number_bottom"]
    assert np.allclose(p, [_yardline_x_m("away_30"), +NUMBER_BOTTOM_Y_M, 0.0])
    p = NFL_LANDMARKS["home_20_right_number_top"]
    assert np.allclose(p, [_yardline_x_m("home_20"), -NUMBER_TOP_Y_M, 0.0])
    assert "away_25_left_number_top" not in NFL_LANDMARKS
    assert "mid_50_left_number_top" in NFL_LANDMARKS


def test_schema_classes_scoped_and_indexable():
    from nfl_gsplat.landmarks.schema import LandmarkSchema
    s = LandmarkSchema(yard_min=-20.0, yard_max=20.0)
    names = s.class_names()
    assert s.num_classes == len(names) == len(set(names))
    assert s.index(names[0]) == 0 and s.index(names[-1]) == s.num_classes - 1
    for n in names:
        xyz = s.world_xyz(n)
        assert xyz.shape == (3,) and -20.0 <= xyz[0] <= 20.0
    assert any("number" in n for n in names) and any("hash" in n for n in names)
