import numpy as np


def test_number_anchor_landmarks_exist_with_correct_Y():
    from nfl_gsplat.calibration.field_landmarks import (
        NFL_LANDMARKS, NUMBER_CENTER_Y_M, _yardline_x_m,
    )
    assert abs(NUMBER_CENTER_Y_M - 14.3256) < 1e-6
    p = NFL_LANDMARKS["away_30_left_number"]
    assert np.allclose(p, [_yardline_x_m("away_30"), +NUMBER_CENTER_Y_M, 0.0])
    p = NFL_LANDMARKS["home_20_right_number"]
    assert np.allclose(p, [_yardline_x_m("home_20"), -NUMBER_CENTER_Y_M, 0.0])
    # numbers only at 10/20/30/40 + mid_50; single center anchor (top/bottom removed)
    assert "away_25_left_number" not in NFL_LANDMARKS
    assert "mid_50_left_number" in NFL_LANDMARKS
    assert "away_30_left_number_top" not in NFL_LANDMARKS


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
