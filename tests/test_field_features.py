from __future__ import annotations

from nfl_gsplat.calibration.field_features import landmark_name, yardline_label


def test_landmark_name_maps_side_and_row():
    assert landmark_name("home", 35, "left", "hash") == "home_35_left_hash"
    assert landmark_name("away", 20, "right", "sideline") == "away_20_right_sideline"
    assert landmark_name("mid", 50, "left", "sideline") == "mid_50_left_sideline"


def test_yardline_label_roundtrips_to_landmarks():
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    side, yd = yardline_label("home", 35)
    name = landmark_name(side, yd, "left", "hash")
    assert name in NFL_LANDMARKS
