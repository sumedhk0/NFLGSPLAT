import pytest

from nfl_gsplat.calibration.field_landmarks import (
    _yardline_names, _yardline_x_m, yardline_name_from_x_m,
)


def test_round_trip_every_yardline():
    # THE consistency guarantee: name -> X -> name is the identity for all
    # 21 painted lines (away_goal..home_goal). A single mismatch here means
    # misidentified lines and silently wrong calibration.
    for name in _yardline_names():
        assert yardline_name_from_x_m(_yardline_x_m(name)) == name


def test_snaps_within_tolerance():
    x30_away = _yardline_x_m("away_30")            # -18.288
    assert yardline_name_from_x_m(x30_away + 0.4) == "away_30"
    assert yardline_name_from_x_m(x30_away - 0.4) == "away_30"


def test_rejects_between_lines():
    # halfway between away_30 and away_35 is 2.286 m from both — no snap
    x = 0.5 * (_yardline_x_m("away_30") + _yardline_x_m("away_35"))
    with pytest.raises(ValueError, match="no painted yard line"):
        yardline_name_from_x_m(x)


def test_rejects_beyond_goal_lines():
    with pytest.raises(ValueError, match="no painted yard line"):
        yardline_name_from_x_m(60.0)
