"""Reading the Helmet Assignment ground truth into field metres."""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.calibration.field_landmarks import GOAL_LINE_X_M, HALF_WIDTH_M
from nfl_gsplat.data.helmet_dataset import (PlayTracking, to_field_metres,
                                            video_name)


def test_midfield_maps_to_the_origin():
    """NFL x=60 is midfield and y=80/3 is the centre line; ours is (0, 0).
    Getting this wrong puts players on the field and 50 m from the truth."""
    x, y = to_field_metres(60.0, 160.0 / 3.0 / 2.0)
    assert float(x) == pytest.approx(0.0, abs=1e-9)
    assert float(y) == pytest.approx(0.0, abs=1e-9)


def test_goal_lines_land_where_this_project_puts_them():
    """NFL x=10 and x=110 are the goal lines, 50 yd either side of midfield."""
    x_near, _ = to_field_metres(10.0, 0.0)
    x_far, _ = to_field_metres(110.0, 0.0)
    assert float(x_near) == pytest.approx(-GOAL_LINE_X_M, abs=1e-6)
    assert float(x_far) == pytest.approx(GOAL_LINE_X_M, abs=1e-6)


def test_sidelines_land_on_the_field_width():
    _, y_low = to_field_metres(60.0, 0.0)
    _, y_high = to_field_metres(60.0, 160.0 / 3.0)
    assert float(y_low) == pytest.approx(-HALF_WIDTH_M, abs=1e-3)
    assert float(y_high) == pytest.approx(HALF_WIDTH_M, abs=1e-3)


def test_conversion_is_vectorised():
    x, y = to_field_metres(np.array([10.0, 60.0, 110.0]), np.array([0.0, 26.0, 53.0]))
    assert x.shape == (3,) and y.shape == (3,)
    assert x[1] == pytest.approx(0.0, abs=1e-9)


def _play():
    return PlayTracking(game_key=1, play_id=2, players=("H97", "V23"),
                        times=np.array([-1.0, 0.0, 1.0]),
                        xy=np.array([[[0.0, 0.0], [10.0, 0.0]],
                                     [[1.0, 0.0], [11.0, 0.0]],
                                     [[2.0, 0.0], [12.0, 0.0]]]),
                        snap_time=1.0)


def test_positions_interpolate_between_samples():
    """Tracking is 10 Hz and video is 59.94 fps, so almost every video frame
    falls between two tracking samples."""
    got = _play().at(0.5)
    assert got[0] == pytest.approx([1.5, 0.0])
    assert got[1] == pytest.approx([11.5, 0.0])


def test_positions_clamp_outside_the_sampled_window():
    play = _play()
    assert play.at(-99.0)[0] == pytest.approx([0.0, 0.0])
    assert play.at(+99.0)[0] == pytest.approx([2.0, 0.0])


def test_video_name_matches_the_dataset_layout():
    assert video_name(57583, 82, "Endzone") == "57583_000082_Endzone.mp4"
    assert video_name(57583, 82, "Sideline") == "57583_000082_Sideline.mp4"
    with pytest.raises(ValueError, match="Endzone or Sideline"):
        video_name(57583, 82, "Broadcast")
