"""ground_positions must key by the PLAYER id, not the tracker's id.

The two have always been equal in this pipeline, so the confusion was invisible until 08s relabelled an
endzone track onto another player's global id (measured 2026-09-12: all four relabelled ids reported
zero endzone ground frames while their track rows and keypoint boxes were intact). ankle_ground keys by
global_player_id, ground_positions looked the ankles up by track_id and emitted its output under
track_id, so a row whose two ids differ lands under a key no caller asks for -- and 05p, the depth snap
and the census all ask by global_player_id.
"""
import numpy as np
import pandas as pd

from nfl_gsplat.render.play_timeline import ground_positions


class _Intr:
    def K(self):
        return np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])


class _Pose:
    R = np.eye(3)
    t = np.array([0.0, 0.0, 50.0])


class _Track:
    """One solved frame, looking straight down the z axis from 50 m."""

    conf = np.ones(4)

    def at(self, f):
        return _Intr(), _Pose()


def _rows():
    return pd.DataFrame([
        {"cam": "endzone", "frame": 1, "track_id": 19, "global_player_id": 11,
         "bbox_x1": 900.0, "bbox_y1": 400.0, "bbox_x2": 1000.0, "bbox_y2": 600.0},
        {"cam": "endzone", "frame": 1, "track_id": 5, "global_player_id": 5,
         "bbox_x1": 100.0, "bbox_y1": 400.0, "bbox_x2": 200.0, "bbox_y2": 600.0},
    ])


def test_the_key_is_the_player_not_the_track():
    ground = ground_positions(_rows(), {"endzone": _Track()})
    assert set(ground[1]) == {11, 5}, "a relabelled row must land under its global_player_id"


def test_a_table_without_the_player_id_is_refused_loudly():
    from nfl_gsplat.errors import SetupError

    rows = _rows().drop(columns=["global_player_id"])
    try:
        ground_positions(rows, {"endzone": _Track()})
    except SetupError as e:
        assert "global_player_id" in str(e)
    else:
        raise AssertionError("a table with no global_player_id must be refused, not keyed by track_id")


def test_the_ankle_point_is_found_by_the_player_id():
    # ankle_ground keys (cam, frame, global_player_id); the row's track_id is deliberately different
    ankles = {("endzone", 1, 11): np.array([12.0, -3.0]), ("endzone", 1, 5): np.array([-7.0, 2.0])}
    ground = ground_positions(_rows(), {"endzone": _Track()}, ankles=ankles)
    assert np.allclose(ground[1][11], [12.0, -3.0]), "the ankle point keyed by the player id was missed"
    assert np.allclose(ground[1][5], [-7.0, 2.0])
