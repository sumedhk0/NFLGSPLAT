"""tracking.gap_fill: a track's missing frames filled from detections the threshold rejected."""
import numpy as np
import pandas as pd

from nfl_gsplat.tracking.gap_fill import fill_gaps, predict_box, track_gaps


def rows(cam, pid, frames, x0=100.0):
    return [{"cam": cam, "frame": f, "track_id": pid, "bbox_x1": x0 + f, "bbox_y1": 200.0,
             "bbox_x2": x0 + f + 60, "bbox_y2": 380.0, "conf": 0.9} for f in frames]


def test_only_interior_holes_up_to_the_limit_count():
    assert track_gaps([1, 2, 5, 6]) == [(2, 5)]
    assert track_gaps([1, 2, 60]) == []                       # too long: the track ended
    assert track_gaps([1, 2, 3]) == []


def test_the_predicted_box_walks_across_the_gap():
    a, b = np.array([0.0, 0, 10, 20]), np.array([10.0, 0, 20, 20])
    assert np.allclose(predict_box(a, b, 5, 0, 10), [5, 0, 15, 20])


def test_a_missed_detection_fills_its_own_track_and_only_once():
    df = pd.DataFrame(rows("sideline", 1, [10, 11, 14, 15]) + rows("sideline", 2, [10, 11, 14, 15], x0=900.0))
    # the low-confidence pool holds the two men on frames 12 and 13, in the right places
    pool = {12: np.array([[112.0, 200, 172, 380], [912.0, 200, 972, 380]]),
            13: np.array([[113.0, 200, 173, 380], [913.0, 200, 973, 380]])}
    add, counts = fill_gaps(df, pool, cam="sideline")
    assert counts == {1: 2, 2: 2}
    assert all(r["filled"] for r in add)
    assert {(r["frame"], r["track_id"]) for r in add} == {(12, 1), (13, 1), (12, 2), (13, 2)}
    # one detection cannot serve two tracks
    pool_one = {12: np.array([[112.0, 200, 172, 380]])}
    add, counts = fill_gaps(df, pool_one, cam="sideline")
    assert counts == {1: 1}


def test_a_detection_somewhere_else_is_not_taken():
    df = pd.DataFrame(rows("sideline", 1, [10, 11, 14, 15]))
    pool = {12: np.array([[800.0, 600, 860, 780]]), 13: np.array([[800.0, 600, 860, 780]])}
    add, counts = fill_gaps(df, pool, cam="sideline")
    assert add == [] and counts == {}
