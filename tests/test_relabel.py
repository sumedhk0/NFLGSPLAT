"""tracking.relabel: caches carried across a re-pairing by the box they came from."""
import numpy as np
import pandas as pd
import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.tracking.relabel import id_map_by_boxes, relabel_keypoints, relabel_pose_cache


def tracks(ids):
    rows = []
    for (cam, f, tid), (x1, y1) in zip([("sideline", 0, ids[0]), ("sideline", 1, ids[1]), ("endzone", 0, ids[2])],
                                       [(10.0, 20.0), (12.0, 21.0), (300.0, 400.0)]):
        rows.append({"cam": cam, "frame": f, "track_id": tid, "bbox_x1": x1, "bbox_y1": y1,
                     "bbox_x2": x1 + 40.0, "bbox_y2": y1 + 100.0})
    return pd.DataFrame(rows)


def test_id_map_follows_the_boxes():
    old = tracks([3, 3, 7])
    new = tracks([1, 1, 1])                          # the endzone track paired with the sideline's
    m = id_map_by_boxes(old, new)
    assert m == {("sideline", 0, 3): 1, ("sideline", 1, 3): 1, ("endzone", 0, 7): 1}


def test_id_map_refuses_a_different_table():
    old = tracks([3, 3, 7])
    new = tracks([1, 1, 1])
    new.loc[2, "bbox_x1"] += 5.0                      # a moved box is another tracks table
    with pytest.raises(SetupError):
        id_map_by_boxes(old, new)


def test_keypoints_and_pose_cache_relabelled():
    old = tracks([3, 3, 7])
    new = tracks([1, 1, 1])
    m = id_map_by_boxes(old, new)
    kdf = pd.DataFrame({"frame": [0, 0, 1, 0], "cam": ["sideline", "endzone", "sideline", "endzone"],
                        "global_player_id": [3, 7, 3, 99], "joint": [0, 0, 0, 0],
                        "x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0], "conf": [0.9] * 4})
    out, dropped = relabel_keypoints(kdf, m)
    assert dropped == 1 and out["global_player_id"].tolist() == [1, 1, 1]
    blob = {"cam": "sideline", "frames": {0: {3: {"betas": np.zeros(10)}, 5: {"betas": np.ones(10)}},
                                          1: {3: {"betas": np.zeros(10)}}}}
    out, dropped = relabel_pose_cache(blob, m, "sideline")
    assert dropped == 1
    assert sorted(out["frames"][0]) == [1] and sorted(out["frames"][1]) == [1]
    assert blob["frames"][0][3] is out["frames"][0][1]
