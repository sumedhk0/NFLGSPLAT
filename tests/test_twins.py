"""tracking.twins: two ids on one body are found by their feet, not their boxes."""
import numpy as np
import pandas as pd

from nfl_gsplat.tracking.twins import apply_merge, merge_map, twin_pairs


def tracks(rows):
    return pd.DataFrame(rows, columns=["cam", "frame", "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "conf"])


def scene():
    """Two ids on one body (1 and 2), a third man a metre behind in the camera's depth (3) whose box
    overlaps just as much, and a fourth id of the other team on the same spot (4)."""
    rows = []
    ank = {}
    teams = {1: "KC", 2: "KC", 3: "KC", 4: "BAL"}
    for f in range(100):
        rows.append(["sideline", f, 1, 100, 100, 160, 280, 0.9])
        rows.append(["sideline", f, 2, 104, 96, 164, 276, 0.8])
        rows.append(["sideline", f, 3, 106, 104, 166, 284, 0.9])
        rows.append(["sideline", f, 4, 101, 101, 161, 281, 0.9])
        ank[("sideline", f, 1)] = np.array([-20.0, 3.0])
        if f % 3 == 0:                                   # the twin carries ankles on a third of the frames
            ank[("sideline", f, 2)] = np.array([-20.05, 3.02])
        ank[("sideline", f, 3)] = np.array([-21.2, 3.1])  # one metre away on the turf
        ank[("sideline", f, 4)] = np.array([-20.0, 3.0])
    return tracks(rows), ank, teams


def test_only_the_same_body_and_the_same_team_merge():
    df, ank, teams = scene()
    pairs = twin_pairs(df, ank, teams, cam="sideline")
    assert [(k, d) for k, d, *_ in pairs] == [(1, 2)]      # 3 is a metre away, 4 is the other team


def test_the_merged_id_holds_one_box_a_frame():
    df, ank, teams = scene()
    pairs = twin_pairs(df, ank, teams, cam="sideline")
    mapping, dropped = merge_map(df, pairs)
    assert dropped == 100
    out, n = apply_merge(df, mapping)
    assert n == 100
    g = out[(out["cam"] == "sideline") & (out["track_id"] == 1)]
    assert len(g) == 100 and g["frame"].is_unique
    # the surviving box is the more confident one (id 1's, conf 0.9 against 0.8)
    assert set(g["bbox_x1"]) == {100}
    assert 2 not in set(out["track_id"])


def test_a_pair_without_enough_ankle_frames_is_left_alone():
    df, ank, teams = scene()
    sparse = {k: v for k, v in ank.items() if k[2] != 2 or k[1] < 9}   # id 2 has ankles on three frames
    assert twin_pairs(df, sparse, teams, cam="sideline") == []
