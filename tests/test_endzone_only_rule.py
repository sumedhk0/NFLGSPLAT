"""endzone_only_rule: endzone-only ids leave the timeline; two-view and sideline ids stay."""
import pandas as pd

from nfl_gsplat.render.endzone_only_rule import endzone_only_ids


def _df(rows):
    return pd.DataFrame(rows, columns=["frame", "cam", "global_player_id"])


def test_endzone_only_ids_are_dropped_and_two_view_ids_kept():
    df = _df([
        (0, "sideline", 1), (1, "sideline", 1),                       # sideline-only: kept
        (0, "endzone", 2), (1, "endzone", 2),                         # endzone-only: dropped
        (0, "sideline", 3), (0, "endzone", 3), (1, "endzone", 3),     # two-view once: kept
        (0, "endzone", -1),                                           # unlinked: ignored
    ])
    views = {0: {1: ["sideline"], 2: ["endzone"], 3: ["sideline", "endzone"]},
             1: {1: ["sideline"], 2: ["endzone"], 3: ["endzone"]}}
    assert endzone_only_ids(df, views) == {2}


def test_the_camera_name_is_a_parameter():
    df = _df([(0, "sideline", 5), (0, "endzone", 6)])
    views = {0: {5: ["sideline"], 6: ["endzone"]}}
    assert endzone_only_ids(df, views, cam="sideline") == {5}
