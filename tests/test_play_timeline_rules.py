"""render.edge_rule.edge_clipped_ids: one-view ids whose boxes all touch a frame edge."""
import pandas as pd

from nfl_gsplat.render.edge_rule import EDGE_PX, edge_clipped_ids


class _Track:
    height = 1080
    width = 1920


def _rows(pid, cam, boxes):
    return [{"frame": i, "cam": cam, "track_id": pid, "global_player_id": pid,
             "bbox_x1": b[0], "bbox_y1": b[1], "bbox_x2": b[2], "bbox_y2": b[3]}
            for i, b in enumerate(boxes)]


def test_top_clipped_endzone_only_id_is_dropped_two_view_and_interior_are_kept():
    rows = []
    rows += _rows(1, "endzone", [(100, 0, 140, 70), (120, 3, 160, 75)])          # clipped at the top
    rows += _rows(2, "endzone", [(100, 200, 140, 350), (120, 0, 160, 75)])       # one interior box: kept
    rows += _rows(3, "endzone", [(100, 0, 140, 70)]) + _rows(3, "sideline", [(500, 400, 560, 560)])
    rows += _rows(4, "sideline", [(500, 1000, 560, 1080 - 2)])                   # clipped at the bottom
    df = pd.DataFrame(rows)
    views = {0: {3: ("endzone", "sideline")}}
    tracks = {"endzone": _Track(), "sideline": _Track()}
    out = edge_clipped_ids(df, tracks, views)
    assert out == {1, 4}
    assert EDGE_PX >= 1.0
