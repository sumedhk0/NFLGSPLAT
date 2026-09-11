"""split_by_kit: a track that changes kit for good is two players; a blip is not."""
import numpy as np
import pandas as pd

from nfl_gsplat.tracking.split_by_kit import cut_points, split_tracks_by_kit


def test_cut_points_find_a_sustained_handover_and_ignore_a_blip():
    red, white = 0.7, -0.7
    m = np.array([red] * 40 + [white] * 40)                            # a handover at 40
    assert cut_points(m) == [40]
    m = np.array([red] * 40 + [white] * 6 + [red] * 40)                # a six-detection blip: shadow
    assert cut_points(m) == []
    m = np.array([red] * 40 + [np.nan] * 10 + [white] * 40)            # unknown kits between: cut at the first white
    assert cut_points(m) == [50]
    m = np.array([red, white] * 40)                                    # never settles: no cut
    assert cut_points(m) == []
    m = np.array([red] * 40 + [white] * 40 + [red] * 40)               # two handovers
    assert cut_points(m) == [40, 80]


def test_split_tracks_by_kit_reassigns_the_tail():
    rows = []
    for f in range(80):
        rows.append({"cam": "sideline", "track_id": 3, "frame": f, "kit_margin": 0.7 if f < 40 else -0.7})
        rows.append({"cam": "sideline", "track_id": 5, "frame": f, "kit_margin": 0.7})
        rows.append({"cam": "endzone", "track_id": 3, "frame": f, "kit_margin": -0.7})
    df = pd.DataFrame(rows)
    df["global_player_id"] = df["track_id"]
    out, cuts = split_tracks_by_kit(df)
    assert cuts == [("sideline", 3, 40, 6)]
    s3 = out[(out.cam == "sideline") & (out.frame >= 40) & (out.kit_margin < 0)]
    assert set(s3.track_id) == {6} and set(s3.global_player_id) == {6}
    assert set(out[(out.cam == "sideline") & (out.frame < 40) & (out.track_id != 5)].track_id) == {3}
    assert set(out[out.cam == "endzone"].track_id) == {3}               # the endzone track 3 is untouched
    assert set(out[out.track_id == 5].track_id) == {5}
