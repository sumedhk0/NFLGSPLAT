import numpy as np
import pandas as pd


def _rows(recs):
    from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
    df = pd.DataFrame(recs)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c not in ("cam",) else ""
    return df


def test_assign_identity_columns_joins_across_cameras():
    from nfl_gsplat.calibration.identity_precompute import assign_identity_columns
    # two teams by color; same jersey/team must yield the SAME uid in both cams
    red = np.full((20, 20, 3), (0, 0, 200), np.uint8)      # BGR red
    blue = np.full((20, 20, 3), (200, 0, 0), np.uint8)     # BGR blue
    color = {("sideline", 58): red, ("endzone", 58): red,
             ("sideline", 20): blue, ("endzone", 20): blue}
    recs = []
    for cam in ("sideline", "endzone"):
        for tid, jersey in ((58, 58), (20, 20)):
            recs.append({"frame": 0, "cam": cam, "track_id": tid,
                         "jersey_number_ocr": jersey})
    df = _rows(recs)

    def crop_provider(cam, frame, track_id):
        return color[(cam, int(track_id))]

    out = assign_identity_columns(df, crop_provider, season=2025)
    uid = {(r.cam, r.track_id): r.player_uid for r in out.itertuples()}
    # #58 same uid across cameras; #20 same across cameras; the two differ
    assert uid[("sideline", 58)] == uid[("endzone", 58)]
    assert uid[("sideline", 20)] == uid[("endzone", 20)]
    assert uid[("sideline", 58)] != uid[("sideline", 20)]
