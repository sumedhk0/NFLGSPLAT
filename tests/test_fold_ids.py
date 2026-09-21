import pandas as pd

from nfl_gsplat.tracking.fold import fold_ids, keypoint_map
from nfl_gsplat.tracking.relabel import relabel_keypoints


def _tracks():
    rows = [
        # cam, frame, track_id, gid, conf
        ("sideline", 1, 100, 74, 0.9),
        ("sideline", 2, 100, 74, 0.9),
        ("sideline", 2, 200, 48, 0.5),   # twin of 74 on frame 2: the weaker box goes
        ("sideline", 3, 200, 48, 0.8),
        ("sideline", 4, 300, 75, 0.7),
        ("sideline", 4, 400, 9, 0.6),    # another man, same frame: untouched
        ("endzone", 3, 500, 48, 0.4),
    ]
    return pd.DataFrame(rows, columns=["cam", "frame", "track_id", "global_player_id", "conf"])


def test_fold_relabels_and_drops_the_weaker_twin_only():
    df = _tracks()
    out, n = fold_ids(df, 74, [48, 75])
    assert n == 1
    got = out[out.global_player_id == 74].sort_values(["cam", "frame"])
    assert list(zip(got.cam, got.frame, got.track_id)) == [("endzone", 3, 500), ("sideline", 1, 100), ("sideline", 2, 100),
                                                          ("sideline", 3, 200), ("sideline", 4, 300)]
    assert set(out.global_player_id) == {74, 9}
    # track ids are never rewritten (08v joins the caches on them)
    assert sorted(out.track_id) == sorted(df.track_id.drop(index=2))


def test_keypoints_follow_the_rows():
    df = _tracks()
    out, _ = fold_ids(df, 74, [48, 75])
    m = keypoint_map(df, out)
    assert m[("sideline", 3, 48)] == 74 and m[("sideline", 4, 75)] == 74 and m[("sideline", 4, 9)] == 9
    assert m[("sideline", 2, 48)] == -1
    kdf = pd.DataFrame([("sideline", 2, 48, 0, 1.0, 1.0, 0.5), ("sideline", 3, 48, 0, 1.0, 1.0, 0.5),
                        ("sideline", 4, 9, 0, 1.0, 1.0, 0.5)],
                       columns=["cam", "frame", "global_player_id", "joint", "x", "y", "conf"])
    kout, kdrop = relabel_keypoints(kdf, m)
    assert kdrop == 1 and list(kout.global_player_id) == [74, 9]


def test_fold_only_a_frame_range_of_the_dropped_id():
    df = _tracks()
    df = pd.concat([df, pd.DataFrame([("endzone", 1, 600, 75, 0.9), ("endzone", 4, 600, 75, 0.9)],
                                     columns=df.columns)], ignore_index=True)
    out, n = fold_ids(df, 74, [75], frames=(4, 9))
    assert n == 0
    got = out[out.global_player_id == 74]
    assert ("endzone", 4) in set(zip(got.cam, got.frame)) and ("sideline", 4) in set(zip(got.cam, got.frame))
    left = out[out.global_player_id == 75]
    assert list(zip(left.cam, left.frame)) == [("endzone", 1)]              # the row before the range keeps its id


def test_fold_ids_one_camera_track_only():
    import pandas as pd
    from nfl_gsplat.tracking.fold import fold_ids

    rows = []
    for f in range(500, 510):
        rows.append(dict(frame=f, cam="sideline", track_id=25, global_player_id=37, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=20, conf=1.0))
        rows.append(dict(frame=f, cam="endzone", track_id=25, global_player_id=37, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=20, conf=1.0))   # the other man's
        rows.append(dict(frame=f, cam="endzone", track_id=37, global_player_id=37, bbox_x1=50, bbox_y1=0, bbox_x2=60, bbox_y2=20, conf=1.0))  # his own
        rows.append(dict(frame=f, cam="sideline", track_id=9, global_player_id=157, bbox_x1=80, bbox_y1=0, bbox_x2=90, bbox_y2=20, conf=1.0))
    df = pd.DataFrame(rows)
    out, n = fold_ids(df, 157, [37], cam="endzone", track_id=25, frames=(500, 509))
    ez25 = out[(out.cam == "endzone") & (out.track_id == 25)]
    assert set(ez25.global_player_id) == {157} and len(ez25) == 10                        # that camera track moved
    assert set(out[(out.cam == "endzone") & (out.track_id == 37)].global_player_id) == {37}   # his own endzone rows stayed
    assert set(out[(out.cam == "sideline") & (out.track_id == 25)].global_player_id) == {37}  # the sideline untouched
    assert n == 0


def test_drop_rows_one_camera_frames():
    import pandas as pd
    from nfl_gsplat.tracking.fold import drop_rows, keypoint_map
    from nfl_gsplat.tracking.relabel import relabel_keypoints

    rows = []
    for f in range(440, 470):
        rows.append(dict(frame=f, cam="endzone", track_id=45, global_player_id=4, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=20, conf=1.0))
    rows.append(dict(frame=443, cam="sideline", track_id=4, global_player_id=4, bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=20, conf=1.0))
    rows.append(dict(frame=465, cam="sideline", track_id=4, global_player_id=4, bbox_x1=50, bbox_y1=0, bbox_x2=60, bbox_y2=20, conf=1.0))  # the stray
    rows.append(dict(frame=465, cam="sideline", track_id=9, global_player_id=40, bbox_x1=80, bbox_y1=0, bbox_x2=90, bbox_y2=20, conf=1.0))
    df = pd.DataFrame(rows)
    out, n = drop_rows(df, 4, cam="sideline", frames=(460, 470))
    assert n == 1 and len(out) == len(df) - 1
    assert set(out[(out.cam == "sideline") & (out.global_player_id == 4)].frame) == {443}      # his real box stays
    assert len(out[(out.cam == "endzone") & (out.global_player_id == 4)]) == 30                # the other camera untouched
    assert set(out[out.frame == 465].global_player_id) == {4, 40}                                # 4's endzone row on 465 and 40 stay
    kdf = pd.DataFrame([dict(cam="sideline", frame=465, global_player_id=4, joint=0, x=1.0, y=1.0, conf=1.0),
                        dict(cam="sideline", frame=443, global_player_id=4, joint=0, x=1.0, y=1.0, conf=1.0)])
    kout, kdrop = relabel_keypoints(kdf, keypoint_map(df, out))
    assert kdrop == 1 and set(kout.frame) == {443}                                               # the keypoints follow
