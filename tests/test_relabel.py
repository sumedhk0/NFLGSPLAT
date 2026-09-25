"""tracking.relabel: caches carried across a re-pairing by the box they came from."""
import numpy as np
import pandas as pd
import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.tracking.relabel import backup_path, id_map_by_boxes, relabel_keypoints, relabel_pose_cache


def test_backup_path_never_takes_a_name_that_exists(tmp_path):
    """08t applied twice overwrote the first pass's .pre08t -- the only copy before any cut."""
    f = tmp_path / "tracks.parquet"
    f.write_bytes(b"x")
    first = backup_path(f, ".pre08t")
    assert first.name == "tracks.parquet.pre08t"
    first.write_bytes(b"1")
    second = backup_path(f, ".pre08t")
    assert second.name == "tracks.parquet.pre08t.1" and not second.exists()
    second.write_bytes(b"2")
    assert backup_path(f, ".pre08t").name == "tracks.parquet.pre08t.2"
    assert backup_path(tmp_path / "poses.json", ".pre08v").name == "poses.json.pre08v"


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


def test_relabel_keypoint_tables_does_every_table_present(tmp_path):
    """08z / 08za relabelled keypoints_2d.parquet only; the fits read keypoints_2d_ft2.parquet (play 1 from v106), which
    every port patched by hand. Every table present follows the mapping, each backed up; an absent one is skipped."""
    from nfl_gsplat.tracking.relabel import relabel_keypoint_tables

    kdf = pd.DataFrame([("sideline", 548, 80, 0, 1.0, 1.0, 0.9), ("sideline", 548, 80, 1, 2.0, 2.0, 0.9),
                        ("sideline", 474, 204, 0, 3.0, 3.0, 0.9), ("endzone", 540, 15, 0, 4.0, 4.0, 0.9)],
                       columns=["cam", "frame", "global_player_id", "joint", "x", "y", "conf"])
    kdf.to_parquet(tmp_path / "keypoints_2d.parquet", index=False)
    kdf.to_parquet(tmp_path / "keypoints_2d_ft2.parquet", index=False)
    m = {("sideline", 548, 80): 204, ("sideline", 474, 204): -1, ("endzone", 540, 15): 15}
    rep = relabel_keypoint_tables(tmp_path, m, ".pre_fold")
    assert [r[0] for r in rep] == ["keypoints_2d.parquet", "keypoints_2d_ft2.parquet"]
    for name, n_rows, n_drop, n_changed, backup in rep:
        got = pd.read_parquet(tmp_path / name)
        assert (n_rows, n_drop, n_changed) == (3, 1, 2)
        assert sorted(got.global_player_id) == [15, 204, 204]
        assert (tmp_path / backup).exists() and len(pd.read_parquet(tmp_path / backup)) == 4
    (tmp_path / "keypoints_2d_ft2.parquet").unlink()
    assert [r[0] for r in relabel_keypoint_tables(tmp_path, m, ".pre_fold")] == ["keypoints_2d.parquet"]
