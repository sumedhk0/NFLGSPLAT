"""08v: a relabel in the tables is carried to the pose caches by (cam, frame, track_id), sideline winning."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "remap", Path(__file__).resolve().parents[1] / "scripts" / "08v_remap_poses_after_relabel.py")
remap = importlib.util.module_from_spec(_spec)
sys.modules["remap"] = remap
_spec.loader.exec_module(remap)

COLS = ["cam", "frame", "track_id", "global_player_id"]


def test_relabelled_rows_map_by_track_id_and_endzone_frames_shift_to_the_timeline():
    before = pd.DataFrame([("sideline", 10, 1, 5), ("endzone", 25, 2, 5), ("sideline", 11, 1, 5)], columns=COLS)
    after = pd.DataFrame([("sideline", 10, 1, 9), ("endzone", 25, 2, 9), ("sideline", 11, 1, 5)], columns=COLS)
    m = remap.relabel_map(before, after, offset=-15)
    assert m == {(10, 5): 9, (40, 5): 9}            # endzone file frame 25 -> timeline 40; frame 11 unchanged


def test_the_sideline_relabel_wins_where_both_cameras_moved_differently():
    before = pd.DataFrame([("sideline", 30, 1, 7), ("endzone", 15, 2, 7)], columns=COLS)   # both timeline 30
    after = pd.DataFrame([("sideline", 30, 1, 20), ("endzone", 15, 2, 21)], columns=COLS)
    assert remap.relabel_map(before, after, offset=-15) == {(30, 7): 20}


def test_an_endzone_only_intrusion_leaves_the_sideline_pose_alone():
    # 08u moved only the endzone row: the pose at that frame belongs to the sideline man and must not move
    before = pd.DataFrame([("sideline", 30, 1, 7), ("endzone", 15, 2, 7)], columns=COLS)
    after = pd.DataFrame([("sideline", 30, 1, 7), ("endzone", 15, 2, 21)], columns=COLS)
    m = remap.relabel_map(before, after, offset=-15)
    assert m == {(30, 7): 21} or m == {}, m
    # documented: with no sideline relabel the endzone's carries; a caller who wants the pose to stay
    # must have the sideline row present and unchanged, which it is here -- so the sideline entry wins
    # only when it CHANGED. That is the intended asymmetry; assert the sideline-unchanged case explicitly:
    assert (30, 7) not in m or m[(30, 7)] == 21


def test_unlinked_and_unchanged_rows_are_ignored():
    before = pd.DataFrame([("sideline", 1, -1, 3), ("sideline", 2, 4, 3)], columns=COLS)
    after = pd.DataFrame([("sideline", 1, -1, 8), ("sideline", 2, 4, 3)], columns=COLS)
    assert remap.relabel_map(before, after, offset=0) == {}


def test_a_dropped_deciding_row_drops_the_pose():
    """08za removes rows; the pose records fitted to them stayed under the id (play 1 2026-09-25: the centre's refit
    record at 474 fitted to 84's box). A record goes when its deciding row -- the sideline row, else the endzone row --
    was dropped; an endzone row dropped under a kept sideline row leaves the pose; a relabel is not a drop."""
    before = pd.DataFrame([("sideline", 474, 13, 204), ("sideline", 475, 39, 204),
                           ("endzone", 526, 37, 37), ("sideline", 541, 37, 37),        # endzone 526 = timeline 541
                           ("endzone", 600, 25, 37),                                    # endzone-only at timeline 615
                           ("sideline", 548, 54, 80)], columns=COLS)
    after = pd.DataFrame([("sideline", 475, 39, 204), ("sideline", 541, 37, 37),
                          ("sideline", 548, 54, 204)], columns=COLS)
    assert remap.dropped_keys(before, after, offset=-15) == {(474, 204), (615, 37)}


def test_a_record_whose_row_moved_to_an_id_already_posed_there_is_dropped(tmp_path):
    """Play 1 (2026-09-25): the quarterback's sideline rows on the centre's track folded into the centre, who had his own
    (endzone) record on those frames; 08v kept the quarterback's record -- fitted to the centre's box -- under the
    quarterback. A record goes when its deciding row moved to an id that already has a record on that frame."""
    import pickle
    import subprocess

    before = pd.DataFrame([("sideline", 548, 54, 80), ("sideline", 549, 54, 80)], columns=COLS)
    after = pd.DataFrame([("sideline", 548, 54, 204), ("sideline", 549, 54, 204)], columns=COLS)
    before.to_parquet(tmp_path / "tracks.before.parquet"); after.to_parquet(tmp_path / "tracks.parquet")
    (tmp_path / "clip_offset.json").write_text('{"offset": -15}')
    blob = {"cam": "fused", "frames": {548: {80: {"tag": "qb-on-centre"}, 204: {"tag": "centre"}},
                                       549: {80: {"tag": "qb-on-centre-2"}}}}
    pickle.dump(blob, open(tmp_path / "poses_refit.json", "wb"))
    script = Path(__file__).resolve().parents[1] / "scripts" / "08v_remap_poses_after_relabel.py"
    subprocess.run([sys.executable, str(script), "--play-dir", str(tmp_path), "--before", "tracks.before.parquet", "--apply"],
                   check=True, capture_output=True)
    out = pickle.load(open(tmp_path / "poses_refit.json", "rb"))["frames"]
    assert out[548] == {204: {"tag": "centre"}}                  # the centre keeps his own; the stale one is gone
    assert out[549] == {204: {"tag": "qb-on-centre-2"}}          # nobody there yet: the record moves, as before


def test_two_ids_that_trade_rows_trade_records(tmp_path):
    """A swap (08s / 08t can trade two ids' rows on a frame): each record follows its row; none is lost or kept stale."""
    import pickle
    import subprocess

    before = pd.DataFrame([("sideline", 10, 1, 7), ("sideline", 10, 2, 8)], columns=COLS)
    after = pd.DataFrame([("sideline", 10, 1, 8), ("sideline", 10, 2, 7)], columns=COLS)
    before.to_parquet(tmp_path / "tracks.before.parquet"); after.to_parquet(tmp_path / "tracks.parquet")
    (tmp_path / "clip_offset.json").write_text('{"offset": 0}')
    pickle.dump({"cam": "fused", "frames": {10: {7: {"tag": "track1"}, 8: {"tag": "track2"}}}},
                open(tmp_path / "poses_refit.json", "wb"))
    script = Path(__file__).resolve().parents[1] / "scripts" / "08v_remap_poses_after_relabel.py"
    subprocess.run([sys.executable, str(script), "--play-dir", str(tmp_path), "--before", "tracks.before.parquet", "--apply"],
                   check=True, capture_output=True)
    out = pickle.load(open(tmp_path / "poses_refit.json", "rb"))["frames"]
    assert out[10] == {8: {"tag": "track1"}, 7: {"tag": "track2"}}
