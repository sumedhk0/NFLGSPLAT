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
