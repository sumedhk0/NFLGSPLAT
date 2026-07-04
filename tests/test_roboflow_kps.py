import json

import pytest

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
from nfl_gsplat.calibration.roboflow_kps import (
    load_kps_json, to_nfl_name, write_kps_json, yard_base,
)
from nfl_gsplat.errors import SetupError


# The full mapping table, both territories. Model classes come from
# football-field-key-points-mvmjf/2 (yard + optional top/bottom + hash/sl).
CASES = [
    # (model_class, territory, expected NFL name)
    ("30",             "away", "away_30_left_number"),   # bare = top number
    ("30-bottom",      "away", "away_30_right_number"),
    ("30-top-hash",    "away", "away_30_left_hash"),
    ("30-bottom-hash", "away", "away_30_right_hash"),
    ("20",             "home", "home_20_left_number"),
    ("50",             "home", "mid_50_left_number"),    # 50 is territory-less
    ("50-top-hash",    "away", "mid_50_left_hash"),
]


@pytest.mark.parametrize("cls,terr,expected", CASES)
def test_to_nfl_name_table(cls, terr, expected):
    name = to_nfl_name(cls, terr)
    assert name == expected
    assert name in NFL_LANDMARKS          # every mapped name must exist


@pytest.mark.parametrize("cls", ["30-top-sl", "30-bottom-sl"])
def test_sidelines_always_unmappable(cls):
    assert to_nfl_name(cls, "away") is None


@pytest.mark.parametrize("cls", ["FG-POST", "goalline", "endzone-corner", "abc"])
def test_unknown_classes_unmappable(cls):
    assert to_nfl_name(cls, "home") is None


def test_yard_base():
    assert yard_base("30-top-hash", "away") == "away_30"
    assert yard_base("50", "away") == "mid_50"
    assert yard_base("FG-POST", "away") is None


def test_json_round_trip(tmp_path):
    frames = {0: [("30", 671.0, 242.0, 0.86)], 7: []}
    p = tmp_path / "roboflow_kps.json"
    write_kps_json(p, model_id="m/2", video_name="v.mp4", num_frames=10,
                   kp_conf=0.5, frames=frames)
    loaded = load_kps_json(p, expect_num_frames=10)
    assert loaded[0] == [("30", 671.0, 242.0, 0.86)]
    assert loaded[7] == []
    assert 3 not in loaded                 # absent frame stays absent


def test_load_missing_file_fails_loud(tmp_path):
    with pytest.raises(SetupError, match="03_roboflow_precompute"):
        load_kps_json(tmp_path / "nope.json")


def test_load_frame_count_mismatch_fails_loud(tmp_path):
    p = tmp_path / "roboflow_kps.json"
    write_kps_json(p, model_id="m/2", video_name="v.mp4", num_frames=10,
                   kp_conf=0.5, frames={})
    with pytest.raises(SetupError, match="stale"):
        load_kps_json(p, expect_num_frames=999)


@pytest.mark.parametrize("cls", ["35", "35-bottom", "5", "45-bottom", "15", "25"])
def test_number_classes_at_non10_yards_unmappable(cls):
    # NFL fields paint numbers only at 10/20/30/40 (+50): NFL_LANDMARKS has no
    # *_number entry for other lines, so these must map to None, not a bad name
    assert to_nfl_name(cls, "away") is None


def test_hash_classes_valid_at_every_5yd_line():
    assert to_nfl_name("35-top-hash", "away") == "away_35_left_hash"
    assert to_nfl_name("45-bottom-hash", "home") == "home_45_right_hash"
