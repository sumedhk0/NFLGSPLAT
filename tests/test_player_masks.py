from unittest.mock import patch

import pandas as pd
import pytest

from nfl_gsplat.calibration.player_masks import boxes_provider_from_tracks
from nfl_gsplat.errors import SetupError
from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS


def _make_test_dataframe():
    """Build test tracks dataframe."""
    rows = [
        {"frame": 0, "cam": "sideline", "bbox_x1": 10.0, "bbox_y1": 20.0,
         "bbox_x2": 30.0, "bbox_y2": 60.0},
        {"frame": 0, "cam": "endzone", "bbox_x1": 1.0, "bbox_y1": 2.0,
         "bbox_x2": 3.0, "bbox_y2": 4.0},
        {"frame": 2, "cam": "sideline", "bbox_x1": 5.0, "bbox_y1": 6.0,
         "bbox_x2": 7.0, "bbox_y2": 8.0},
    ]
    df = pd.DataFrame(rows)
    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = -1 if c != "cam" else ""
    return df


def _tracks(tmp_path):
    """Create parquet file with test tracks; use pickle to avoid pyarrow issues."""
    df = _make_test_dataframe()
    p = tmp_path / "tracks.parquet"
    df.to_pickle(p)
    return p


def test_provider_yields_boxes_per_cam_and_frame(tmp_path):
    """Test that boxes are retrieved correctly per camera and frame."""
    p = _tracks(tmp_path)
    with patch("pandas.read_parquet", pd.read_pickle):
        prov = boxes_provider_from_tracks(p)
    sl = prov("sideline")
    assert sl(0) == [(10.0, 20.0, 30.0, 60.0)]
    assert sl(2) == [(5.0, 6.0, 7.0, 8.0)]
    assert sl(1) == []                                  # no detection that frame
    ez = prov("endzone")
    assert ez(0) == [(1.0, 2.0, 3.0, 4.0)]
    assert ez(9) == []


def test_provider_unknown_cam_empty(tmp_path):
    """Test that unknown cameras return empty boxes."""
    p = _tracks(tmp_path)
    with patch("pandas.read_parquet", pd.read_pickle):
        prov = boxes_provider_from_tracks(p)
    assert prov("skycam")(0) == []


def test_provider_missing_file_fails_loud(tmp_path):
    with pytest.raises(SetupError, match="tracks"):
        boxes_provider_from_tracks(tmp_path / "nope.parquet")
