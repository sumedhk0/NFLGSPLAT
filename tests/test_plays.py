"""plays.yaml manifest loader (T1.1)."""
from __future__ import annotations

import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.plays import load_plays

_GOOD = """
meta:
  season: 2024
  home_team: ATL
  away_team: "NO"
  fps: 30.0
plays:
  play_001: {start_frame: 1200, end_frame: 1380, gsis_play_id: 36}
  play_002: {start_frame: 1500, end_frame: 1700}
"""


def _write(tmp_path, text):
    p = tmp_path / "plays.yaml"
    p.write_text(text)
    return p


def test_load_good_manifest(tmp_path):
    m = load_plays(_write(tmp_path, _GOOD))
    assert m.season == "2024"
    assert m.game_teams == ("ATL", "NO")
    assert m.fps == 30.0
    assert m.play_ids() == ["play_001", "play_002"]
    w = m.window("play_001")
    assert w.start_frame == 1200 and w.end_frame == 1380
    assert w.num_frames == 181
    assert w.gsis_play_id == "36"
    assert m.window("play_002").gsis_play_id is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(SetupError, match="plays manifest missing"):
        load_plays(tmp_path / "nope.yaml")


def test_missing_meta_field_raises(tmp_path):
    bad = "meta:\n  season: 2024\n  home_team: ATL\nplays:\n  p1: {start_frame: 0, end_frame: 5}\n"
    with pytest.raises(SetupError, match="away_team is required"):
        load_plays(_write(tmp_path, bad))


def test_inverted_window_raises(tmp_path):
    bad = _GOOD.replace("start_frame: 1200, end_frame: 1380", "start_frame: 1380, end_frame: 1200")
    with pytest.raises(SetupError, match="end_frame .* < start_frame"):
        load_plays(_write(tmp_path, bad))


def test_unquoted_team_abbrev_raises(tmp_path):
    # Bare NO → YAML boolean false; loader should reject with a quoting hint.
    bad = "meta: {season: 2024, home_team: ATL, away_team: NO}\nplays:\n  p1: {start_frame: 0, end_frame: 5}\n"
    with pytest.raises(SetupError, match="parsed as a boolean"):
        load_plays(_write(tmp_path, bad))


def test_play_missing_frames_raises(tmp_path):
    bad = "meta: {season: 2024, home_team: A, away_team: B}\nplays:\n  p1: {start_frame: 0}\n"
    with pytest.raises(SetupError, match="needs start_frame and end_frame"):
        load_plays(_write(tmp_path, bad))
