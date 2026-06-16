from __future__ import annotations

import pytest

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.meta import PlayMeta, load_meta

_VALID = """\
season: 2024
week: 1
home_team: ATL
away_team: "NO"
fps: 30.0
gsis_play_id: 36
"""


def test_load_meta_valid(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text(_VALID)
    m = load_meta(p)
    assert isinstance(m, PlayMeta)
    assert m.season == "2024"
    assert m.week == 1
    assert m.game_teams == ("ATL", "NO")
    assert m.fps == 30.0
    assert m.gsis_play_id == "36"


def test_load_meta_missing_file(tmp_path):
    with pytest.raises(SetupError, match="meta.yaml"):
        load_meta(tmp_path / "nope.yaml")


def test_load_meta_boolean_team_abbrev(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text("season: 2024\nweek: 1\nhome_team: ATL\naway_team: NO\nfps: 30\n")
    with pytest.raises(SetupError, match="quote"):
        load_meta(p)


def test_load_meta_missing_required(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text("season: 2024\nweek: 1\nhome_team: ATL\nfps: 30\n")
    with pytest.raises(SetupError, match="away_team"):
        load_meta(p)
