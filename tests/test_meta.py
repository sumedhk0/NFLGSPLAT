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


def test_load_meta_parses_calib_hints(tmp_path):
    from nfl_gsplat.utils.meta import load_meta
    p = tmp_path / "meta.yaml"
    p.write_text(
        'season: 2025\nweek: 4\nhome_team: AZ\naway_team: "SEA"\nfps: 59.94\n'
        "calib_hints:\n"
        "  sideline: {ref_frame: 0, ref_x: 866, yard: 30, side: away, increasing: right}\n"
        "  endzone:  {ref_frame: 5, ref_x: 540, yard: 50, side: mid, increasing: left}\n"
    )
    m = load_meta(p)
    assert set(m.calib_hints) == {"sideline", "endzone"}
    h = m.calib_hints["sideline"]
    assert (h.ref_frame, h.ref_x, h.yard, h.side, h.increasing) == (0, 866.0, 30, "away", "right")
    assert m.calib_hints["endzone"].side == "mid"


def test_calib_hints_default_empty(tmp_path):
    from nfl_gsplat.utils.meta import load_meta
    p = tmp_path / "meta.yaml"
    p.write_text('season: 2025\nweek: 4\nhome_team: AZ\naway_team: "SEA"\nfps: 30\n')
    assert load_meta(p).calib_hints == {}


def test_calib_hint_bad_side_raises(tmp_path):
    import pytest
    from nfl_gsplat.errors import SetupError
    from nfl_gsplat.utils.meta import load_meta
    p = tmp_path / "meta.yaml"
    p.write_text(
        'season: 2025\nweek: 4\nhome_team: AZ\naway_team: "SEA"\nfps: 30\n'
        "calib_hints:\n  sideline: {ref_frame: 0, ref_x: 5, yard: 30, side: nope, increasing: right}\n"
    )
    with pytest.raises(SetupError, match="side"):
        load_meta(p)
