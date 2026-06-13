from __future__ import annotations

import pytest

from nfl_gsplat.season.scaffold import scaffold_play
from nfl_gsplat.utils.meta import load_meta


def test_scaffold_creates_folder_and_meta(tmp_path):
    pd = scaffold_play(tmp_path / "data", season="2024", week=1,
                       away="NO", home="ATL", play="play_001", fps=30.0,
                       gsis_play_id="36")
    assert pd.dir == tmp_path / "data" / "2024" / "week_01" / "NO_at_ATL" / "play_001"
    assert pd.dir.is_dir()
    m = load_meta(pd.meta_yaml)
    assert m.game_teams == ("ATL", "NO")
    assert m.fps == 30.0
    assert m.gsis_play_id == "36"


def test_scaffold_refuses_overwrite(tmp_path):
    scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                  home="ATL", play="play_001")
    with pytest.raises(FileExistsError):
        scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                      home="ATL", play="play_001")


def test_scaffold_force_overwrites(tmp_path):
    scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                  home="ATL", play="play_001", fps=30.0)
    pd = scaffold_play(tmp_path / "data", season="2024", week=1, away="NO",
                       home="ATL", play="play_001", fps=60.0, force=True)
    assert load_meta(pd.meta_yaml).fps == 60.0
