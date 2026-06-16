"""Config loader + path resolver foundation (T1.0)."""
from __future__ import annotations

from pathlib import Path

from nfl_gsplat.config import load_config


def test_load_config_reads_pipeline():
    cfg = load_config()
    assert cfg.identity.enabled is True
    assert cfg.avatars.library.path == "library"


def test_dotlist_override_wins_and_propagates_interpolation():
    cfg = load_config(overrides=["identity.season=2024"])
    assert int(cfg.identity.season) == 2024
    # roster_path: data/rosters/${identity.season} resolves through the override.
    assert cfg.identity.roster_path == "data/rosters/2024"


def test_overlay_stage_yaml(tmp_path: Path):
    stage = tmp_path / "stage.yaml"
    stage.write_text("avatars:\n  library:\n    rebuild: true\n")
    cfg = load_config(stage)
    assert cfg.avatars.library.rebuild is True
    # base keys survive the overlay.
    assert cfg.identity.enabled is True


def test_play_dir_layout():
    from nfl_gsplat.paths import PlayDir
    pd = PlayDir(season="2024", week=1, matchup="NO_at_ATL", play_id="play_001")
    assert pd.dir == Path("data/2024/week_01/NO_at_ATL/play_001")
    assert pd.video("sideline") == Path("data/2024/week_01/NO_at_ATL/play_001/sideline.mp4")
    assert pd.cameras_json == Path("data/2024/week_01/NO_at_ATL/play_001/cameras.json")
    assert pd.field_ply == Path("data/2024/week_01/NO_at_ATL/play_001/field.ply")
    assert pd.tracks == Path("data/2024/week_01/NO_at_ATL/play_001/tracks.parquet")
    assert pd.entities == Path("data/2024/week_01/NO_at_ATL/play_001/entities.json")
    assert pd.pose("00-1234") == Path("data/2024/week_01/NO_at_ATL/play_001/poses/00-1234.npz")
    assert pd.ball == Path("data/2024/week_01/NO_at_ATL/play_001/ball.npz")
    assert pd.render_mp4 == Path("data/2024/week_01/NO_at_ATL/play_001/render.mp4")
    assert pd.meta_yaml == Path("data/2024/week_01/NO_at_ATL/play_001/meta.yaml")


def test_play_dir_season_shared_roots():
    from nfl_gsplat.paths import PlayDir
    pd = PlayDir(season="2024", week=12, matchup="NO_at_ATL", play_id="play_003")
    assert pd.library_root == Path("data/2024/_library")
    assert pd.rosters_root == Path("data/2024/_rosters")
    assert pd.registry_path == Path("data/2024/_registry.json")
    assert pd.teams == ("ATL", "NO")          # (home, away)


def test_play_dir_from_dir_roundtrip(tmp_path):
    from nfl_gsplat.paths import PlayDir
    p = tmp_path / "data" / "2024" / "week_05" / "NO_at_ATL" / "play_002"
    p.mkdir(parents=True)
    pd = PlayDir.from_dir(p)
    assert (pd.season, pd.week, pd.matchup, pd.play_id) == ("2024", 5, "NO_at_ATL", "play_002")
    assert pd.dir == p
    assert pd.library_root == tmp_path / "data" / "2024" / "_library"
