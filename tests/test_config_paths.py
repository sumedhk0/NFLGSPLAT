"""Config loader + path resolver foundation (T1.0)."""
from __future__ import annotations

from pathlib import Path

from nfl_gsplat.config import load_config
from nfl_gsplat.paths import game_paths, play_paths


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


def test_game_paths_layout():
    cfg = load_config(overrides=["identity.season=2024"])
    gp = game_paths(cfg, "game_001")
    assert gp.raw_video("sideline") == Path("data/raw/game_001/sideline.mp4")
    assert gp.plays_yaml == Path("data/raw/game_001/plays.yaml")
    assert gp.field_ply == Path("outputs/game_001/field/field.ply")
    assert gp.calib_json == Path("outputs/game_001/calib/cameras.json")
    assert gp.annotations("endzone") == Path("data/annotations/game_001/endzone_landmarks.json")
    assert gp.library_dir == Path("library/2024")
    assert gp.rosters_dir == Path("data/rosters/2024")


def test_play_paths_layout():
    cfg = load_config(overrides=["identity.season=2024"])
    pp = play_paths(cfg, "game_001", "play_007")
    assert pp.dir == Path("outputs/game_001/play_007")
    assert pp.tracks == Path("outputs/game_001/play_007/tracks.parquet")
    assert pp.entities == Path("outputs/game_001/play_007/entities.json")
    assert pp.pose("00-1234") == Path("outputs/game_001/play_007/poses/00-1234.npz")
    assert pp.ball == Path("outputs/game_001/play_007/ball.npz")
    assert pp.render_mp4 == Path("outputs/game_001/play_007/render.mp4")
