from __future__ import annotations

from nfl_gsplat.season.discover import discover_plays


def _make_play(root, season, week, matchup, play, *, full=True):
    d = root / "data" / season / f"week_{week:02d}" / matchup / play
    d.mkdir(parents=True)
    if full:
        (d / "sideline.mp4").write_bytes(b"x")
        (d / "endzone.mp4").write_bytes(b"x")
        (d / "meta.yaml").write_text("season: 2024\nweek: 1\nhome_team: ATL\naway_team: \"NO\"\nfps: 30\n")
    return d


def test_discover_orders_week_matchup_play(tmp_path):
    _make_play(tmp_path, "2024", 2, "NO_at_ATL", "play_001")
    _make_play(tmp_path, "2024", 1, "GB_at_CHI", "play_002")
    _make_play(tmp_path, "2024", 1, "GB_at_CHI", "play_001")
    plays = discover_plays(tmp_path / "data", "2024")
    got = [(p.week, p.matchup, p.play_id) for p in plays]
    assert got == [
        (1, "GB_at_CHI", "play_001"),
        (1, "GB_at_CHI", "play_002"),
        (2, "NO_at_ATL", "play_001"),
    ]


def test_discover_skips_incomplete(tmp_path, caplog):
    _make_play(tmp_path, "2024", 1, "NO_at_ATL", "play_001")          # complete
    _make_play(tmp_path, "2024", 1, "NO_at_ATL", "play_002", full=False)  # no videos/meta
    plays = discover_plays(tmp_path / "data", "2024")
    assert [p.play_id for p in plays] == ["play_001"]
