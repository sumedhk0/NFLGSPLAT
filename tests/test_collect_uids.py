from __future__ import annotations
from nfl_gsplat.season.collect_uids import find_entities_files
from nfl_gsplat.utils.io import write_json


def test_find_entities_files_walks_tree(tmp_path):
    base = tmp_path / "data" / "2024" / "week_01" / "NO_at_ATL"
    for play in ("play_001", "play_002"):
        d = base / play
        d.mkdir(parents=True)
        write_json(d / "entities.json", [{"entity_type": "player", "player_uid": "p1"}])
    found = find_entities_files(tmp_path / "data", "2024")
    assert len(found) == 2
    assert all(f.name == "entities.json" for f in found)
