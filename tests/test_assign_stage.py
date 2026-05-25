"""Identity wiring stage (T1.4): team/referee classification → entities.json."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_gsplat.identity.assign_stage import (
    assign_play_identities,
    classify_entities,
    write_entities_json,
)
from nfl_gsplat.identity.registry import REFEREE_UID, EntityType, IdentityMatchConfig
from nfl_gsplat.identity.roster import RosterEntry
from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
from nfl_gsplat.utils.io import read_json


def _solid(bgr, h=80, w=60):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def _stripes(h=80, w=60, period=6):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        if (x // period) % 2 == 0:
            img[:, x] = (255, 255, 255)
    return img


RED, BLUE = (0, 0, 200), (200, 0, 0)
_CANDS = [
    RosterEntry("h12", "HOME", 12, "QB"), RosterEntry("h80", "HOME", 80, "WR"),
    RosterEntry("a12", "AWAY", 12, "CB"), RosterEntry("a81", "AWAY", 81, "WR"),
]


def _tracks(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        base = {c: 0 for c in TRACK_COLUMNS}
        base.update({"cam": "sideline", "conf": 0.9})
        base.update(r)
        out.append(base)
    return pd.DataFrame(out)


def test_classify_labels_clusters_by_roster_membership():
    crops = {1: _solid(RED), 2: _solid(RED), 3: _solid(BLUE), 4: _solid(BLUE), 9: _stripes()}
    jerseys = {1: 12, 2: 80, 3: 12, 4: 81, 9: -1}
    ent = classify_entities(crops, jerseys, _CANDS, "HOME", "AWAY")
    # Red cluster carries HOME jerseys (12, 80); blue carries AWAY (12, 81).
    assert ent[1][0] == "HOME" and ent[2][0] == "HOME"
    assert ent[3][0] == "AWAY" and ent[4][0] == "AWAY"
    # Striped track flagged referee, team None.
    assert ent[9] == (None, True)


def test_assign_play_identities_resolves_colliding_jerseys():
    # Two #12s — one HOME (red), one AWAY (blue) — disambiguated by team color.
    # Unambiguous anchors (#80 HOME, #81 AWAY) label which color cluster is which.
    rows = []
    for gid, jersey in [(1, 12), (2, 12), (3, 80), (4, 81)]:
        for f in range(4):
            rows.append({"frame": f, "global_player_id": gid, "track_id": gid,
                         "jersey_number_ocr": jersey})
    rows.append({"frame": 0, "global_player_id": 9, "track_id": 9, "jersey_number_ocr": -1})
    df = _tracks(rows)
    crops = {1: _solid(RED), 2: _solid(BLUE), 3: _solid(RED), 4: _solid(BLUE), 9: _stripes()}

    cfg = IdentityMatchConfig(season=2024)
    _, assignments = assign_play_identities(df, crops, _CANDS, "HOME", "AWAY", cfg)
    by_id = {a.track_id: a for a in assignments}
    assert by_id[1].player_uid == "h12"     # red #12 → HOME (anchored by red #80)
    assert by_id[2].player_uid == "a12"     # blue #12 → AWAY (anchored by blue #81)
    assert by_id[9].entity_type == EntityType.REFEREE.value


def test_write_entities_json_keeps_instances_and_drops_other(tmp_path):
    rows = []
    # gid 1: HOME #80 (unambiguous). gid 5: #99 on neither roster → OTHER.
    for gid, jersey in [(1, 80), (5, 99)]:
        rows.append({"frame": 0, "global_player_id": gid, "track_id": gid,
                     "jersey_number_ocr": jersey, "cam": "sideline", "conf": 0.9})
    # add a referee + a second referee (distinct instances, shared uid)
    rows.append({"frame": 0, "global_player_id": 9, "track_id": 9, "jersey_number_ocr": -1,
                 "cam": "sideline", "conf": 0.9})
    rows.append({"frame": 0, "global_player_id": 10, "track_id": 10, "jersey_number_ocr": -1,
                 "cam": "sideline", "conf": 0.9})
    df = _tracks(rows)
    # gid 5 gets a team but no matching jersey AND we make it OTHER by team gating:
    crops = {1: _solid(RED), 5: _solid(BLUE), 9: _stripes(), 10: _stripes()}
    cfg = IdentityMatchConfig(season=2024)
    _, assignments = assign_play_identities(df, crops, _CANDS, "HOME", "AWAY", cfg)

    out = write_entities_json(tmp_path / "entities.json", assignments)
    entities = read_json(out)
    uids = [e["player_uid"] for e in entities]
    # Two distinct referee instances share the avatar uid but keep separate ids.
    ref_entries = [e for e in entities if e["player_uid"] == REFEREE_UID]
    assert len(ref_entries) == 2
    assert {e["instance_id"] for e in ref_entries} == {"9", "10"}
    # The real player is present; OTHER (uid "") is dropped.
    assert "h80" in uids
    assert "" not in uids
