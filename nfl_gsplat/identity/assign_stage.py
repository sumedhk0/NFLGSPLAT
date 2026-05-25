"""Identity wiring stage: tracks → team/referee classification → entities.json.

Sits between cross-camera re-ID + jersey OCR and the pose/avatar stages. For
each cross-camera player (``global_player_id``) it:

1. takes a representative torso crop and runs :mod:`team_color`
   (team cluster + referee stripe test);
2. labels the two color clusters to real teams by **roster membership** of their
   OCR'd jerseys (no need for hand-specified team colors);
3. attaches ``team`` / ``is_referee`` columns and calls
   :func:`nfl_gsplat.identity.registry.resolve_tracks` (grouped by
   ``global_player_id``);
4. writes ``entities.json`` and updates the season registry.

``entities.json`` records, per on-field instance::

    {instance_id, player_uid, entity_type}

``instance_id`` (the global_player_id) keys the per-instance pose file; multiple
referees share ``player_uid == "__referee__"`` but keep distinct ``instance_id``s
so each is posed by its own motion.

The classification core takes pre-extracted crops so it is CPU-testable;
:func:`extract_representative_crops` (which reads the video) is the env-side glue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from nfl_gsplat.identity.registry import (
    REFEREE_UID,
    Assignment,
    EntityType,
    IdentityMatchConfig,
    assign_identities,
    resolve_tracks,
)
from nfl_gsplat.identity.roster import RosterEntry
from nfl_gsplat.identity.team_color import (
    RefereeConfig,
    dominant_jersey_color,
    is_referee,
    split_two_teams,
)
from nfl_gsplat.utils.io import write_json
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def _voted_jerseys(tracks_df: pd.DataFrame, id_col: str) -> dict:
    out: dict = {}
    for gid, grp in tracks_df.groupby(id_col):
        v = grp["jersey_number_ocr"]
        v = v[v >= 0]
        out[gid] = int(v.value_counts().idxmax()) if len(v) else -1
    return out


def classify_entities(
    crops_by_id: Mapping,
    jerseys_by_id: Mapping,
    candidates: list[RosterEntry],
    home_team: str,
    away_team: str,
    ref_cfg: RefereeConfig | None = None,
) -> dict:
    """Return ``{id: (team|None, is_referee)}``.

    Teams come from 2-means color clustering, with clusters labeled to real
    teams by which roster their members' jerseys match more often.
    """
    ids = list(crops_by_id.keys())
    if not ids:
        return {}
    refs = {g: bool(is_referee(crops_by_id[g], ref_cfg)) for g in ids}

    # Cluster team colors over *players only* — a referee's grayscale stripes
    # would otherwise pull a cluster center and flip the labels.
    player_ids = [g for g in ids if not refs[g]]
    label_by_id: dict = {}
    if player_ids:
        colors = np.stack([dominant_jersey_color(crops_by_id[g]) for g in player_ids])
        labels = split_two_teams(colors)
        label_by_id = {g: int(lab) for g, lab in zip(player_ids, labels)}

    home_j = {c.jersey for c in candidates if c.team == home_team}
    away_j = {c.jersey for c in candidates if c.team == away_team}

    # Per-cluster home-vs-away affinity from members' jerseys.
    margin = {0: 0, 1: 0}
    for g in player_ids:
        j = jerseys_by_id.get(g, -1)
        c = label_by_id[g]
        if j in home_j:
            margin[c] += 1
        if j in away_j:
            margin[c] -= 1

    if not candidates:
        cluster_team: dict = {0: None, 1: None}
    else:
        home_cluster = 0 if margin[0] >= margin[1] else 1
        cluster_team = {home_cluster: home_team, 1 - home_cluster: away_team}

    return {
        g: (None, True) if refs[g] else (cluster_team[label_by_id[g]], False)
        for g in ids
    }


def assign_play_identities(
    tracks_df: pd.DataFrame,
    crops_by_id: Mapping,
    candidates: list[RosterEntry],
    home_team: str,
    away_team: str,
    cfg: IdentityMatchConfig,
    *,
    id_col: str = "global_player_id",
    ref_cfg: RefereeConfig | None = None,
) -> tuple[pd.DataFrame, list[Assignment]]:
    """Classify teams/referees, attach columns, and resolve identities."""
    jerseys = _voted_jerseys(tracks_df, id_col)
    ent = classify_entities(crops_by_id, jerseys, candidates, home_team, away_team, ref_cfg)

    df = tracks_df.copy()
    df["team"] = df[id_col].map({g: t for g, (t, _) in ent.items()})
    df["is_referee"] = df[id_col].map({g: r for g, (_, r) in ent.items()}).fillna(False)

    resolved = resolve_tracks(df, candidates, cfg, id_col=id_col)
    assignments = assign_identities(df, candidates, cfg, id_col=id_col)
    return resolved, assignments


def write_entities_json(path: Path | str, assignments: list[Assignment]) -> Path:
    """Persist the renderable entities (players + referees; OTHER dropped).

    One entry per instance (``instance_id`` = the resolved id), so referees that
    share ``__referee__`` keep distinct pose files.
    """
    entities = [
        {"instance_id": str(a.track_id), "player_uid": a.player_uid, "entity_type": a.entity_type}
        for a in assignments
        if a.entity_type != EntityType.OTHER.value and a.player_uid
    ]
    write_json(path, entities)
    n_ref = sum(1 for e in entities if e["player_uid"] == REFEREE_UID)
    _LOG.info(f"entities.json: {len(entities)} instances ({n_ref} referee) → {path}")
    return Path(path)
