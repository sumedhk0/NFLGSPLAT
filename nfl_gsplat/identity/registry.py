"""Resolve tracks to stable ``player_uid`` + ``entity_type`` and persist them.

Given a play's tracks (the :data:`~nfl_gsplat.tracking.detect_track.TRACK_COLUMNS`
DataFrame, optionally augmented upstream with ``team`` and ``is_referee``
columns from :mod:`nfl_gsplat.identity.team_color`) and the candidate roster for
that play (:class:`~nfl_gsplat.identity.roster.IdentitySource`), we:

1. Summarize each track: majority-voted jersey, majority team, referee flag.
2. Hungarian-assign tracks to roster candidates on a cost combining jersey
   agreement (a misread snaps to the nearest valid jersey *for the matched
   team*) and team agreement (a hard gate that disambiguates colliding home /
   away numbers). Reuses ``scipy.optimize.linear_sum_assignment`` exactly like
   :func:`nfl_gsplat.tracking.cross_cam_reid.assign_global_ids`.
3. Tracks that match no candidate become ``REFEREE`` (if the stripe detector
   fired) or ``OTHER`` (dropped downstream).

With no roster (``OcrOnlySource`` → empty candidates) we synthesize uids from
the observed (team, jersey) so the pipeline still runs, just without cross-game
identity guarantees.

The persistent store at ``outputs/{season}/identity/registry.json`` accumulates
per-play assignments + per-uid jersey-vote histograms, written atomically via
:func:`nfl_gsplat.utils.io.write_json`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from nfl_gsplat.identity.roster import RosterEntry, synthetic_uid
from nfl_gsplat.utils.io import read_json, write_json
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

REFEREE_UID = "__referee__"
OTHER_UID = ""


class EntityType(str, Enum):
    PLAYER = "player"
    REFEREE = "referee"
    OTHER = "other"


@dataclass(frozen=True)
class IdentityMatchConfig:
    season: int | str = 0
    jersey_weight: float = 1.0
    team_mismatch_cost: float = 1000.0   # hard gate against cross-team matches
    unknown_jersey_cost: float = 5.0     # cost when a track has no voted jersey
    max_jersey_digit_cost: float = 10.0  # cap on |voted - candidate| digit gap
    max_match_cost: float = 8.0          # reject assignments above this cost


@dataclass(frozen=True)
class _TrackSummary:
    track_id: int
    jersey: int           # -1 when no confident OCR
    team: str | None
    is_referee: bool


def _summarize(tracks_df: pd.DataFrame) -> list[_TrackSummary]:
    """Collapse per-frame/per-cam rows into one summary per ``track_id``."""
    has_team = "team" in tracks_df.columns
    has_ref = "is_referee" in tracks_df.columns
    out: list[_TrackSummary] = []
    for tid, grp in tracks_df.groupby("track_id"):
        votes = grp["jersey_number_ocr"]
        votes = votes[votes >= 0]
        jersey = int(votes.value_counts().idxmax()) if len(votes) else -1

        team = None
        if has_team:
            tvals = grp["team"].dropna()
            tvals = tvals[tvals.astype(str) != ""]
            team = str(tvals.value_counts().idxmax()) if len(tvals) else None

        is_ref = bool(grp["is_referee"].mean() >= 0.5) if has_ref else False
        out.append(_TrackSummary(int(tid), jersey, team, is_ref))
    return out


def _jersey_cost(voted: int, candidate: int, cfg: IdentityMatchConfig) -> float:
    if voted < 0:
        return cfg.unknown_jersey_cost
    return float(min(abs(voted - candidate), cfg.max_jersey_digit_cost))


def _pair_cost(s: _TrackSummary, c: RosterEntry, cfg: IdentityMatchConfig) -> float:
    if s.team is not None and s.team != c.team:
        return cfg.team_mismatch_cost
    return cfg.jersey_weight * _jersey_cost(s.jersey, c.jersey, cfg)


@dataclass(frozen=True)
class Assignment:
    track_id: int
    player_uid: str
    entity_type: str
    confidence: float
    jersey: int
    team: str | None


def assign_identities(
    tracks_df: pd.DataFrame,
    candidates: list[RosterEntry],
    cfg: IdentityMatchConfig,
) -> list[Assignment]:
    """Resolve every track to an :class:`Assignment`. Pure / deterministic."""
    summaries = _summarize(tracks_df)
    if not summaries:
        return []

    # Referees never compete for roster slots — route them out first so a
    # jersey-less striped official can't claim a leftover candidate.
    refs = [s for s in summaries if s.is_referee]
    players = [s for s in summaries if not s.is_referee]
    out: list[Assignment] = [_fallback(s) for s in refs]

    if not candidates:
        out.extend(_synthesize(s, cfg) for s in players)
        return _reorder(out, summaries)

    if players:
        cost = np.array(
            [[_pair_cost(s, c, cfg) for c in candidates] for s in players],
            dtype=np.float64,
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        matched: dict[int, tuple[int, float]] = {}
        for r, cidx in zip(row_ind, col_ind):
            if cost[r, cidx] <= cfg.max_match_cost:
                matched[r] = (int(cidx), float(cost[r, cidx]))

        for r, s in enumerate(players):
            if r in matched:
                cidx, c_cost = matched[r]
                cand = candidates[cidx]
                conf = max(0.0, 1.0 - c_cost / cfg.max_match_cost)
                out.append(Assignment(s.track_id, cand.player_uid, EntityType.PLAYER.value,
                                      conf, cand.jersey, cand.team))
            else:
                out.append(_fallback(s))
    return _reorder(out, summaries)


def _reorder(assignments: list[Assignment], summaries: list[_TrackSummary]) -> list[Assignment]:
    """Restore the original track order (referees were split out first)."""
    by_track = {a.track_id: a for a in assignments}
    return [by_track[s.track_id] for s in summaries]


def _synthesize(s: _TrackSummary, cfg: IdentityMatchConfig) -> Assignment:
    """OCR-only identity: build a uid from observed (team, jersey)."""
    if s.is_referee:
        return Assignment(s.track_id, REFEREE_UID, EntityType.REFEREE.value, 1.0, s.jersey, s.team)
    if s.jersey >= 0:
        team = s.team or "UNK"
        return Assignment(s.track_id, synthetic_uid(cfg.season, team, s.jersey),
                          EntityType.PLAYER.value, 0.5, s.jersey, s.team)
    return Assignment(s.track_id, OTHER_UID, EntityType.OTHER.value, 0.0, s.jersey, s.team)


def _fallback(s: _TrackSummary) -> Assignment:
    """Track matched no roster candidate → referee (if striped) or other."""
    if s.is_referee:
        return Assignment(s.track_id, REFEREE_UID, EntityType.REFEREE.value, 1.0, s.jersey, s.team)
    return Assignment(s.track_id, OTHER_UID, EntityType.OTHER.value, 0.0, s.jersey, s.team)


def resolve_tracks(
    tracks_df: pd.DataFrame,
    candidates: list[RosterEntry],
    cfg: IdentityMatchConfig,
) -> pd.DataFrame:
    """Return a copy of ``tracks_df`` with ``player_uid`` + ``entity_type``
    columns filled by mapping each row's ``track_id`` to its assignment."""
    assignments = assign_identities(tracks_df, candidates, cfg)
    uid_by_track = {a.track_id: a.player_uid for a in assignments}
    type_by_track = {a.track_id: a.entity_type for a in assignments}
    out = tracks_df.copy()
    out["player_uid"] = out["track_id"].map(uid_by_track).fillna(OTHER_UID)
    out["entity_type"] = out["track_id"].map(type_by_track).fillna(EntityType.OTHER.value)
    return out


# --- Persistent registry store ---------------------------------------------

def _registry_path(season: int | str, outputs_root: Path | str = "outputs") -> Path:
    return Path(outputs_root) / str(season) / "identity" / "registry.json"


def load_registry(season: int | str, outputs_root: Path | str = "outputs") -> dict[str, Any]:
    """Load the season registry, or an empty skeleton if none exists yet."""
    path = _registry_path(season, outputs_root)
    if not path.exists():
        return {"season": str(season), "plays": {}, "uids": {}}
    return read_json(path)


def register_play(
    season: int | str,
    game_id: str,
    play_id: str,
    assignments: list[Assignment],
    *,
    outputs_root: Path | str = "outputs",
) -> dict[str, Any]:
    """Persist a play's assignments and update per-uid jersey-vote histograms.

    Returns the updated registry dict (also written to disk atomically).
    """
    reg = load_registry(season, outputs_root)
    key = f"{game_id}/{play_id}"
    reg["plays"][key] = [
        {
            "track_id": a.track_id,
            "player_uid": a.player_uid,
            "entity_type": a.entity_type,
            "confidence": round(a.confidence, 4),
        }
        for a in assignments
    ]
    uids: dict[str, Any] = reg.setdefault("uids", {})
    for a in assignments:
        if a.entity_type != EntityType.PLAYER.value or not a.player_uid:
            continue
        rec = uids.setdefault(a.player_uid, {"jersey_votes": {}, "games": [], "team": a.team})
        if a.jersey >= 0:
            jv = rec["jersey_votes"]
            jv[str(a.jersey)] = jv.get(str(a.jersey), 0) + 1
        if game_id not in rec["games"]:
            rec["games"].append(game_id)
    write_json(_registry_path(season, outputs_root), reg)
    _LOG.info(f"registry: recorded {len(assignments)} tracks for {key} "
              f"({len(reg['uids'])} known players)")
    return reg


def consensus_jersey(season: int | str, player_uid: str,
                     outputs_root: Path | str = "outputs") -> int | None:
    """Most-voted jersey for a uid across all recorded plays, or None."""
    reg = load_registry(season, outputs_root)
    rec = reg.get("uids", {}).get(player_uid)
    if not rec or not rec.get("jersey_votes"):
        return None
    return int(max(rec["jersey_votes"].items(), key=lambda kv: kv[1])[0])
