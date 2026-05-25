"""Cross-play identity consensus for OCR-only synthesized uids.

In ``ocr_only`` mode, identities are synthesized per play as ``{season}_{team}_{jersey}``,
so a single OCR misread spawns a spurious uid (e.g. ``2024_HOME_72`` for one play
while the player is ``2024_HOME_12`` in 30 others). Using the season registry's
accumulated counts, we remap rare uids onto a dominant same-team neighbor whose
jersey is within OCR-confusable distance.

Roster-mode uids (nflverse gsis ids) don't parse as ``season_team_jersey`` and are
left untouched — they're already canonical.
"""
from __future__ import annotations

from typing import Any

from nfl_gsplat.identity.registry import EntityType


def _parse_synthetic(uid: str) -> tuple[str, str, int] | None:
    """Return ``(season, team, jersey)`` if ``uid`` is a synthesized id, else None."""
    parts = uid.split("_")
    if len(parts) != 3:
        return None
    season, team, jersey = parts
    if not season.isdigit() or not jersey.lstrip("-").isdigit():
        return None
    return season, team, int(jersey)


def build_consensus_remap(
    registry: dict[str, Any],
    *,
    min_majority: int = 3,
    max_minority: int = 1,
    max_jersey_gap: int = 1,
) -> dict[str, str]:
    """Map rare synthesized uids → a dominant same-(season, team) neighbor.

    ``registry`` is the dict from ``identity.registry.load_registry``. A uid is
    remapped when it was seen in ``<= max_minority`` plays and a uid with the
    same (season, team) whose jersey differs by ``<= max_jersey_gap`` was seen in
    ``>= min_majority`` plays.
    """
    uids = registry.get("uids", {})
    # Play-count per uid (number of distinct plays it was assigned in).
    counts: dict[str, int] = {}
    for key, recs in registry.get("plays", {}).items():
        for rec in recs:
            if rec.get("entity_type") == EntityType.PLAYER.value and rec.get("player_uid"):
                counts[rec["player_uid"]] = counts.get(rec["player_uid"], 0) + 1
    # Fall back to jersey-vote totals if plays aren't recorded.
    for uid, rec in uids.items():
        counts.setdefault(uid, sum(rec.get("jersey_votes", {}).values()))

    parsed = {u: _parse_synthetic(u) for u in counts}
    remap: dict[str, str] = {}
    for uid, p in parsed.items():
        if p is None or counts.get(uid, 0) > max_minority:
            continue
        season, team, jersey = p
        best, best_count = None, min_majority - 1
        for other, q in parsed.items():
            if other == uid or q is None:
                continue
            os_, ot, oj = q
            if os_ == season and ot == team and abs(oj - jersey) <= max_jersey_gap:
                if counts.get(other, 0) > best_count:
                    best, best_count = other, counts.get(other, 0)
        if best is not None:
            remap[uid] = best
    return remap


def apply_remap(entities: list[dict], remap: dict[str, str]) -> list[dict]:
    """Return entities with ``player_uid`` rewritten per ``remap`` (copy)."""
    out = []
    for e in entities:
        e2 = dict(e)
        e2["player_uid"] = remap.get(e["player_uid"], e["player_uid"])
        out.append(e2)
    return out
