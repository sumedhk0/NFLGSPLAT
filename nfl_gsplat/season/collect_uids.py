"""Collect the unique player uids needing avatars across the season.

This is the dedup that makes caching correct under cluster parallelism: the
avatar-build SLURM array runs **one task per unique uid** (never two jobs racing
to write the same ``library/{season}/{uid}``). We scan every play's
``entities.json``, take the distinct player uids, and drop those already cached.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.identity.registry import EntityType
from nfl_gsplat.utils.io import read_json


def find_entities_files(outputs_root: Path | str, game_ids: Iterable[str]) -> list[Path]:
    """All ``entities.json`` under ``outputs/{game}/*/`` for the given games."""
    root = Path(outputs_root)
    out: list[Path] = []
    for game in game_ids:
        out.extend(sorted((root / game).glob("*/entities.json")))
    return out


def collect_player_uids(entities_files: Iterable[Path | str]) -> set[str]:
    """Distinct ``player_uid``s of entity_type player across the given files."""
    uids: set[str] = set()
    for path in entities_files:
        for ent in read_json(path):
            if ent.get("entity_type") == EntityType.PLAYER.value and ent.get("player_uid"):
                uids.add(ent["player_uid"])
    return uids


def uids_to_build(entities_files: Iterable[Path | str], library: AvatarLibrary) -> list[str]:
    """Sorted player uids that are not yet in the library (the S3 array work)."""
    return sorted(u for u in collect_player_uids(entities_files) if not library.has_avatar(u))
