"""Season-scoped player identity: roster prior, team/referee classification,
and a persistent registry resolving tracks to stable ``player_uid`` values.

The identity layer turns per-play, play-local tracking into stable cross-game
identities so the avatar/shape library (``nfl_gsplat.avatars.library``) can
build each player once and reuse them for every play.

All modules here are CPU-only (numpy / pandas / scipy / opencv) and safe to
import from any conda env.
"""
from __future__ import annotations

from nfl_gsplat.identity.registry import (
    EntityType,
    IdentityMatchConfig,
    REFEREE_UID,
    load_registry,
    register_play,
    resolve_tracks,
)
from nfl_gsplat.identity.roster import (
    IdentitySource,
    OcrOnlySource,
    RosterEntry,
    RosterSource,
)

__all__ = [
    "EntityType",
    "IdentityMatchConfig",
    "REFEREE_UID",
    "load_registry",
    "register_play",
    "resolve_tracks",
    "IdentitySource",
    "OcrOnlySource",
    "RosterEntry",
    "RosterSource",
]
