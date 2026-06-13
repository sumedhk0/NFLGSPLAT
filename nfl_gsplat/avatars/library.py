"""Per-player avatar + shape library — a cache tier keyed by ``player_uid``.

This sits **above** the per-play content-hash manifest in
:mod:`nfl_gsplat.utils.io`. That manifest skips re-running a stage for the same
play; this library skips re-running the *expensive LHM++ feed-forward* for a
player we have already reconstructed — in any play, in any game of the season.

On a cache hit the avatar stage loads the canonical Gaussian avatar instead of
generating it, and the pose stage reuses the frozen ``betas`` so the cached
avatar's rig and the per-play pose skeleton share bone lengths.

On-disk layout (``root`` defaults to ``library/``)::

    {root}/{season}/{player_uid}/avatar.npz   canonical Gaussian schema
    {root}/{season}/{player_uid}/betas.npz    frozen SMPL-X shape vector
    {root}/{season}/{player_uid}/meta.json    provenance + sha256s
    {root}/{season}/index.json                uid -> {entity_type, paths}
    {root}/{season}/_assets/referee/...        generic striped-shirt avatar
    {root}/_assets/football/asset.npz          canonical football (season-agnostic)

Reserved uids ``__referee__`` (per-season) and ``__football__`` (global) hold
the generic assets. Sticky by default; pass ``rebuild=True`` to force
regeneration (``has_avatar`` then reports misses so callers regenerate + put).

CPU-only; reuses the atomic NPZ/JSON writers and sha256 hashing from
:mod:`nfl_gsplat.utils.io`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nfl_gsplat.identity.registry import REFEREE_UID
from nfl_gsplat.utils.io import read_json, read_npz, sha256_file, write_json, write_npz
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

FOOTBALL_UID = "__football__"

# Canonical avatar NPZ keys (must match lhm_wrapper.write_mock_avatar output).
AVATAR_KEYS = (
    "canonical_xyz", "canonical_rot", "canonical_scale",
    "canonical_opacity", "canonical_sh", "lbs_weights",
)
# Football asset NPZ keys (GaussianBatch fields).
ASSET_KEYS = ("xyz", "rot", "scale", "opacity", "sh")


class AvatarLibrary:
    """Season-scoped avatar + shape cache rooted at ``root``."""

    def __init__(self, root: Path | str = "library", season: int | str = 0,
                 *, rebuild: bool = False) -> None:
        self.root = Path(root)
        self.season = str(season)
        self.rebuild = rebuild

    # --- path helpers -------------------------------------------------------

    def _player_dir(self, uid: str) -> Path:
        if uid == REFEREE_UID:
            base = self.root / self.season if self.season else self.root
            return base / "_assets" / "referee"
        if uid == FOOTBALL_UID:
            return self.root / "_assets" / "football"
        return (self.root / self.season / uid) if self.season else (self.root / uid)

    def _avatar_path(self, uid: str) -> Path:
        name = "asset.npz" if uid == FOOTBALL_UID else "avatar.npz"
        return self._player_dir(uid) / name

    def _betas_path(self, uid: str) -> Path:
        return self._player_dir(uid) / "betas.npz"

    def _meta_path(self, uid: str) -> Path:
        return self._player_dir(uid) / "meta.json"

    def _index_path(self) -> Path:
        return self.root / self.season / "index.json"

    # --- avatar -------------------------------------------------------------

    def has_avatar(self, uid: str) -> bool:
        """True if a cached avatar exists. Always False under ``rebuild`` so
        callers regenerate and overwrite."""
        if self.rebuild:
            return False
        return self._avatar_path(uid).exists()

    def get_avatar(self, uid: str) -> dict[str, np.ndarray]:
        path = self._avatar_path(uid)
        if not path.exists():
            raise FileNotFoundError(f"no cached avatar for uid {uid!r} at {path}")
        return read_npz(path)

    def put_avatar(
        self,
        uid: str,
        avatar: dict[str, np.ndarray],
        *,
        betas: np.ndarray | None = None,
        entity_type: str = "player",
        provenance: dict[str, Any] | None = None,
    ) -> Path:
        """Write an avatar (and optional betas) + provenance meta.

        Accepts either the *canonical* LBS schema (:data:`AVATAR_KEYS`, driven by
        ``animate_gaussians``) or the *LHM-native* schema (:data:`LHM_NATIVE_KEYS`,
        animated by LHM's own engine at render time). The stored keys are
        whichever full schema the dict satisfies; the render scene re-detects the
        format on load.
        """
        from nfl_gsplat.avatars.lhm_wrapper import LHM_NATIVE_KEYS

        if all(k in avatar for k in AVATAR_KEYS):
            schema = AVATAR_KEYS
        elif all(k in avatar for k in LHM_NATIVE_KEYS):
            schema = LHM_NATIVE_KEYS
        else:
            missing = [k for k in AVATAR_KEYS if k not in avatar]
            raise ValueError(
                f"avatar dict matches neither schema; missing canonical keys "
                f"{missing} (and not LHM-native either)"
            )
        path = self._avatar_path(uid)
        # ``tier`` is an optional string array; pass through if present.
        arrays = {k: avatar[k] for k in schema}
        if "tier" in avatar:
            arrays["tier"] = avatar["tier"]
        write_npz(path, **arrays)

        if betas is not None:
            self.put_betas(uid, betas, provenance=provenance)

        self._write_meta(uid, entity_type=entity_type, provenance=provenance)
        self._update_index(uid, entity_type=entity_type)
        _LOG.info(f"library: stored avatar for {uid!r} → {path}")
        return path

    # --- betas --------------------------------------------------------------

    def get_betas(self, uid: str) -> np.ndarray | None:
        path = self._betas_path(uid)
        if not path.exists():
            return None
        return read_npz(path)["betas"]

    def put_betas(self, uid: str, betas: np.ndarray,
                  *, provenance: dict[str, Any] | None = None) -> Path:
        path = self._betas_path(uid)
        write_npz(path, betas=np.asarray(betas, dtype=np.float32))
        return path

    # --- generic assets -----------------------------------------------------

    def get_referee_avatar(self) -> dict[str, np.ndarray]:
        return self.get_avatar(REFEREE_UID)

    def put_referee_avatar(self, avatar: dict[str, np.ndarray],
                           *, provenance: dict[str, Any] | None = None) -> Path:
        return self.put_avatar(REFEREE_UID, avatar,
                               entity_type="referee", provenance=provenance)

    def has_referee_avatar(self) -> bool:
        return self.has_avatar(REFEREE_UID)

    def get_football_asset(self) -> dict[str, np.ndarray]:
        path = self._avatar_path(FOOTBALL_UID)
        if not path.exists():
            raise FileNotFoundError(f"no football asset at {path}")
        return read_npz(path)

    def put_football_asset(self, asset: dict[str, np.ndarray],
                           *, provenance: dict[str, Any] | None = None) -> Path:
        missing = [k for k in ASSET_KEYS if k not in asset]
        if missing:
            raise ValueError(f"football asset missing keys {missing}")
        path = self._avatar_path(FOOTBALL_UID)
        write_npz(path, **{k: asset[k] for k in ASSET_KEYS})
        self._write_meta(FOOTBALL_UID, entity_type="football", provenance=provenance)
        return path

    def has_football_asset(self) -> bool:
        return self._avatar_path(FOOTBALL_UID).exists()

    # --- meta + index -------------------------------------------------------

    def _write_meta(self, uid: str, *, entity_type: str,
                    provenance: dict[str, Any] | None) -> None:
        avatar_path = self._avatar_path(uid)
        meta = {
            "player_uid": uid,
            "entity_type": entity_type,
            "season": self.season,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "avatar_sha256": sha256_file(avatar_path) if avatar_path.exists() else None,
            "provenance": provenance or {},
        }
        betas_path = self._betas_path(uid)
        if betas_path.exists():
            meta["betas_sha256"] = sha256_file(betas_path)
        write_json(self._meta_path(uid), meta)

    def _update_index(self, uid: str, *, entity_type: str) -> None:
        # Reserved global/non-player assets stay out of the per-season index.
        if uid in (REFEREE_UID, FOOTBALL_UID):
            return
        path = self._index_path()
        index = read_json(path) if path.exists() else {}
        index[uid] = {
            "entity_type": entity_type,
            "avatar": str(self._avatar_path(uid).relative_to(self.root)),
            "betas": str(self._betas_path(uid).relative_to(self.root)),
        }
        write_json(path, index)

    def index(self) -> dict[str, Any]:
        path = self._index_path()
        return read_json(path) if path.exists() else {}
