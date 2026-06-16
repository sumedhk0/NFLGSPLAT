"""Per-player avatar/shape library: round-trips, betas, provenance, rebuild,
and the generic referee / football asset slots.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.avatars.library import FOOTBALL_UID, AvatarLibrary
from nfl_gsplat.identity.registry import REFEREE_UID
from nfl_gsplat.utils.io import read_json


def _avatar(n: int = 200, j: int = 22, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "canonical_xyz": rng.normal(0, 0.2, (n, 3)).astype(np.float32),
        "canonical_rot": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "canonical_scale": np.full((n, 3), np.log(0.04), dtype=np.float32),
        "canonical_opacity": np.full((n,), 2.0, dtype=np.float32),
        "canonical_sh": rng.normal(0, 0.1, (n, 3, 1)).astype(np.float32),
        "lbs_weights": np.eye(j)[rng.integers(0, j, n)].astype(np.float32),
    }


def _football(n: int = 150, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "xyz": rng.normal(0, 0.05, (n, 3)).astype(np.float32),
        "rot": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "scale": np.full((n, 3), np.log(0.02), dtype=np.float32),
        "opacity": np.full((n,), 2.0, dtype=np.float32),
        "sh": rng.normal(0, 0.1, (n, 3, 1)).astype(np.float32),
    }


def test_put_get_has_roundtrip(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    uid = "00-1234"
    assert not lib.has_avatar(uid)
    lib.put_avatar(uid, _avatar(seed=5), provenance={"game_id": "game_001", "play_id": "p1"})
    assert lib.has_avatar(uid)
    got = lib.get_avatar(uid)
    assert got["canonical_xyz"].shape == (200, 3)
    assert np.allclose(got["canonical_xyz"], _avatar(seed=5)["canonical_xyz"])


def test_betas_persist(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    uid = "00-7"
    betas = np.arange(10, dtype=np.float32)
    lib.put_avatar(uid, _avatar(), betas=betas)
    assert np.allclose(lib.get_betas(uid), betas)


def test_betas_absent_returns_none(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    lib.put_avatar("00-8", _avatar())
    assert lib.get_betas("00-8") is None


def test_meta_written_with_provenance_and_hash(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    uid = "00-9"
    lib.put_avatar(uid, _avatar(), provenance={"game_id": "g", "frame": 42, "tier": "lhm_mini"})
    meta = read_json(lib._meta_path(uid))
    assert meta["player_uid"] == uid
    assert meta["entity_type"] == "player"
    assert meta["provenance"]["frame"] == 42
    assert meta["avatar_sha256"] and len(meta["avatar_sha256"]) == 64


def test_rebuild_forces_miss_and_overwrite(tmp_path):
    root = tmp_path / "library"
    lib = AvatarLibrary(root, season=2024)
    uid = "00-r"
    lib.put_avatar(uid, _avatar(seed=1))
    # New handle with rebuild=True: existing file reported as a miss.
    lib_rebuild = AvatarLibrary(root, season=2024, rebuild=True)
    assert lib_rebuild.has_avatar(uid) is False
    lib_rebuild.put_avatar(uid, _avatar(seed=2))
    # Sticky handle now reads the overwritten avatar.
    got = AvatarLibrary(root, season=2024).get_avatar(uid)
    assert np.allclose(got["canonical_xyz"], _avatar(seed=2)["canonical_xyz"])


def test_referee_asset_slot(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    assert not lib.has_referee_avatar()
    lib.put_referee_avatar(_avatar(seed=3))
    assert lib.has_referee_avatar()
    assert lib.get_referee_avatar()["canonical_xyz"].shape == (200, 3)


def test_football_asset_slot_is_season_agnostic(tmp_path):
    root = tmp_path / "library"
    lib = AvatarLibrary(root, season=2024)
    lib.put_football_asset(_football())
    assert lib.has_football_asset()
    # A different season sees the same global football asset.
    lib_other = AvatarLibrary(root, season=2025)
    assert lib_other.has_football_asset()
    assert set(lib_other.get_football_asset().keys()) == {"xyz", "rot", "scale", "opacity", "sh"}


def test_index_excludes_reserved_assets(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    lib.put_avatar("00-A", _avatar())
    lib.put_referee_avatar(_avatar(seed=9))
    lib.put_football_asset(_football())
    idx = lib.index()
    assert "00-A" in idx
    assert REFEREE_UID not in idx and FOOTBALL_UID not in idx
    assert idx["00-A"]["entity_type"] == "player"


def test_library_empty_season_flat_layout(tmp_path):
    import numpy as np  # noqa: F401  (kept for parity if needed)
    from nfl_gsplat.avatars.library import AvatarLibrary
    from nfl_gsplat.avatars.lhm_wrapper import write_mock_avatar
    from nfl_gsplat.utils.io import read_npz

    lib = AvatarLibrary(root=tmp_path / "_library", season="")
    out = write_mock_avatar(tmp_path / "mock.npz", num_gaussians=64, num_joints=22)
    lib.put_avatar("p_7", read_npz(out))
    assert (tmp_path / "_library" / "p_7" / "avatar.npz").exists()
    assert lib.has_avatar("p_7")
