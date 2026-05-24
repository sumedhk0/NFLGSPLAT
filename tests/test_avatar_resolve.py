"""Avatar-stage short-circuit + referee routing (Layer D).

A mock generator stands in for env-gated LHM++ so the cache logic is exercised
on CPU: repeat appearances of a uid must reuse the library, never regenerate.
"""
from __future__ import annotations

import numpy as np
import pytest

from nfl_gsplat.avatars.library import AvatarLibrary
from nfl_gsplat.avatars.lhm_wrapper import resolve_avatars, select_reference_index
from nfl_gsplat.errors import SetupError
from nfl_gsplat.identity.registry import REFEREE_UID, EntityType


def _avatar(n: int = 120, j: int = 22, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "canonical_xyz": rng.normal(0, 0.2, (n, 3)).astype(np.float32),
        "canonical_rot": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "canonical_scale": np.full((n, 3), np.log(0.04), dtype=np.float32),
        "canonical_opacity": np.full((n,), 2.0, dtype=np.float32),
        "canonical_sh": rng.normal(0, 0.1, (n, 3, 1)).astype(np.float32),
        "lbs_weights": np.eye(j)[rng.integers(0, j, n)].astype(np.float32),
    }


class _CountingGenerator:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, uid: str) -> dict[str, np.ndarray]:
        self.calls.append(uid)
        return _avatar(seed=abs(hash(uid)) % 1000)


PLAYER = EntityType.PLAYER.value
REFEREE = EntityType.REFEREE.value
OTHER = EntityType.OTHER.value


def test_select_reference_prefers_large_confident_bbox():
    areas = np.array([100.0, 400.0, 50.0])
    confs = np.array([0.9, 0.8, 0.95])
    # 400*0.8=320 beats 100*0.9=90 and 50*0.95≈47.5.
    assert select_reference_index(areas, confs) == 1


def test_select_reference_respects_conf_gate():
    areas = np.array([1000.0, 10.0])
    confs = np.array([0.2, 0.5])      # first is too low-conf despite huge area
    assert select_reference_index(areas, confs, min_conf=0.4) == 1


def test_select_reference_none_eligible():
    assert select_reference_index(np.array([100.0]), np.array([0.1])) == -1


def test_player_generated_then_cached(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    gen = _CountingGenerator()

    play1 = [("p_A", PLAYER), ("p_B", PLAYER)]
    plan1 = resolve_avatars(play1, lib, gen)
    assert set(plan1.generated) == {"p_A", "p_B"}
    assert plan1.cache_hits == []
    assert gen.calls == ["p_A", "p_B"]

    # Next play: p_A recurs (cache hit, no regen), p_C is new (generated once).
    play2 = [("p_A", PLAYER), ("p_C", PLAYER)]
    plan2 = resolve_avatars(play2, lib, gen)
    assert plan2.cache_hits == ["p_A"]
    assert plan2.generated == ["p_C"]
    assert gen.calls == ["p_A", "p_B", "p_C"], "p_A must not be regenerated"


def test_duplicate_uid_within_play_generated_once(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    gen = _CountingGenerator()
    # Same player on two tracks in one play → one generation.
    plan = resolve_avatars([("p_A", PLAYER), ("p_A", PLAYER)], lib, gen)
    assert gen.calls == ["p_A"]
    assert plan.generated == ["p_A"]


def test_referee_routed_to_generic_and_other_dropped(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    lib.put_referee_avatar(_avatar(seed=99))
    gen = _CountingGenerator()

    entities = [("p_A", PLAYER), (REFEREE_UID, REFEREE), ("", OTHER)]
    plan = resolve_avatars(entities, lib, gen)
    assert plan.referees == [REFEREE_UID]
    assert REFEREE_UID in plan.avatars
    assert "" in plan.dropped
    assert "p_A" in plan.avatars


def test_missing_referee_asset_raises(tmp_path):
    lib = AvatarLibrary(tmp_path / "library", season=2024)
    gen = _CountingGenerator()
    with pytest.raises(SetupError, match="referee avatar"):
        resolve_avatars([(REFEREE_UID, REFEREE)], lib, gen)
