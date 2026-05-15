"""PLY merge + I/O tests. No torch, no gsplat."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nfl_gsplat.compositing.merge_ply import (
    GaussianBatch,
    batch_from_arrays,
    load_gaussian_ply,
    merge_batches,
    save_gaussian_ply,
)
from nfl_gsplat.field.train_field import write_mock_field_ply


def _make_batch(n: int, sh_degree: int = 0, seed: int = 0) -> GaussianBatch:
    rng = np.random.default_rng(seed)
    k = (sh_degree + 1) ** 2
    return batch_from_arrays(
        xyz=rng.normal(0, 1, (n, 3)).astype(np.float32),
        rot=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        scale=np.full((n, 3), np.log(0.05), dtype=np.float32),
        opacity=np.full((n,), 2.0, dtype=np.float32),
        sh=rng.normal(0, 0.1, (n, 3, k)).astype(np.float32),
    )


def test_merge_preserves_total_gaussian_count():
    a = _make_batch(100, sh_degree=0, seed=1)
    b = _make_batch(200, sh_degree=0, seed=2)
    c = _make_batch(300, sh_degree=0, seed=3)
    merged = merge_batches([a, b, c])
    assert merged.num_gaussians == 600


def test_merge_trims_to_minimum_sh_degree():
    low = _make_batch(50, sh_degree=0, seed=1)
    high = _make_batch(60, sh_degree=3, seed=2)
    merged = merge_batches([low, high])
    assert merged.sh_degree == 0
    assert merged.sh.shape == (110, 3, 1)


def test_merge_rejects_nan_input():
    bad = _make_batch(10, sh_degree=0, seed=0)
    bad.xyz[0, 0] = np.nan
    good = _make_batch(5, sh_degree=0, seed=1)
    with pytest.raises(ValueError, match="non-finite"):
        merge_batches([bad, good])


def test_ply_roundtrip_preserves_arrays(tmp_path: Path):
    batch = _make_batch(250, sh_degree=2, seed=7)
    out = tmp_path / "x.ply"
    save_gaussian_ply(out, batch)
    loaded = load_gaussian_ply(out)
    assert loaded.num_gaussians == 250
    assert loaded.sh_degree == 2
    np.testing.assert_allclose(loaded.xyz, batch.xyz, atol=1e-6)
    np.testing.assert_allclose(loaded.rot, batch.rot, atol=1e-6)
    np.testing.assert_allclose(loaded.scale, batch.scale, atol=1e-6)
    np.testing.assert_allclose(loaded.opacity, batch.opacity, atol=1e-6)
    np.testing.assert_allclose(loaded.sh, batch.sh, atol=1e-6)


def test_mock_field_ply_loads_as_gaussian_batch(tmp_path: Path):
    ply = write_mock_field_ply(tmp_path / "field.ply", num_gaussians=10_000, seed=0)
    batch = load_gaussian_ply(ply)
    assert batch.num_gaussians == 10_000
    assert batch.sh_degree == 0
    assert batch.sh.shape == (10_000, 3, 1)


def test_smoke_style_merge_counts(tmp_path: Path):
    """Plan contract: composite PLY has ``field_N + 3 × avatar_N + 1 × ball_N``."""
    field = _make_batch(60_000, sh_degree=0, seed=1)
    avatar = _make_batch(3_000, sh_degree=0, seed=2)
    ball = _make_batch(500, sh_degree=0, seed=3)
    merged = merge_batches([field, avatar, avatar, avatar, ball])
    assert merged.num_gaussians == 60_000 + 3 * 3_000 + 500
