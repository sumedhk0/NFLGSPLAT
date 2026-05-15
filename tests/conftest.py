"""Shared pytest fixtures. Kept deliberately thin — each module owns its data."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def generated_dir(fixtures_dir: Path) -> Path:
    out = fixtures_dir / "generated"
    out.mkdir(parents=True, exist_ok=True)
    return out
