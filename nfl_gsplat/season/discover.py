"""Discover the plays of a season by walking the on-disk tree.

The filesystem is the source of truth: every directory matching
``data/{season}/week_NN/{away}_at_{home}/play_NNN`` that contains both clips and
a ``meta.yaml`` is a play. Drives the season DAG (no explicit games manifest).
"""
from __future__ import annotations

from pathlib import Path

from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

_REQUIRED = ("sideline.mp4", "endzone.mp4", "meta.yaml")


def discover_plays(data_root, season, *, cameras=("sideline", "endzone")) -> list[PlayDir]:
    """Return the season's plays as :class:`PlayDir`s, ordered week→matchup→play.

    Directories missing any required file (both clips + meta.yaml) are skipped
    with a warning so a half-uploaded play never silently enters the DAG.
    """
    root = Path(data_root) / str(season)
    plays: list[PlayDir] = []
    for play in sorted(root.glob("week_*/*_at_*/play_*")):
        if not play.is_dir():
            continue
        missing = [f for f in _REQUIRED if not (play / f).exists()]
        if missing:
            _LOG.warning(f"discover: skipping {play} (missing {missing})")
            continue
        plays.append(PlayDir.from_dir(play, cameras=tuple(cameras)))
    return plays
