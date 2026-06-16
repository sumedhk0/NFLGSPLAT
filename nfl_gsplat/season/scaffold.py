"""Scaffold a new play folder + meta.yaml stub. Core for scripts/new_play.py."""
from __future__ import annotations

from pathlib import Path

from nfl_gsplat.paths import PlayDir
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


def scaffold_play(
    data_root,
    *,
    season,
    week: int,
    away: str,
    home: str,
    play: str,
    fps: float = 30.0,
    gsis_play_id: str | None = None,
    force: bool = False,
) -> PlayDir:
    """Create ``data/{season}/week_NN/{away}_at_{home}/{play}/`` + a meta.yaml stub.

    Raises :class:`FileExistsError` if meta.yaml already exists and ``force`` is
    False. Returns the resolved :class:`PlayDir`. The user drops the two clips
    into ``pd.dir`` afterward.
    """
    pd = PlayDir(season=str(season), week=int(week), matchup=f"{away}_at_{home}",
                 play_id=str(play), data_root=Path(data_root))
    pd.dir.mkdir(parents=True, exist_ok=True)
    if pd.meta_yaml.exists() and not force:
        raise FileExistsError(
            f"{pd.meta_yaml} already exists; pass force=True to overwrite."
        )
    lines = [
        f"season: {season}",
        f"week: {int(week)}",
        f"home_team: {home}",
        f'away_team: "{away}"',
        f"fps: {fps}",
    ]
    if gsis_play_id is not None:
        lines.append(f"gsis_play_id: {gsis_play_id}")
    pd.meta_yaml.write_text("\n".join(lines) + "\n")
    _LOG.info(f"scaffolded play → {pd.dir} (drop sideline.mp4 + endzone.mp4 here)")
    return pd
