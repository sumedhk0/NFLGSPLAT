"""Who was actually on the field for a play, from nflverse.

Identity was previously attempted bottom-up: read jersey numbers off the video
and hope enough of them resolve. Measured yield was 5 usable uids across a play
and ZERO frames carrying the four shared identities the old solver needed.

This inverts it. The league publishes, per play, the 22 players on the field.
Joined to the weekly roster that gives jersey number, position, HEIGHT and
WEIGHT for each -- so the question stops being "who is this?" and becomes
"which of these 22 known players is this track?", which is a far smaller problem
and one where a wrong answer is detectable.

The height column is load-bearing beyond identity. SMPLest-X regresses
near-neutral shape from small crops -- measured betas ~0 and every player
1.62-1.63 m, against a real range here of 67-78 inches (1.70-1.98 m). Per-player
stature cannot come from the pixels, so it comes from here.

Data lives outside the repo (``data/`` is gitignored); ``fetch_nflverse`` pulls
it and is idempotent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
_FILES = {
    "pbp": f"{_BASE}/pbp/play_by_play_{{year}}.parquet",
    "participation": f"{_BASE}/pbp_participation/pbp_participation_{{year}}.parquet",
    "weekly_roster": f"{_BASE}/weekly_rosters/roster_weekly_{{year}}.parquet",
}

INCH_TO_M: float = 0.0254


def fetch_nflverse(year: int, dest: Path | str = "data/nflverse") -> dict[str, Path]:
    """Download the three nflverse tables for ``year``. Idempotent."""
    import urllib.request

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, url in _FILES.items():
        url = url.format(year=year)
        path = dest / Path(url).name
        if not path.exists():
            _LOG.info("fetching %s -> %s", url, path)
            urllib.request.urlretrieve(url, path)
        out[key] = path
    return out


def game_id(season: int, week: int, away: str, home: str) -> str:
    """nflverse game id, e.g. ``2025_04_SEA_ARI``.

    Team codes here are nflverse's, which differ from the ones this project
    uses for directories -- Arizona is ARI upstream and AZ locally.
    """
    fix = {"AZ": "ARI", "LA": "LAR", "WSH": "WAS", "JAX": "JAC"}
    away = fix.get(away.upper(), away.upper())
    home = fix.get(home.upper(), home.upper())
    return f"{season}_{week:02d}_{away}_{home}"


def find_play(pbp: pd.DataFrame, game: str, clip_title: str) -> pd.Series:
    """Locate the play a downloaded clip corresponds to, from its title.

    The All-22 download manifest stores the broadcast play description
    verbatim, e.g. ``(14:54) (Shotgun) 1-K.Murray pass short left to
    18-M.Harrison ...``. That text is the play-by-play ``desc``, so matching is
    exact rather than heuristic -- but the tail is often truncated by the
    filename, so only the leading portion is compared.

    Raises when the match is not UNIQUE. Silently taking the first of several
    would attach the wrong 22 players to a play, and every downstream identity
    would be confidently wrong.
    """
    plays = pbp[pbp["game_id"] == game]
    if plays.empty:
        raise SetupError(
            f"no plays for game {game!r} in the play-by-play table. Check the "
            "season/week/team codes -- nflverse uses ARI, not AZ.")

    clock = re.match(r"\s*\((\d+:\d+)\)", clip_title)
    stem = re.sub(r"^\s*\(\d+:\d+\)\s*", "", clip_title).strip()
    # Compare on a prefix: clip filenames truncate the description.
    probe = stem[:60]
    hits = plays[plays["desc"].astype(str).str.contains(re.escape(probe), na=False)]
    if len(hits) > 1 and clock:
        hits = hits[hits["time"].astype(str) == clock.group(1)]
    if hits.empty:
        raise SetupError(
            f"no play in {game} matches clip title {clip_title[:70]!r}. The "
            "manifest and the play-by-play disagree; do not guess.")
    if len(hits) > 1:
        raise SetupError(
            f"{len(hits)} plays in {game} match {clip_title[:50]!r}. Attaching "
            "the wrong 22 players would make every downstream identity "
            "confidently wrong, so this refuses rather than picking one.")
    return hits.iloc[0]


def players_on_play(participation: pd.DataFrame, roster: pd.DataFrame,
                    game: str, play_id: int, week: int) -> pd.DataFrame:
    """The players on the field, with jersey, position, height and weight.

    Returns one row per player with ``height_m`` added. Raises if the roster
    join loses anyone -- a partial join means some track can never be matched
    to a real player, and it is better to know that here than to wonder later
    why one player never resolves.
    """
    rows = participation[(participation["nflverse_game_id"] == game)
                         & (participation["play_id"] == play_id)]
    if rows.empty:
        raise SetupError(
            f"no participation record for {game} play {play_id}. nflverse does "
            "not publish participation for every season; check coverage before "
            "relying on it.")
    row = rows.iloc[0]
    ids = [i for i in
           f"{row['offense_players']};{row['defense_players']}".split(";")
           if i and i != "None"]

    week_roster = roster[roster["week"] == week]
    got = week_roster[week_roster["gsis_id"].isin(ids)].copy()
    if len(got) != len(ids):
        missing = sorted(set(ids) - set(got["gsis_id"]))
        raise SetupError(
            f"roster join matched {len(got)} of {len(ids)} players for {game} "
            f"play {play_id}; missing gsis_ids {missing[:5]}. Those tracks "
            "could never be identified, so this fails rather than silently "
            "dropping them.")

    got["height_m"] = got["height"].astype(float) * INCH_TO_M
    keep = ["team", "jersey_number", "full_name", "position", "height",
            "weight", "height_m", "gsis_id"]
    return got[[c for c in keep if c in got.columns]].sort_values(
        ["team", "jersey_number"]).reset_index(drop=True)
