"""Fetch the roster / participation prior for a season via nflverse.

Writes ``data/rosters/{season}/rosters.parquet`` (and ``participation.parquet``
when available) in the schema :class:`nfl_gsplat.identity.roster.RosterSource`
expects:

    rosters.parquet:       gsis_id, team, jersey_number, position, player_name
    participation.parquet: game_id, play_id, player_uid   (one row per on-field player)

The roster turns player recognition into constrained classification (see
nfl_gsplat.identity). It is optional — without it the pipeline falls back to
``OcrOnlySource`` — but strongly recommended.

``nfl_data_py`` is imported lazily; this script needs network access and is run
once per season, not inside the per-play pipeline. Participation coverage varies
by season; when the export schema is unrecognized we still write rosters and
warn (the identity layer then uses the full per-game roster as candidates).

Usage::

    python scripts/fetch_roster.py --season 2024
"""
from __future__ import annotations

from pathlib import Path

import typer

from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _load_nfl_data_py():
    try:
        import nfl_data_py  # type: ignore
    except ImportError as e:
        raise SetupError(
            "nfl_data_py not installed. `pip install nfl_data_py` in the "
            "nfl_smplx env, then rerun. See SETUP.md §9."
        ) from e
    return nfl_data_py


@app.command()
def main(
    season: int = typer.Option(..., help="Season year, e.g. 2024"),
    out_root: Path = typer.Option(Path("data/rosters")),
) -> None:
    nfl = _load_nfl_data_py()
    out_dir = out_root / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- rosters ------------------------------------------------------------
    _LOG.info(f"fetching weekly rosters for {season} …")
    rosters = nfl.import_weekly_rosters([season])
    # Normalize to our schema; nflverse column names are stable enough to map.
    colmap = {
        "player_id": "gsis_id", "gsis_id": "gsis_id",
        "team": "team", "recent_team": "team",
        "jersey_number": "jersey_number",
        "position": "position",
        "player_name": "player_name", "full_name": "player_name",
    }
    present = {src: dst for src, dst in colmap.items() if src in rosters.columns}
    norm = rosters.rename(columns=present)
    keep = [c for c in ("gsis_id", "team", "jersey_number", "position", "player_name")
            if c in norm.columns]
    norm = norm[keep].dropna(subset=["team", "jersey_number"]).drop_duplicates()
    roster_path = out_dir / "rosters.parquet"
    norm.to_parquet(roster_path, index=False)
    _LOG.info(f"wrote {len(norm)} roster rows → {roster_path}")

    # --- participation (optional) ------------------------------------------
    try:
        _LOG.info(f"fetching participation for {season} …")
        part = nfl.import_participation([season])
    except Exception as exc:  # noqa: BLE001 — participation is best-effort
        _LOG.warning(f"participation unavailable ({exc}); rosters-only candidates.")
        return

    rows = _explode_participation(part)
    if rows is None:
        _LOG.warning("participation schema unrecognized; rosters-only candidates.")
        return
    part_path = out_dir / "participation.parquet"
    rows.to_parquet(part_path, index=False)
    _LOG.info(f"wrote {len(rows)} participation rows → {part_path}")


def _explode_participation(part):
    """Best-effort map of nflverse participation → (game_id, play_id, player_uid).

    nflverse stores on-field players as space-separated gsis-id strings in
    ``offense_players`` / ``defense_players``. Returns a long DataFrame or None
    if the expected columns are absent.
    """
    import pandas as pd

    cols = set(part.columns)
    gid = "nflverse_game_id" if "nflverse_game_id" in cols else ("game_id" if "game_id" in cols else None)
    if gid is None or "play_id" not in cols:
        return None
    player_cols = [c for c in ("offense_players", "defense_players") if c in cols]
    if not player_cols:
        return None

    records: list[dict] = []
    for _, row in part.iterrows():
        for pc in player_cols:
            val = row[pc]
            if not isinstance(val, str) or not val:
                continue
            for uid in val.split(";") if ";" in val else val.split():
                uid = uid.strip()
                if uid:
                    records.append({"game_id": str(row[gid]), "play_id": str(row["play_id"]),
                                    "player_uid": uid})
    return pd.DataFrame.from_records(records) if records else None


if __name__ == "__main__":
    app()
