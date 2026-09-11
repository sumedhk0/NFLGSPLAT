#!/usr/bin/env python
"""Fetch a season's rosters straight from the nflverse release files.

scripts/fetch_roster.py goes through nfl_data_py, which pins a pandas that
will not build on Python 3.12 or 3.14, and keeps only five columns. Identity
for the All-22 pipeline (scripts/08c) wants the weekly roster -- who wore a
number in THIS week, not whoever wore it last -- plus name, height and weight
for the avatar's body. nflverse publishes both files as parquet; this writes
them where 08c looks:

    data/rosters/{season}/rosters.parquet         one row per player-week
    data/rosters/{season}/roster_weekly.parquet   the same weekly file

The season file is deliberately the FULL weekly table (not deduplicated), so
that fetch_roster.py's five-column output and this one never get confused:
08c sorts by week and takes the latest row per (team, number) when no --week
is given, and filters to the week when it is.

    python scripts/fetch_nflverse_rosters.py --season 2024
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"


def fetch(url: str) -> pd.DataFrame:
    data = urllib.request.urlopen(url, timeout=180).read()
    return pd.read_parquet(io.BytesIO(data))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/rosters"))
    args = ap.parse_args()

    out = args.out / str(args.season)
    out.mkdir(parents=True, exist_ok=True)
    weekly = fetch(f"{RELEASE}/weekly_rosters/roster_weekly_{args.season}.parquet")
    needed = {"season", "week", "team", "jersey_number", "full_name", "position",
              "height", "weight", "gsis_id"}
    missing = needed - set(weekly.columns)
    if missing:
        raise SystemExit(f"nflverse weekly roster lacks {sorted(missing)}; "
                         "the release schema changed, look before adapting.")
    weekly.to_parquet(out / "roster_weekly.parquet", index=False)
    weekly.to_parquet(out / "rosters.parquet", index=False)
    print(f"{len(weekly)} player-weeks, weeks {int(weekly.week.min())}-"
          f"{int(weekly.week.max())}, {weekly.team.nunique()} teams -> {out}")


if __name__ == "__main__":
    main()
