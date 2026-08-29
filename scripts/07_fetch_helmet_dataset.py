#!/usr/bin/env python
"""Fetch the NFL Helmet Assignment set: two-camera video with GROUND-TRUTH identity.

This is the only public source found that matches this project's rig AND says who
each player is:

    60 plays, 120 videos -- Endzone and Sideline, synchronised, 1280x720 at 59.94 fps
    train_player_tracking.csv  exactly 22 players per play, ~26 s at 10 Hz,
                               labelled H97 / V23 (team + jersey)
    train_labels.csv           per-frame helmet boxes ALREADY assigned to a player,
                               in both views
    train_baseline_helmets.csv detector output, for comparison

Why it matters here. Every accuracy number this project reports is cross-
CONSISTENCY between two of its own estimates -- two cameras agreeing is not the
same as either being right. This set supplies external truth, so placement,
calibration and pose can be measured rather than corroborated. It also removes
the correspondence problem outright: the same player is labelled in both views,
which is the thing geometry could not recover (measured three ways, all failed:
4.5 m, 3.31 m, and a scatter equal to player spacing).

The Big Data Bowl was checked first and rejected: its 2026 release covers 2023
only, and is a pass-prediction slice with a median of 13 players over a 2.7 s
window -- not the 22 for a whole play that identity needs.

Credentials: a Kaggle token at ~/.kaggle/access_token (or KAGGLE_API_TOKEN), and
the competition rules must be ACCEPTED on the site -- the API 403s otherwise, and
the message does not say why.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

COMP = "nfl-health-and-safety-helmet-assignment"
TABLES = ("train_player_tracking.csv", "train_labels.csv",
          "train_baseline_helmets.csv")


def fetch(remote: str, dest: Path) -> bool:
    """Download one competition file and unzip it if Kaggle wrapped it."""
    final = dest / Path(remote).name
    if final.exists() and final.stat().st_size > 0:
        return True
    cmd = [sys.executable, "-m", "kaggle", "competitions", "download",
           "-c", COMP, "-f", remote, "-p", str(dest)]
    if subprocess.run(cmd, capture_output=True, text=True).returncode != 0:
        return False
    # Kaggle serves single files zipped; unwrap and drop the archive.
    zipped = dest / (Path(remote).name + ".zip")
    if zipped.exists():
        try:
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(dest)
            zipped.unlink()
        except zipfile.BadZipFile:
            return False
    return final.exists() and final.stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/helmet"))
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N videos (a smoke test before the full pull)")
    args = ap.parse_args()

    vid_dir = args.out / "video"
    args.out.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    print("tables:")
    for name in TABLES:
        ok = fetch(name, args.out)
        print(f"   {'ok  ' if ok else 'FAIL'} {name}")
        if not ok and name == "train_labels.csv":
            raise SystemExit(
                "could not fetch train_labels.csv. Accept the rules at "
                f"https://www.kaggle.com/competitions/{COMP}/rules -- the API "
                "returns 403 until you do, without saying so.")

    import pandas as pd
    labels = pd.read_csv(args.out / "train_labels.csv")
    videos = sorted(labels["video"].unique())
    if args.limit:
        videos = videos[:args.limit]
    print(f"\n{len(videos)} videos to fetch into {vid_dir}")

    n_ok = 0
    for i, name in enumerate(videos, 1):
        ok = fetch(f"train/{name}", vid_dir)
        n_ok += ok
        if ok:
            print(f"   [{i:3d}/{len(videos)}] {name}")
        else:
            print(f"   [{i:3d}/{len(videos)}] {name}  FAILED")
    print(f"\n{n_ok}/{len(videos)} videos -> {vid_dir}")
    print(f"tracking + labels -> {args.out}")


if __name__ == "__main__":
    main()
