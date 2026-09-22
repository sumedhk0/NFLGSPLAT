#!/usr/bin/env python
"""Find camera tracks that die through contact and the tracks born on the same man moments later, score the pairs
(tracking.contact_links) and, with --apply, fold them through 08za / 08z (per camera track, with backups).

WHY. The 2026-09-22 receiver case (id 9 -> 77 through the defender 6 at 489-494) was found and fixed by hand from the
09c table; this is that reading as a tool, so the next play's contact switches are proposed from the tables instead of
typed. Read the table before applying: every score is a heuristic.

USAGE (nflgsplat env for the table; --apply shells out to the smplx venv for 08za/08z because they rewrite the
numpy-1 pickles):
  python scripts/09e_contact_links.py --play-dir P [--cam sideline] [--frames 395 607] [--min-score 2] [--apply]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.render.play_timeline import _teams, box_ground, clip_offset  # noqa: E402
from nfl_gsplat.tracking import contact_links as cl  # noqa: E402

PYS = "C:/venvs/smplx312/Scripts/python.exe"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--cam", default="sideline", choices=["sideline", "endzone"])
    ap.add_argument("--frames", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the play window (default: play_end.json's snap and end)")
    ap.add_argument("--min-score", type=float, default=cl.MIN_SCORE)
    ap.add_argument("--max-gap", type=int, default=cl.MAX_GAP)
    ap.add_argument("--max-m", type=float, default=cl.MAX_M)
    ap.add_argument("--apply", action="store_true", help="fold every un-rejected link at or over --min-score")
    a = ap.parse_args()
    P = Path(a.play_dir)
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0].copy()
    offset = clip_offset(P)
    shift = {"endzone": offset} if offset else None
    if offset and "endzone" in set(df["cam"].unique()):
        df.loc[df["cam"] == "endzone", "frame"] = df.loc[df["cam"] == "endzone", "frame"].astype(int) - offset
    if a.frames is None:
        pe = json.load(open(P / "play_end.json")) if (P / "play_end.json").exists() else {}
        lo, hi = int(pe.get("start", 300)), int(pe.get("end", 460))
    else:
        lo, hi = a.frames
    tracks = load_camera_track(P / "cameras.npz")
    ground = box_ground(df, tracks, frame_shift=shift)
    teams = _teams(P)
    links = cl.find_links(df, ground, teams, cam=a.cam, lo=lo, hi=hi, max_gap=a.max_gap, max_m=a.max_m)
    print(f"{a.cam}: window {lo}-{hi}, {len(links)} candidate pairs")
    for l in links:
        tag = "REJECT" if l.reject else ("LINK  " if l.score >= a.min_score else "weak  ")
        print(f"{tag} score {l.score:4.1f}  {l.keep:4d} <- {l.drop:4d} (track {l.track}): death {l.death_last}, birth "
              f"{l.birth_first}, overlap {l.overlap}, {l.dist_m} m | " + "; ".join(l.reasons) + (f" | {l.reject}" if l.reject else ""))
    chosen = [l for l in links if l.reject is None and l.score >= a.min_score]
    if not chosen:
        return
    print("\nto apply:")
    for l in chosen:
        for cmd in cl.fold_commands(l, str(P)):
            print("  " + " ".join(cmd))
    if not a.apply:
        print("dry run. Re-run with --apply.")
        return
    for l in chosen:
        for cmd in cl.fold_commands(l, str(P)):
            subprocess.run([PYS] + cmd, check=True)


if __name__ == "__main__":
    main()
