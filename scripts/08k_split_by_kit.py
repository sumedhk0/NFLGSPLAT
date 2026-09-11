#!/usr/bin/env python
"""Split per-camera tracks where the kit changes for good (tracking.split_by_kit).

Runs on 08b's per-camera tracks BEFORE identity and pairing, so a track
the linker handed from one player to another becomes two tracks with their
own numbers, partners and avatars. Rewrites <play-dir>/tracks.parquet (the
input kept as tracks_unsplit.parquet) and prints the cuts.

Play 1 v22 (2026-09-08): 30 cuts on 151 per-camera tracks -- 13 sideline,
17 endzone -- each a sustained run (>= 15 confident detections) of the
other kit inside one track.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_gsplat.tracking.split_by_kit import KIT_MARGIN, MIN_RUN, WINDOW, split_tracks_by_kit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=KIT_MARGIN)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--min-run", type=int, default=MIN_RUN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    play = args.play_dir
    src = play / "tracks.parquet"
    df = pd.read_parquet(src)
    if "kit_margin" not in df.columns:
        raise SystemExit(f"{src} carries no kit_margin column (08b writes it)")
    out, cuts = split_tracks_by_kit(df, margin=args.margin, window=args.window, min_run=args.min_run)
    by_cam = {}
    for cam, *_ in cuts:
        by_cam[cam] = by_cam.get(cam, 0) + 1
    print(f"kit split: {len(cuts)} cuts on {int((df['track_id'] >= 0).sum() and df[df['track_id'] >= 0].groupby(['cam', 'track_id']).ngroups)} "
          f"per-camera tracks ({', '.join(f'{c} {n}' for c, n in sorted(by_cam.items()))}); "
          f"ids {df['track_id'].nunique()} -> {out['track_id'].nunique()}")
    for cam, tid, frame, new in cuts[:20]:
        print(f"   {cam} track {tid} -> {new} from frame {frame}")
    if args.dry_run:
        return
    backup = play / "tracks_unsplit.parquet"
    if not backup.exists():
        src.replace(backup)
        print(f"kept the input as {backup.name}")
    out.to_parquet(src, index=False)
    print(f"wrote {src} ({len(out)} rows)")


if __name__ == "__main__":
    main()
