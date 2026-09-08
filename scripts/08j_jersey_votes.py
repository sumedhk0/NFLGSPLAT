#!/usr/bin/env python
"""Jersey OCR vote counts per (cam, global id) -> <play-dir>/jersey_votes.json.

For a play whose identity cache predates the vote columns: the OCR is run
again on the SAME detections under the global ids (no re-pairing, so the
pose caches keyed by id stay valid) and only the counts are written; 08c
--from-cache reads them as the evidence for identity.exclusive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_gsplat.tracking.jersey_ocr import JerseyOCRConfig, vote_jersey_numbers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--ocr-backend", default="easyocr")
    ap.add_argument("--ocr-top-k", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    play = args.play_dir
    df = pd.read_parquet(play / "tracks_identity.parquet")
    df = df[df["track_id"] >= 0].reset_index(drop=True)
    videos = {cam: play / f"{cam}.mp4" for cam in ("sideline", "endzone")}
    cfg = JerseyOCRConfig(backend=args.ocr_backend, top_k_frames=args.ocr_top_k, use_gpu=not args.cpu)
    votes: dict = {}
    out = vote_jersey_numbers(df, videos, cfg, votes_out=votes)
    agree = 0
    n = 0
    for (cam, tid), counts in votes.items():
        if not counts:
            continue
        n += 1
        prev = df[(df["cam"] == cam) & (df["track_id"] == tid)]["jersey_number_ocr"]
        if len(prev) and int(prev.iloc[0]) == max(counts, key=counts.get):
            agree += 1
    with open(play / "jersey_votes.json", "w") as fh:
        json.dump({f"{cam},{tid}": counts for (cam, tid), counts in votes.items()}, fh)
    print(f"jersey votes on {len(votes)} (cam, id) keys, {n} with a read; the winner agrees with the cached number "
          f"on {agree} of {n}; wrote {play / 'jersey_votes.json'}")


if __name__ == "__main__":
    main()
