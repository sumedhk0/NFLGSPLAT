"""Detect players per camera for one play -> <play_dir>/tracks.parquet (local).

Feeds player MASKS to calibration (players' bright uniforms otherwise corrupt
hash-mark detection, esp. on the endzone view). --detect-only skips BoT-SORT
(faster on CPU; masking needs boxes, not track IDs).

    python scripts/03b_detect_players.py data/2025/week_04/SEA_at_AZ/play_001 \
        --detect-only --weights yolov8n.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_gsplat.paths import PlayDir
from nfl_gsplat.tracking.detect_track import (
    TrackingConfig, detect_and_track, detect_only, empty_tracks,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir")
    ap.add_argument("--detect-only", action="store_true",
                    help="per-frame detection, no BoT-SORT (faster on CPU)")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence floor; 0.25 recovers small "
                         "distant players that 0.35 drops")
    ap.add_argument("--imgsz", type=int, default=1920,
                    help="inference resolution; ultralytics defaults to 640, "
                         "which downsamples broadcast frames 3x")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cameras", default="sideline,endzone")
    args = ap.parse_args()

    cams = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pd_ = PlayDir.from_dir(args.play_dir, cameras=cams)
    cfg = TrackingConfig(yolo_weights=args.weights, min_detection_conf=args.conf,
                         device=args.device, imgsz=args.imgsz)
    dfs = []
    for cam in pd_.cameras:
        video = pd_.video(cam)
        if not Path(video).exists():
            print(f"skip {cam}: no video at {video}")
            continue
        fn = detect_only if args.detect_only else detect_and_track
        dfs.append(fn(video, cam, cfg))
        print(f"{cam}: {len(dfs[-1])} detections")
    df = pd.concat(dfs, ignore_index=True) if dfs else empty_tracks()
    df.to_parquet(pd_.tracks, index=False)
    print(f"wrote {len(df)} detections -> {pd_.tracks}")


if __name__ == "__main__":
    main()
