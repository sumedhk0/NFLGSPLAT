"""Detect+track both cameras, OCR jerseys, assign geometry-free player_uid ->
tracks.parquet (with track_id, jersey_number_ocr, team, player_uid).

    python scripts/03c_identity_tracks.py data/2025/week_04/SEA_at_AZ/play_001 \
        --weights yolov8n.pt --season 2025      # env: nfl_smplx (PaddleOCR, GPU)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from nfl_gsplat.calibration.identity_precompute import assign_identity_columns
from nfl_gsplat.paths import PlayDir
from nfl_gsplat.tracking.detect_track import TrackingConfig, detect_and_track
from nfl_gsplat.tracking.jersey_ocr import JerseyOCRConfig, vote_jersey_numbers


def _crop_provider(pd_):
    caps = {cam: cv2.VideoCapture(str(pd_.video(cam))) for cam in pd_.cameras}

    def provider(cam, frame, track_id, _df=None):
        cap = caps[cam]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = cap.read()
        return img if ok else None

    return provider


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("play_dir")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--cameras", default="sideline,endzone")
    args = ap.parse_args()
    cams = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    pd_ = PlayDir.from_dir(args.play_dir, cameras=cams)
    cfg = TrackingConfig(yolo_weights=args.weights, device=args.device)
    dfs = [detect_and_track(pd_.video(cam), cam, cfg) for cam in pd_.cameras
           if Path(pd_.video(cam)).exists()]
    df = pd.concat(dfs, ignore_index=True)
    df = vote_jersey_numbers(df, {cam: pd_.video(cam) for cam in pd_.cameras},
                             JerseyOCRConfig())      # fills jersey_number_ocr
    df = assign_identity_columns(df, _crop_provider(pd_), season=args.season)
    df.to_parquet(pd_.tracks, index=False)
    print(f"wrote {len(df)} rows with player_uid -> {pd_.tracks}")


if __name__ == "__main__":
    main()
