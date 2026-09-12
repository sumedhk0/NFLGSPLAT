#!/usr/bin/env python
"""Fill the frames a track loses, from detections below the detector's threshold (tracking.gap_fill).

    python scripts/08q_fill_track_gaps.py --play-dir <P> [--conf 0.08] [--dry-run]

Eleven a side is the check that needs no labels, and play 1's sideline holds 8 of Kansas
City's 11 at the snap: the detector loses the linemen in the pile. It does see them --
at frame 300 the boxes that appear only below 0.35 are a lineman at confidence 0.17 --
but lowering the threshold everywhere triples the detections once the camera pans and
the crowd enters frame. So the low-confidence pool is used only to fill a gap INSIDE a
track's own life, near where that track must be. No new identities.

Writes the extra rows into tracks.parquet and tracks_identity.parquet (originals kept as
*_nogaps.parquet), marked with filled = True. The keypoints (05m) must be re-run for the
new boxes afterwards, then the fits.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.tracking.gap_fill import MAX_GAP, MIN_IOU, fill_gaps


def low_conf_pool(video, frames, *, conf: float, keep_above: float, imgsz: int = 1920):
    """``{frame: boxes [N, 4]}``: person detections between ``conf`` and ``keep_above`` -- the ones
    the tracker's threshold threw away."""
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov8x.pt")
    cap = cv2.VideoCapture(str(video))
    pool = {}
    for f in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        r = model.predict(img, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.conf.cpu().numpy()
        take = (c >= conf) & (c < keep_above)
        if take.any():
            pool[int(f)] = b[take]
    cap.release()
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.08, help="floor for the low-confidence pool")
    ap.add_argument("--keep-above", type=float, default=0.35, help="the tracker's own threshold")
    ap.add_argument("--cams", nargs="*", default=["sideline", "endzone"])
    ap.add_argument("--max-gap", type=int, default=MAX_GAP)
    ap.add_argument("--min-iou", type=float, default=MIN_IOU)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    df = pd.read_parquet(P / "tracks.parquet")
    add_all = []
    for cam in args.cams:
        g = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
        if not len(g):
            continue
        # only the frames some track is missing are worth detecting on
        want = set()
        from nfl_gsplat.tracking.gap_fill import track_gaps
        for _pid, rows in g.groupby("track_id"):
            fr = rows["frame"].to_numpy(int)
            for fa, fb in track_gaps(fr, max_gap=args.max_gap):
                want.update(range(fa + 1, fb))
        if not want:
            print(f"{cam}: no interior gaps")
            continue
        print(f"{cam}: {len(want)} frames have a track missing; detecting there at {args.conf}-{args.keep_above}")
        pool = low_conf_pool(P / f"{cam}.mp4", want, conf=args.conf, keep_above=args.keep_above)
        add, counts = fill_gaps(df, pool, cam=cam, max_gap=args.max_gap, min_iou=args.min_iou)
        n_pool = sum(len(v) for v in pool.values())
        print(f"{cam}: {n_pool} rejected detections in those frames, {len(add)} taken by a track "
              f"({len(counts)} tracks; the busiest: "
              + ", ".join(f"{p}:{n}" for p, n in sorted(counts.items(), key=lambda kv: -kv[1])[:6]) + ")")
        add_all.extend(add)
    if not add_all:
        print("nothing to fill")
        return
    if args.dry_run:
        print("dry run; no table written")
        return
    extra = pd.DataFrame(add_all)
    for name in ("tracks.parquet", "tracks_identity.parquet"):
        path = P / name
        if not path.exists():
            continue
        t = pd.read_parquet(path)
        backup = path.with_name(path.stem + "_nogaps" + path.suffix)
        if not backup.exists():
            shutil.copy(path, backup)
        e = extra.copy()
        for col in t.columns:
            if col not in e:
                e[col] = np.nan
        if "global_player_id" in t.columns:
            e["global_player_id"] = e["track_id"]
        out = pd.concat([t, e[t.columns]], ignore_index=True).sort_values(["cam", "frame", "track_id"])
        out.to_parquet(path, index=False)
        print(f"{name}: {len(t)} -> {len(out)} rows; kept {backup.name}")
    print("re-run 05m for the keypoints of the new boxes, then the fits")


if __name__ == "__main__":
    main()
