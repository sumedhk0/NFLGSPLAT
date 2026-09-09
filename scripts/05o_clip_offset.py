#!/usr/bin/env python
"""The endzone clip's frame offset from the players who move (pose.clip_offset).

    python scripts/05o_clip_offset.py --play-dir <P> [--range -40 10] [--speed-min 3.0]

Writes <play-dir>/clip_offset.json {"offset": int, "curve": {...}}: sideline frame f
sits beside endzone frame f + offset. 05n (--offset) and the re-pairing (08i --lag)
read it. Refuses when the curve has no minimum inside the range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.pose.clip_offset import SPEED_MIN_MPS, ground_speeds, miss_by_offset, solve_offset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--range", type=int, nargs=2, default=(-40, 10))
    ap.add_argument("--speed-min", type=float, default=SPEED_MIN_MPS)
    ap.add_argument("--coarse", type=int, default=3, help="first pass stride; the second pass is every frame around the best")
    args = ap.parse_args()
    P = args.play_dir
    tracks = load_camera_track(P / "cameras.npz")
    fps = float(np.load(P / "cameras.npz", allow_pickle=True)["fps"])
    tdf = pd.read_parquet(P / "tracks.parquet")
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    moving = ground_speeds(tdf, tracks["sideline"], fps=fps, speed_min=args.speed_min)
    print(f"{len(moving)} (frame, player) with sideline ground speed >= {args.speed_min} m/s")
    lo, hi = args.range
    curve = miss_by_offset(kdf, tracks["sideline"], tracks["endzone"], moving, range(lo, hi + 1, args.coarse))
    best = solve_offset(curve)
    fine = miss_by_offset(kdf, tracks["sideline"], tracks["endzone"], moving, range(best - args.coarse, best + args.coarse + 1))
    curve.update(fine)
    best = solve_offset(curve)
    print("miss by offset (m): " + ", ".join(f"{o:+d}: {curve[o][0]:.3f}" for o in sorted(curve) if np.isfinite(curve[o][0])))
    print(f"endzone clip offset {best:+d} frames ({best / fps:+.3f} s): miss {curve[best][0]:.3f} m over {curve[best][1]} pairs")
    (P / "clip_offset.json").write_text(json.dumps({"offset": int(best), "fps": fps, "speed_min": args.speed_min,
                                                    "curve": {str(o): curve[o] for o in sorted(curve)}}, indent=1))
    print(f"wrote {P / 'clip_offset.json'}")


if __name__ == "__main__":
    main()
