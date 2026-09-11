#!/usr/bin/env python
"""Refine a play-dir's ENDZONE camera track to its own paint (calibration.endzone_paint).

The endzone camera exported by 08/08b comes from the players' feet against the
sideline's placement, with the mount centre a grid prior. Measured on play 1
(2026-09-09) its grid sits 40-85 px off the painted lines and the two cameras'
rays through the same keypoint miss by 0.2-0.5 m; every two-view pose defect
traced back to it. This solves the mount centre once from the paint of a sample
of frames, then every frame's rotation and focal, and judges the result with
the PLAYERS (the independent ruler): the sideline's and the endzone's rays
through the same keypoints must meet closer than before, triangulated ankles
must come down to the turf. It refuses to write when they do not.

    python scripts/08l_endzone_paint.py --play-dir <P> [--out <npz>] [--apply]

--out writes the refined cameras to a side file (measure first); --apply
rewrites cameras.npz (the original kept as cameras_endzone_players.npz).
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration import endzone_paint as ep
from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
from nfl_gsplat.errors import CalibrationError


def boxes_by_frame(tracks_df: pd.DataFrame, cam: str) -> dict:
    out: dict = {}
    sub = tracks_df[tracks_df["cam"] == cam]
    for f, g in sub.groupby("frame"):
        out[int(f)] = [tuple(map(float, b)) for b in g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--cam", default="endzone")
    ap.add_argument("--other", default="sideline", help="the trusted camera the players are judged against")
    ap.add_argument("--out", type=Path, default=None, help="write the refined cameras here (default <play-dir>/cameras_endzone_paint.npz)")
    ap.add_argument("--apply", action="store_true", help="rewrite cameras.npz (original kept as cameras_endzone_players.npz)")
    ap.add_argument("--centre-frames", type=int, default=12)
    ap.add_argument("--centre", type=float, nargs=3, default=None, help="hold the mount centre at this x y z instead of solving it")
    ap.add_argument("--offset", type=int, default=None, help="endzone frame offset for the player ruler (default: poses_tri.json's, else 0)")
    ap.add_argument("--ruler-stride", type=int, default=2)
    args = ap.parse_args()
    P = args.play_dir

    tracks = load_camera_track(P / "cameras.npz")
    track = tracks[args.cam]
    tdf = pd.read_parquet(P / "tracks.parquet")
    boxes = boxes_by_frame(tdf, args.cam)
    video = P / f"{args.cam}.mp4"

    refined, stats = ep.refine_track(video, track, boxes, centre_frames=args.centre_frames, centre=args.centre)
    before = np.asarray(stats["before"], float)
    after = np.asarray(stats["after"], float)
    dash = np.asarray(stats["dash"], float)
    print(f"{args.cam}: mount centre {np.round(stats['centre'], 1)}; {stats['n_refined']}/{len(stats['frames'])} frames refined; "
          f"line px median {np.nanmedian(before):.1f} -> {np.nanmedian(after):.1f}, dash px {np.nanmedian(dash):.1f}; "
          f"segments/frame {np.median(stats['n_seg']):.0f}, dashes/frame {np.median(stats['n_dash']):.0f}")
    stats_csv = (args.out or (P / "cameras_endzone_paint.npz")).with_suffix(".stats.csv")
    pd.DataFrame({k: stats[k] for k in ("frames", "before", "after", "dash", "n_seg", "n_dash")}).to_csv(stats_csv, index=False)
    print(f"per-frame paint stats: {stats_csv}")
    if not (np.nanmedian(after) < np.nanmedian(before)):
        raise CalibrationError("the paint refinement did not lower the line distance; cameras untouched")

    # the players: independent of the paint
    kp = P / "keypoints_2d.parquet"
    verdict = None
    if kp.exists():
        offset = args.offset
        if offset is None:
            offset = 0
            if (P / "poses_tri.json").exists():
                offset = int(pickle.load(open(P / "poses_tri.json", "rb")).get("offset", 0))
        kdf = pd.read_parquet(kp)
        other = tracks[args.other]
        r0 = ep.player_rulers(other, track, kdf, offset=offset, stride=args.ruler_stride)
        r1 = ep.player_rulers(other, refined, kdf, offset=offset, stride=args.ruler_stride)
        m0, m1 = np.nanmedian(r0["miss_p50"]), np.nanmedian(r1["miss_p50"])
        a0, a1 = np.nanmedian(r0["ankle_z"]), np.nanmedian(r1["ankle_z"])
        h0, h1 = np.nanmedian(r0["hip_z"]), np.nanmedian(r1["hip_z"])
        print(f"players ({len(r1['frames'])} frames, endzone offset {offset:+d}): ray miss p50 {m0:.3f} -> {m1:.3f} m "
              f"(p90 of frames {np.nanpercentile(r0['miss_p50'], 90):.3f} -> {np.nanpercentile(r1['miss_p50'], 90):.3f}); "
              f"ankle z {a0:+.2f} -> {a1:+.2f} m; hip z {h0:.2f} -> {h1:.2f} m")
        verdict = (m1 < m0) and (abs(a1 - 0.08) < abs(a0 - 0.08))
        if not verdict:
            raise CalibrationError("the players do not confirm the refinement (the ray miss did not drop or the ankles "
                                   "did not come down to the turf); cameras untouched")
    else:
        print("no keypoints_2d.parquet: the players cannot judge this refinement; writing on the paint alone")

    tracks[args.cam] = refined
    fps = float(np.load(P / "cameras.npz", allow_pickle=True)["fps"])
    if args.apply:
        backup = P / "cameras_endzone_players.npz"
        if not backup.exists():
            shutil.copy(P / "cameras.npz", backup)
        write_camera_track(P / "cameras.npz", tracks, fps=fps)
        print(f"cameras.npz rewritten ({args.cam} refined to its paint; original in {backup.name})")
    else:
        out = args.out or (P / "cameras_endzone_paint.npz")
        write_camera_track(out, tracks, fps=fps)
        print(f"refined {args.cam} written to {out}; cameras.npz untouched")


if __name__ == "__main__":
    main()
