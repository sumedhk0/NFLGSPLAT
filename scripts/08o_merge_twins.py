#!/usr/bin/env python
"""Fold two tracker ids that hold ONE body into one id (tracking.twins).

    C:/venvs/smplx312/Scripts/python scripts/08o_merge_twins.py --play-dir <P> [--dry-run]

The detector fires twice on a body in a pile and the render stands two avatars on one
man. Two ids of one camera are the same body when their ankle rays land on the same spot
on the turf (0.03-0.11 m on play 1 against 0.80 m for the next same-team pair), their
boxes overlap, and they are the same team. The longer track's id survives; where both
boxes exist on a frame the more confident one does.

Rewrites tracks.parquet (input kept as tracks_untwinned.parquet) and carries
keypoints_2d.parquet and poses_{sideline,endzone}.json to the merged ids by their boxes
(originals kept as *_pretwin.*). Must run under numpy 1 (smplx312): the pose caches are
numpy-1 pickles. Re-run 08c --from-cache and 08n afterwards -- the ids changed.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.render.play_timeline import ankle_ground
from nfl_gsplat.tracking.relabel import assert_numpy1_for_pickles, relabel_keypoints, relabel_pose_cache
from nfl_gsplat.tracking.twins import MAX_ANKLE_GAP_M, MIN_ANKLE_FRAMES, MIN_IOU, apply_merge, merge_map, twin_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--max-gap", type=float, default=MAX_ANKLE_GAP_M, help="metres between the two ids' feet")
    ap.add_argument("--min-frames", type=int, default=MIN_ANKLE_FRAMES)
    ap.add_argument("--min-iou", type=float, default=MIN_IOU)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    assert_numpy1_for_pickles()
    df = pd.read_parquet(P / "tracks.parquet")
    tracks = load_camera_track(P / "cameras.npz")
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    ankles = ankle_ground(kdf, tracks)
    ident = pickle.load(open(P / "identity_resolved.pkl", "rb")).get("merged", {})
    teams = {int(k): getattr(v, "team", None) for k, v in ident.items()}
    names = {int(k): str(getattr(v, "player", "?")) for k, v in ident.items()}

    all_pairs = []
    for cam in sorted(df["cam"].unique()):
        pairs = twin_pairs(df, ankles, teams, cam=str(cam), max_gap_m=args.max_gap,
                           min_frames=args.min_frames, min_iou=args.min_iou)
        for keep, drop, gap, iou, n in pairs:
            print(f"{cam}: ids {keep} and {drop} are one body -- feet {gap:.2f} m apart over {n} frames, boxes "
                  f"{iou:.2f} IoU ({names.get(keep, '?')} / {names.get(drop, '?')})")
        all_pairs.extend(pairs)
    if not all_pairs:
        print("no twins found; nothing to do")
        return
    # ids are global: a merge found in one camera moves the other camera's rows too
    mapping, n_boxes = merge_map(df, all_pairs)
    out, n_rows = apply_merge(df, mapping)
    print(f"{len(all_pairs)} pairs merged; {n_rows} boxes dropped (two boxes of one camera under one id), "
          f"{out['track_id'].nunique()} ids left of {df[df['track_id'] >= 0]['track_id'].nunique()}")
    if args.dry_run:
        print("dry run; nothing written")
        return
    for name, backup in (("tracks.parquet", "tracks_untwinned.parquet"),
                         ("keypoints_2d.parquet", "keypoints_2d_pretwin.parquet"),
                         ("poses_sideline.json", "poses_sideline_pretwin.json"),
                         ("poses_endzone.json", "poses_endzone_pretwin.json")):
        src = P / name
        if src.exists() and not (P / backup).exists():
            shutil.copy(src, P / backup)
    out.to_parquet(P / "tracks.parquet", index=False)
    if (P / "tracks_identity.parquet").exists():
        out.to_parquet(P / "tracks_identity.parquet", index=False)
    kout, kdrop = relabel_keypoints(kdf, mapping)
    kout.to_parquet(P / "keypoints_2d.parquet", index=False)
    print(f"keypoints: {len(kout)} rows written, {kdrop} dropped")
    for cam in ("sideline", "endzone"):
        path = P / f"poses_{cam}.json"
        if not path.exists():
            continue
        blob = pickle.load(open(path, "rb"))
        pout, pdrop = relabel_pose_cache(blob, mapping, cam)
        with open(path, "wb") as fh:
            pickle.dump(pout, fh)
        print(f"poses_{cam}: {sum(len(r) for r in pout['frames'].values())} records written, {pdrop} dropped")
    print("ids changed: re-run 08c --from-cache, 08n, then the triangulation and the fits")


if __name__ == "__main__":
    main()
