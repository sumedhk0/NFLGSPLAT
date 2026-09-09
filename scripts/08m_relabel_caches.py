#!/usr/bin/env python
"""Carry the keypoints and pose caches across a re-pairing (tracking.relabel).

After 08i re-pairs the camera tracks (new global ids, same boxes) run this
BEFORE copying tracks_identity.parquet over tracks.parquet:

    C:/venvs/smplx312/Scripts/python scripts/08m_relabel_caches.py --play-dir <P>

Reads the old ids from tracks.parquet and the new from tracks_identity.parquet,
rewrites keypoints_2d.parquet, poses_sideline.json and poses_endzone.json with
the new ids (originals kept as *_oldids.*). Must run under numpy 1 (smplx312):
the pose caches are numpy-1 pickles.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import pandas as pd

from nfl_gsplat.tracking.relabel import (assert_numpy1_for_pickles, id_map_by_boxes, relabel_keypoints,
                                         relabel_pose_cache)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--old", type=Path, default=None, help="tracks with the OLD ids (default tracks.parquet)")
    ap.add_argument("--new", type=Path, default=None, help="tracks with the NEW ids (default tracks_identity.parquet)")
    args = ap.parse_args()
    P = args.play_dir
    assert_numpy1_for_pickles()
    old = pd.read_parquet(args.old or P / "tracks.parquet")
    new = pd.read_parquet(args.new or P / "tracks_identity.parquet")
    old = old[old["track_id"] >= 0]
    new = new[new["track_id"] >= 0]
    mapping = id_map_by_boxes(old, new)
    changed = sum(1 for (cam, f, oid), nid in mapping.items() if oid != nid)
    print(f"id map: {len(mapping)} (camera, frame, id) keys, {changed} change id")

    kp = P / "keypoints_2d.parquet"
    if kp.exists():
        backup = kp.with_name("keypoints_2d_oldids.parquet")
        if not backup.exists():
            shutil.copy(kp, backup)
        kdf = pd.read_parquet(kp)
        out, dropped = relabel_keypoints(kdf, mapping)
        out.to_parquet(kp, index=False)
        print(f"keypoints: {len(out)} rows written, {dropped} dropped (no box in the tracks)")

    for cam in ("sideline", "endzone"):
        path = P / f"poses_{cam}.json"
        if not path.exists():
            continue
        backup = path.with_name(f"poses_{cam}_oldids.json")
        if not backup.exists():
            shutil.copy(path, backup)
        blob = pickle.load(open(path, "rb"))
        out, dropped = relabel_pose_cache(blob, mapping, cam)
        n = sum(len(r) for r in out["frames"].values())
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        print(f"poses_{cam}: {n} records written, {dropped} dropped")


if __name__ == "__main__":
    main()
