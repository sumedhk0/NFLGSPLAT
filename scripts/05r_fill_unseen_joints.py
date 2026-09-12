#!/usr/bin/env python
"""Interpolate the limbs no camera saw, between the poses either side (pose.fill_unseen).

    C:/venvs/smplx312/Scripts/python scripts/05r_fill_unseen_joints.py --play-dir <P> [--dry-run]

The fit walks forward in time, so a limb that disappears keeps whatever the previous frame and
the generic prior gave it -- play 1's motion man swings his arms at 8.9 m/s in his own frame
through frames 262-272 while his shoulders sit at 3 px. The pose AFTER the blackout is the
information a sequential fit cannot use. This pass finds, per player and per joint, the runs of
frames where no camera saw that joint or anything below it, and replaces the joint's rotation
across the run by the interpolation of its two ends. Seen frames are untouched.

Rewrites poses_refit.json (the input kept as poses_refit_unfilled.json). numpy 1 (smplx312).
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.pose.coco import coco_to_body
from nfl_gsplat.pose.fill_unseen import fill_track
from nfl_gsplat.pose.fit_mono2d import NUM_BODY_JOINTS, unseen_joints
from nfl_gsplat.render.play_timeline import clip_offset
from nfl_gsplat.tracking.relabel import assert_numpy1_for_pickles


def seen_by_frame(kdf, pid: int, frames, *, offset: int, min_conf: float):
    """``{joint: [T] bool}``: was this body joint constrained by any camera on that frame?"""
    seen = np.zeros((len(frames), NUM_BODY_JOINTS), bool)
    for cam, shift in (("sideline", 0), ("endzone", offset)):
        g = kdf[(kdf["cam"] == cam) & (kdf["global_player_id"] == pid)]
        if not len(g):
            continue
        by = {int(f): rows for f, rows in g.groupby("frame")}
        for i, f in enumerate(frames):
            rows = by.get(int(f) + int(shift))
            if rows is None or len(rows) != 17:
                continue
            rows = rows.sort_values("joint")
            _uv, c = coco_to_body(rows[["x", "y"]].to_numpy(float), rows["conf"].to_numpy(float), min_conf=min_conf)
            seen[i] |= np.asarray(c, float)[:NUM_BODY_JOINTS] >= min_conf
    out = {}
    for i in range(len(frames)):
        for j in unseen_joints(seen[i]):
            out.setdefault(int(j), np.ones(len(frames), bool))[i] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--refit", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--max-run", type=int, default=40)
    ap.add_argument("--no-keypoint-filter", action="store_true",
                    help="judge 'seen' on the RAW keypoints; by default the same temporal filter the fit "
                         "used (pose.keypoint_filter) is applied first, because a joint the filter threw "
                         "out is a joint the fit did not have")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    assert_numpy1_for_pickles()
    refit_path = args.refit or P / "poses_refit.json"
    blob = pickle.load(open(refit_path, "rb"))
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    if not args.no_keypoint_filter:
        from nfl_gsplat.pose.keypoint_filter import reject_outliers

        kdf, n_rej = reject_outliers(kdf)
        print(f"{n_rej} keypoints thrown out by the temporal filter, as the fit threw them out")
    offset = clip_offset(P)

    by_pid: dict = {}
    for f, recs in blob["frames"].items():
        for pid, r in recs.items():
            by_pid.setdefault(int(pid), []).append((int(f), r))
    total = 0
    touched = 0
    for pid, items in sorted(by_pid.items()):
        items.sort()
        frames = [f for f, _ in items]
        if len(frames) < 3:
            continue
        bp = np.stack([np.asarray(r["body_pose"], float).reshape(-1)[:63] for _, r in items])
        seen = seen_by_frame(kdf, pid, frames, offset=offset, min_conf=args.min_conf)
        if not seen:
            continue
        out, n = fill_track(bp, seen, max_run=args.max_run)
        if not n:
            continue
        total += n
        touched += 1
        if not args.dry_run:
            for (f, r), row in zip(items, out):
                r["body_pose"] = np.asarray(row, np.float32)
    n_frames = sum(len(r) for r in blob["frames"].values())
    print(f"{total} (joint, frame) rotations interpolated across blackouts on {touched} players "
          f"({n_frames} body-frames in the cache)")
    if args.dry_run:
        print("dry run; nothing written")
        return
    out_path = args.out or refit_path
    backup = refit_path.with_name(refit_path.stem + "_unfilled.json")
    if not backup.exists():
        shutil.copy(refit_path, backup)
    with open(out_path, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"wrote {out_path} (the fit's own poses kept in {backup.name})")


if __name__ == "__main__":
    main()
