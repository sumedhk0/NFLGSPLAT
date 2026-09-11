#!/usr/bin/env python
"""Recompute every box's kit margin from the POSE torso (identity.torso_colours).

    python scripts/08p_kit_from_pose.py --play-dir <P> [--tables tracks.parquet ...] [--dry-run]

WHY. The kit margin 08b writes comes from the middle band of the box, and for a body in
a three-point stance that band is his white pants: play 1's endzone camera read Kansas
City's crouched linemen as the white kit, which cost them their team and vetoed their
cross-camera pairs (a kit clash blocks a pair). The keypoints give the real torso -- the
shoulders-and-hips quadrilateral -- and there the two kits separate cleanly.

WHAT. For each camera: the torso colour per box from the pose where the keypoints have
one, the old band where they do not; the camera's saturation split (identity.team_color);
and the signed, gap-normalised margin per box. Every named table is rewritten with the new
kit_margin, matched BY BOX so a table with other ids (the unpaired one) is updated too.
Originals are kept as *_bandkit.parquet. Run it after the keypoints (05m) and before the
pairing repair; the stages downstream of kit_margin are 08i, 08k and 08f.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.identity.team_color import KitSplitError, saturation_split
from nfl_gsplat.identity.torso_colours import detection_colours
from nfl_gsplat.tracking.kits import KIT_MARGIN

MIN_CONF = 0.3
KEY = ["cam", "frame", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def joints_by_box(kdf: pd.DataFrame, cam: str) -> dict:
    """``{(frame, track_id): {joint: (x, y)}}`` for one camera's confident keypoints."""
    g = kdf[(kdf["cam"] == cam) & (kdf["conf"] >= MIN_CONF)]
    out: dict = {}
    for (f, pid), rows in g.groupby(["frame", "global_player_id"]):
        out[(int(f), int(pid))] = {int(r.joint): (float(r.x), float(r.y)) for r in rows.itertuples()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--tables", nargs="*", default=["tracks.parquet", "tracks_identity.parquet",
                                                    "tracks_identity_unpaired.parquet"],
                    help="tables to rewrite (those that exist); matched by box, so ids may differ")
    ap.add_argument("--source", default="tracks.parquet", help="the table whose ids the keypoints carry")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    src = pd.read_parquet(P / args.source)
    src = src[src["track_id"] >= 0].copy()
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    new_margin = {}
    for cam in sorted(src["cam"].unique()):
        view = src[src["cam"] == cam].reset_index(drop=True)
        sat = detection_colours(view, P / f"{cam}.mp4", max_per_frame=512,
                                joints_by=joints_by_box(kdf, str(cam)))[:, 1]
        ok = np.isfinite(sat)
        try:
            c, _, _sep = saturation_split(sat[ok])
        except KitSplitError as exc:
            raise SystemExit(f"{cam}: {exc}")
        mid, span = 0.5 * (c[0] + c[1]), max(float(c[1] - c[0]), 1e-6)
        m = np.full(len(sat), np.nan)
        m[ok] = (sat[ok] - mid) / span
        old = view["kit_margin"].to_numpy(float) if "kit_margin" in view else np.full(len(view), np.nan)
        both = np.isfinite(old) & np.isfinite(m)
        flip = int((np.sign(old[both]) != np.sign(m[both])).sum())
        sure_old = int((np.abs(old[np.isfinite(old)]) >= KIT_MARGIN).sum())
        sure_new = int((np.abs(m[ok]) >= KIT_MARGIN).sum())
        print(f"{cam}: {int(ok.sum())}/{len(view)} boxes measured; split {c[0]:.0f}/{c[1]:.0f} saturation; "
              f"labelled (|margin| >= {KIT_MARGIN}) {sure_old} -> {sure_new}; {flip} boxes change side")
        for key, val in zip(zip(view["cam"].astype(str), view["frame"].astype(int),
                                view["bbox_x1"].round(2), view["bbox_y1"].round(2),
                                view["bbox_x2"].round(2), view["bbox_y2"].round(2)), m):
            new_margin[key] = float(val)
    if args.dry_run:
        print("dry run; no table rewritten")
        return
    for name in args.tables:
        path = P / name
        if not path.exists():
            continue
        t = pd.read_parquet(path)
        keys = list(zip(t["cam"].astype(str), t["frame"].astype(int), t["bbox_x1"].round(2), t["bbox_y1"].round(2),
                        t["bbox_x2"].round(2), t["bbox_y2"].round(2)))
        vals = np.array([new_margin.get(k, np.nan) for k in keys], float)
        hit = int(np.isfinite(vals).sum())
        backup = path.with_name(path.stem + "_bandkit" + path.suffix)
        if not backup.exists():
            shutil.copy(path, backup)
        t["kit_margin"] = vals
        t.to_parquet(path, index=False)
        print(f"{name}: {hit}/{len(t)} rows given a pose kit margin (the rest unknown); kept {backup.name}")


if __name__ == "__main__":
    main()
