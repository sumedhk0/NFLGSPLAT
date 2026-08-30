#!/usr/bin/env python
"""Measure this pipeline's 3D player positions against external ground truth.

This is the first measurement in the project that is not self-referential.
Everything reported before now has been cross-CONSISTENCY -- two of our own
estimates agreeing, which says nothing about whether either is right. Two-view
pose agreed to 0.10 m while placement disagreed by 2.90 m; consistency did not
notice. The Helmet Assignment set has real tracking, so error can be measured.

WHAT IS BEING MEASURED. Per play: calibrate both cameras from the labelled
helmets and the tracked positions, triangulate every player seen in both views,
and compare against tracking. Reported as

    xy error   metres, against tracking. The headline number.
    z spread   metres. Tracking has NO height, so this is not scored against
               anything -- it is a free consistency check, because triangulated
               helmets should scatter about the ~0.3 m that real helmet heights
               vary. A large spread means the cameras are wrong even when xy
               happens to look acceptable.
    centre     recovered camera position, for a plausibility read: a broadcast
               sideline camera sits tens of metres out and tens of metres up.

WHAT IT IS NOT. The cameras here are fitted USING the tracking, so xy error is
not an end-to-end score of the full pipeline -- it is the geometry's error floor
given perfect identity and perfect correspondence. That floor is exactly what
was missing: it says how much of the 2.90 m placement error is geometry and how
much is everything upstream of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.decompose_homography import (
    camera_centre,
    projection_matrix,
)
from nfl_gsplat.calibration.from_helmets import (
    cameras_for_view,
    triangulate_matched,
)
from nfl_gsplat.data.align_video import VIDEO_FPS, helmet_boxes_by_frame
from nfl_gsplat.data.helmet_dataset import load_labels, load_tracking
from nfl_gsplat.errors import CalibrationError

WIDTH, HEIGHT = 1280, 720


def measure_play(labels, track, game, play, offset, *, stride=5):
    index = {p: i for i, p in enumerate(track.players)}

    def world_at(frame):
        return track.at(offset + frame / VIDEO_FPS)

    views = {}
    for view in ("Sideline", "Endzone"):
        sub = labels[labels["video"] == f"{game}_{play:06d}_{view}.mp4"]
        if sub.empty:
            return None, f"no labels for {view}"
        byf = helmet_boxes_by_frame(sub, index)
        byf = {f: v for i, (f, v) in enumerate(sorted(byf.items()))
               if i % stride == 0}
        if not byf:
            return None, f"no usable frames in {view}"
        # Frames whose time falls outside the record would be scored against
        # clamped positions, which are frozen and fit anything.
        byf = {f: v for f, v in byf.items()
               if track.covers(offset + f / VIDEO_FPS)}
        if not byf:
            return None, f"{view} lies outside the tracking record"
        try:
            cams, focal = cameras_for_view(byf, world_at, WIDTH, HEIGHT)
        except CalibrationError as exc:
            return None, f"{view}: {exc}"
        views[view] = (byf, cams, focal)

    (byf_s, cam_s, f_s) = views["Sideline"]
    (byf_e, cam_e, f_e) = views["Endzone"]

    errs, zs, n_pts = [], [], 0
    for frame in sorted(set(cam_s) & set(cam_e)):
        Ks, Rs, ts = cam_s[frame]
        Ke, Re, te = cam_e[frame]
        uv_s, cols_s = byf_s[frame]
        uv_e, cols_e = byf_e[frame]
        cols, xyz = triangulate_matched(
            projection_matrix(Ks, Rs, ts), uv_s, cols_s,
            projection_matrix(Ke, Re, te), uv_e, cols_e)
        if len(cols) == 0:
            continue
        truth = world_at(frame)[cols]
        ok = np.isfinite(truth).all(axis=1)
        if not ok.any():
            continue
        errs.append(np.linalg.norm(xyz[ok, :2] - truth[ok], axis=1))
        zs.append(xyz[ok, 2])
        n_pts += int(ok.sum())

    if not errs:
        return None, "no frame had players in both views"
    err = np.concatenate(errs)
    z = np.concatenate(zs)
    cs = np.median([camera_centre(R, t) for _K, R, t in cam_s.values()], axis=0)
    ce = np.median([camera_centre(R, t) for _K, R, t in cam_e.values()], axis=0)
    return {
        "frames": len(set(cam_s) & set(cam_e)), "points": n_pts,
        "xy_median_m": float(np.median(err)),
        "xy_p90_m": float(np.percentile(err, 90)),
        "z_iqr_m": float(np.subtract(*np.percentile(z, [75, 25]))),
        "focal_sideline": round(f_s, 1), "focal_endzone": round(f_e, 1),
        "centre_sideline": [round(float(v), 1) for v in cs],
        "centre_endzone": [round(float(v), 1) for v in ce],
    }, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--alignment", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=5,
                    help="use every Nth labelled frame")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    align_path = args.alignment or (args.root / "alignment.json")
    out_path = args.out or (args.root / "accuracy.json")
    alignment = json.loads(align_path.read_text(encoding="utf-8"))
    labels = load_labels(args.root / "train_labels.csv")
    plays = load_tracking(args.root / "train_player_tracking.csv")

    usable = [r for r in alignment.values() if r["offset_s"] is not None]
    if args.limit:
        usable = usable[:args.limit]
    print(f"{len(usable)} aligned plays to measure\n")

    out, good = {}, []
    for i, rec in enumerate(usable, 1):
        game, play = rec["game_key"], rec["play_id"]
        track = plays.get((game, play))
        if track is None:
            continue
        got, reason = measure_play(labels, track, game, play, rec["offset_s"],
                                   stride=args.stride)
        out[f"{game}_{play}"] = got or {"failed": reason}
        if got:
            good.append(got["xy_median_m"])
            print(f"[{i:3d}/{len(usable)}] {game}/{play:<6d} "
                  f"xy {got['xy_median_m']:5.2f} m (p90 {got['xy_p90_m']:5.2f})  "
                  f"z IQR {got['z_iqr_m']:4.2f} m  "
                  f"f {got['focal_sideline']:.0f}/{got['focal_endzone']:.0f}  "
                  f"side {got['centre_sideline']}  end {got['centre_endzone']}",
                  flush=True)
        else:
            print(f"[{i:3d}/{len(usable)}] {game}/{play:<6d} FAILED  {reason}",
                  flush=True)

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{len(good)}/{len(usable)} plays measured  ->  {out_path}")
    if good:
        g = np.asarray(good)
        print(f"xy error across plays: median {np.median(g):.2f} m, "
              f"best {g.min():.2f} m, worst {g.max():.2f} m")


if __name__ == "__main__":
    main()
