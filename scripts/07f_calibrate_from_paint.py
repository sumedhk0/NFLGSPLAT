#!/usr/bin/env python
"""Score the TRACKING-FREE camera against the tracking-fitted one.

This is the measurement the project could not make before. The production path
has to calibrate from the picture alone, because arbitrary game footage carries
no tracking, and until now there was no way to tell how good that was. The
Helmet Assignment set supplies a camera fitted from real tracking, so a
paint-only camera can be scored against something known.

BOTH ARE SCORED THE SAME WAY, by single-view ray-plane placement: cast a ray
through each labelled helmet box, meet the helmet plane, compare to tracking.
Using one view for both keeps the comparison about the CAMERAS. Two-view
triangulation is better than this for either camera -- 0.13 m against 0.32 m for
the reference on 57583/82 -- so these numbers are not the pipeline's accuracy,
they are a like-for-like camera comparison.

THE UNKNOWN X SHIFT is removed before scoring. Paint cannot resolve which
5-yard line is which, so the recovered camera is right only up to a whole-yard
shift along the field, which does not affect reconstruction. The shift is
reported too, because a correct solve returns one CLOSE TO A WHOLE NUMBER of
5-yard steps -- that near-integer is free evidence the labelling was right, and
a shift landing mid-step means it was not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration import field_detect
from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
from nfl_gsplat.calibration.from_helmets import cameras_fixed_centre
from nfl_gsplat.calibration.from_paint import (
    cameras_from_paint,
    cameras_from_paint_pooled,
    ray_to_plane,
)
from nfl_gsplat.data.align_video import VIDEO_FPS, helmet_boxes_by_frame
from nfl_gsplat.data.helmet_dataset import load_labels, load_tracking
from nfl_gsplat.errors import CalibrationError

WIDTH, HEIGHT = 1280, 720
# Turf below the helmet plane, measured in 07e against the painted lines.
TURF_DROP_M = 1.5


def placement_errors(cams, byf, world_at, z_plane, *, remove_x_shift):
    """(errors_m, x_shift_m) for one camera set."""
    per_frame, shifts = [], []
    for frame, (K, R, t) in cams.items():
        if frame not in byf:
            continue
        uv, cols = byf[frame]
        truth = world_at(frame)[cols]
        got = [ray_to_plane(K, R, t, p, z_plane) for p in uv]
        keep = [i for i, g in enumerate(got)
                if g is not None and np.isfinite(truth[i]).all()]
        if len(keep) < 5:
            continue
        est = np.array([got[i] for i in keep])
        ref = truth[keep]
        shifts.append(float(np.median(ref[:, 0] - est[:, 0])))
        per_frame.append((est, ref))
    if not per_frame:
        return None, None
    shift = float(np.median(shifts)) if remove_x_shift else 0.0
    err = np.concatenate([
        np.linalg.norm((est + np.array([shift, 0.0])) - ref, axis=1)
        for est, ref in per_frame])
    return err, shift


def measure_view(root, labels, track, game, play, view, offset, *, stride, frames):
    import cv2

    index = {p: i for i, p in enumerate(track.players)}
    name = f"{game}_{play:06d}_{view}.mp4"
    byf0 = helmet_boxes_by_frame(labels[labels["video"] == name], index)
    byf = {f: v for i, (f, v) in enumerate(sorted(byf0.items()))
           if i % stride == 0 and track.covers(offset + f / VIDEO_FPS)}
    if len(byf) < 10:
        return {"failed": "too few labelled frames"}

    def world_at(frame):
        return track.at(offset + frame / VIDEO_FPS)

    try:
        ref_cams, ref_centre, mirrored = cameras_fixed_centre(
            byf, world_at, WIDTH, HEIGHT)
        if mirrored:
            return {"failed": "reference solve chose a mirrored world"}
    except CalibrationError as exc:
        return {"failed": f"reference: {str(exc)[:90]}"}

    chosen = sorted(ref_cams)[:frames]
    cap = cv2.VideoCapture(str(root / "video" / name))
    feats, images = {}, {}
    for f in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, img = cap.read()
        if ok:
            feats[f] = field_detect.detect_field_features(img)
            images[f] = img
    cap.release()
    if len(feats) < 5:
        return {"failed": "could not read frames"}

    # Pooled first: one shared centre is far more accurate, but it refuses
    # plays it cannot fit, so per-frame remains the fallback and the result
    # records which was used.
    residual, method = float("nan"), "pooled"
    try:
        paint_cams, focal, _centre, mirrored = cameras_from_paint_pooled(
            feats, WIDTH, HEIGHT, images=images)
        if mirrored:
            raise CalibrationError("pooled solve chose a mirrored world")
    except CalibrationError:
        method = "per-frame"
        try:
            paint_cams, focal, residual = cameras_from_paint(
                feats, WIDTH, HEIGHT, images=images)
        except CalibrationError as exc:
            return {"failed": f"paint: {str(exc)[:90]}"}

    # Reference lives in the helmet-plane frame (helmets at z=0); paint lives in
    # the ground frame (helmets at z = TURF_DROP_M).
    ref_err, _ = placement_errors(ref_cams, byf, world_at, 0.0,
                                  remove_x_shift=False)
    paint_err, shift = placement_errors(paint_cams, byf, world_at, TURF_DROP_M,
                                        remove_x_shift=True)
    if ref_err is None or paint_err is None:
        return {"failed": "no placements"}

    steps = shift / YARD_LINE_SPACING_M
    return {
        "reference_median_m": round(float(np.median(ref_err)), 3),
        "paint_median_m": round(float(np.median(paint_err)), 3),
        "paint_p90_m": round(float(np.percentile(paint_err, 90)), 3),
        "ratio": round(float(np.median(paint_err) / max(np.median(ref_err), 1e-9)), 2),
        "focal_paint": round(float(focal), 1),
        "line_residual_px": (None if not np.isfinite(residual)
                             else round(float(residual), 2)),
        "x_shift_m": round(shift, 2),
        "x_shift_steps": round(float(steps), 2),
        "shift_off_integer": round(float(abs(steps - round(steps))), 3),
        "frames": len(paint_cams),
        "method": method,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--alignment", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--frames", type=int, default=30,
                    help="frames per view to run field detection on")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", default="Sideline,Endzone")
    args = ap.parse_args()

    align_path = args.alignment or (args.root / "alignment.json")
    out_path = args.out or (args.root / "paint_calibration.json")
    alignment = json.loads(align_path.read_text(encoding="utf-8"))
    labels = load_labels(args.root / "train_labels.csv")
    plays = load_tracking(args.root / "train_player_tracking.csv")

    usable = [r for r in alignment.values() if r["offset_s"] is not None]
    if args.limit:
        usable = usable[:args.limit]
    views = [v for v in args.views.split(",") if v]
    print(f"{len(usable)} aligned plays x {len(views)} views\n")

    out = {}
    for i, rec in enumerate(usable, 1):
        game, play, offset = rec["game_key"], rec["play_id"], rec["offset_s"]
        track = plays.get((game, play))
        if track is None:
            continue
        for view in views:
            got = measure_view(args.root, labels, track, game, play, view,
                               offset, stride=args.stride, frames=args.frames)
            out[f"{game}_{play}_{view}"] = got
            if "failed" in got:
                print(f"[{i:3d}] {game}/{play:<6d} {view:8s} FAILED {got['failed']}",
                      flush=True)
            else:
                print(f"[{i:3d}] {game}/{play:<6d} {view:8s} "
                      f"paint {got['paint_median_m']:5.2f} m vs "
                      f"reference {got['reference_median_m']:5.2f} m "
                      f"({got['ratio']:4.1f}x)  shift {got['x_shift_steps']:+6.2f} steps "
                      f"(off {got['shift_off_integer']:.2f})  "
                      f"f={got['focal_paint']:.0f} [{got['method']}]",
                      flush=True)

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    good = [v for v in out.values() if "failed" not in v]
    print(f"\n{len(good)}/{len(out)} views measured  ->  {out_path}")
    if good:
        p = np.array([v["paint_median_m"] for v in good])
        r = np.array([v["reference_median_m"] for v in good])
        off = np.array([v["shift_off_integer"] for v in good])
        print(f"paint-only placement:  median {np.median(p):.2f} m, "
              f"best {p.min():.2f}, worst {p.max():.2f}")
        print(f"tracking-fitted:       median {np.median(r):.2f} m")
        print(f"ratio:                 {np.median(p) / np.median(r):.1f}x looser")
        print(f"X shift lands within 0.15 of a whole 5-yard step in "
              f"{int((off <= 0.15).sum())}/{len(off)} views")


if __name__ == "__main__":
    main()
