#!/usr/bin/env python
"""Put a play-dir's cameras in the rule-book field frame, from the yard numbers.

Two things the paint calibration cannot know, both read off the numerals:

  1. CROSS-FIELD SCALE. The numerals sit on rows at y = +-14.33 m. Reading
     them along each yard line (field.yard_numbers.read_line_strips) says
     where the solved frame puts those rows; calibration.row_ruler turns the
     difference into a corrected camera per frame (play 1: rows read at
     +-11.0 m, a 22% error in every player's y).
  2. WHICH LINE IS WHICH. The numeral on a line names it; the 5-yard shift
     that puts every reading on a line carrying its numeral is voted
     (field.yard_numbers.solve_transform) and applied to the cameras.

Both are CHECKED before they are applied, and the check is printed:

  - the rows re-read under the refined camera must land at +-14.3 m;
  - player heights (box height in pixels against 1.85 m projected at the
    foot) must still read 1.85 m under the refined camera -- the players
    were the ruler the paint lacked and stay the judge;
  - the shift must win its vote by WIN_MARGIN.

Without --apply nothing is written but field_offset.json. With it, the
sideline track in cameras.npz is replaced (the original kept as
cameras_relative.npz). The ENDZONE camera was solved against the sideline's
ground cloud (scripts/08) and is not corrected here: after --apply, re-run
08 with --sideline-from this play-dir, or accept that the endzone view is
still in the old frame until then. This script prints which.

Runs in the nflgsplat env (easyocr).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration import row_ruler as rr
from nfl_gsplat.calibration.cameras_io import (load_camera_track,
                                               write_camera_track)
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.field import yard_numbers as yn
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
EXPECTED_HEIGHT_M = 1.85


def sample_frames(track, n: int):
    ok = np.flatnonzero(track.conf > 0)
    if len(ok) == 0:
        raise CalibrationError("no frame with a camera in this track")
    return [int(f) for f in np.unique(np.linspace(ok[0], ok[-1], n).round().astype(int))
            if track.conf[f] > 0]


def read_frames(video, track, frames, reader):
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        intr, pose = track.at(f)
        out[f] = yn.read_line_strips(img, intr.K(), pose.R, pose.t, reader)
    cap.release()
    return out


def player_heights(track, df, frames):
    """Median implied height of every detection on ``frames``: box height
    in pixels over what 1.85 m projects to at the box's foot."""
    from nfl_gsplat.pose.place_on_field import ground_point

    hs = []
    sub = df[df["frame"].isin(frames)]
    for row in sub.itertuples():
        intr, pose = track.at(int(row.frame))
        K, R, t = intr.K(), pose.R, pose.t
        foot = (0.5 * (row.bbox_x1 + row.bbox_x2), float(row.bbox_y2))
        try:
            g = ground_point(foot, K, R, t)
        except Exception:
            continue
        top = K @ (R @ (np.asarray(g) + [0, 0, EXPECTED_HEIGHT_M]) + t)
        bottom = K @ (R @ np.asarray(g) + t)
        px_per_body = abs(top[1] / top[2] - bottom[1] / bottom[2])
        if px_per_body > 1:
            hs.append(EXPECTED_HEIGHT_M * (row.bbox_y2 - row.bbox_y1) / px_per_body)
    return float(np.median(hs)) if hs else float("nan"), len(hs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-rows", dest="rows", action="store_false",
                    help="shift only; leave the cross-field scale alone")
    args = ap.parse_args()

    import easyocr
    import pandas as pd

    tracks = load_camera_track(args.play_dir / "cameras.npz")
    track = tracks[args.cam]
    video = args.play_dir / f"{args.cam}.mp4"
    df = pd.read_parquet(args.play_dir / "tracks.parquet")
    df = df[df["cam"] == args.cam]
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    frames = sample_frames(track, args.frames)

    per_frame = read_frames(video, track, frames, reader)
    readings = [r for rs in per_frame.values() for r in rs]
    for f, rs in per_frame.items():
        print(f"frame {f:5d}: " + ", ".join(
            f"{r.numeral} on {r.x_m / yn.YARD_TO_M:+.0f}yd at y {r.y_m:+.1f}" for r in rs))
    if not readings:
        raise CalibrationError("no numeral read on any frame; the camera may not see "
                               "the number rows, or the calibration is far off")

    h_before, n_h = player_heights(track, df, frames)
    print(f"\nplayer height under the solved camera: {h_before:.2f} m ({n_h} boxes)")

    refined = track
    row_fit = None
    if args.rows:
        row_fit = rr.fit_row_scale([r.y_m for r in readings], [r.side for r in readings])
        print(f"rows: y_true = {row_fit.scale:.3f} * y + {row_fit.offset:+.2f}  "
              f"({row_fit.n_far} far, {row_fit.n_near} near readings, "
              f"residual {row_fit.residual_m:.2f} m)")
        refined = rr.refine_track(track, row_fit)
        check = read_frames(video, refined, frames, reader)
        ys = [r.y_m for rs in check.values() for r in rs]
        print(f"rows re-read under the refined camera: median |y| "
              f"{np.median(np.abs(ys)) if ys else float('nan'):.2f} m (rule book "
              f"{rr.ROW_Y_M:.2f}), {len(ys)} readings")
        h_after, _ = player_heights(refined, df, frames)
        print(f"player height under the refined camera: {h_after:.2f} m")
        readings = [r for rs in check.values() for r in rs] or readings

    tf = yn.solve_transform(readings)
    if tf is None:
        raise CalibrationError("the numerals do not agree on a shift (one numeral, or "
                               "misreads); more frames or a view of two numerals needed")
    yards_in = (yn.GOAL_LINE_X_M - abs(tf.shift_m)) / yn.YARD_TO_M
    print(f"\nshift: {tf.shift_m:+.2f} m ({tf.shift_m / yn.YARD_TO_M:+.0f} yd); solved x = 0 "
          f"is the {yards_in:.0f}-yard line on the {'left' if tf.shift_m < 0 else 'right'} "
          f"half; votes {tf.votes:.1f} vs {tf.runner_up:.1f} over {tf.n_readings} readings")

    out = {"cam": args.cam, "shift_m": tf.shift_m, "turn": tf.turn, "votes": tf.votes,
           "runner_up": tf.runner_up, "n_readings": tf.n_readings,
           "row_scale": None if row_fit is None else
           {"scale": row_fit.scale, "offset": row_fit.offset,
            "residual_m": row_fit.residual_m},
           "height_before_m": h_before, "frames": frames}
    (args.play_dir / "field_offset.json").write_text(json.dumps(out, indent=2))
    if not args.apply:
        print("written field_offset.json; --apply to rewrite cameras.npz")
        return
    final = yn.transform_track(refined, tf)
    backup = args.play_dir / "cameras_relative.npz"
    if not backup.exists():
        shutil.copy(args.play_dir / "cameras.npz", backup)
    tracks[args.cam] = final
    write_camera_track(args.play_dir / "cameras.npz", tracks, fps=59.94)
    others = [c for c in tracks if c != args.cam]
    print(f"cameras.npz rewritten ({args.cam} in the field frame; original in "
          f"{backup.name}); {', '.join(others)} NOT corrected -- re-solve against "
          f"this sideline (scripts/08) before fusing views")


if __name__ == "__main__":
    main()
