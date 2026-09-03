#!/usr/bin/env python
"""Put a play-dir's cameras in the rule-book field frame, from the paint.

Two things the paint calibration cannot know, both read off the turf:

  1. CROSS-FIELD SCALE. Two kinds of paint sit at a y the rule book fixes:
     the hash marks (+-2.82 m) and the numeral rows (+-12.50 m). Both are
     measured through the solved camera (calibration.row_ruler,
     field.yard_numbers) and each ruler's implied scale is printed. They
     must AGREE within RULER_TOL before the correction is applied: the
     first version of this script trusted the numerals alone against a row
     constant that was wrong, and would have stretched the camera by 27%
     where the true error was 6%. The hash marks caught it.
  2. WHICH LINE IS WHICH. The numeral on a line names it; the 5-yard shift
     that puts every reading on a line carrying its numeral is voted
     (field.yard_numbers.solve_transform) and applied to the cameras.

Checks printed before anything is applied: both rulers re-read under the
refined camera; player heights (box height against 1.85 m projected at the
foot) before and after -- reported, not trusted, since the solved camera
was selected on them and snap boxes crouch; the shift's vote margin; and,
with --los-yards (the line of scrimmage from the play description, e.g.
"BLT 24" -> 24), where the snap formation lands in yards from the nearest
goal line, which must match.

Without --apply nothing is written but field_offset.json. With it, the
sideline track in cameras.npz is replaced (the original kept as
cameras_relative.npz). The ENDZONE camera was solved against the sideline's
ground cloud (scripts/08) and is not corrected here: after --apply, re-run
08 with --sideline-from this play-dir. This script says so.

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
RULER_TOL = 0.05          # the numeral and hash rulers must agree on the scale this well


def sample_frames(track, n: int):
    ok = np.flatnonzero(track.conf > 0)
    if len(ok) == 0:
        raise CalibrationError("no frame with a camera in this track")
    return [int(f) for f in np.unique(np.linspace(ok[0], ok[-1], n).round().astype(int))
            if track.conf[f] > 0]


def read_frames(video, track, frames, reader):
    """Per frame: numeral readings and hash rows, through the track's camera."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    numerals, hashes = {}, {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        intr, pose = track.at(f)
        K, R, t = intr.K(), pose.R, pose.t
        numerals[f] = yn.read_line_strips(img, K, R, t, reader)
        hashes[f] = rr.measure_hash_rows(img, K, R, t)
    cap.release()
    return numerals, hashes


def row_readings(numerals, hashes):
    """``(ys_solved, ys_true, rulers)`` over every frame's rows."""
    ys, yt, rl = [], [], []
    for rs in numerals.values():
        for r in rs:
            ys.append(r.y_m)
            yt.append(r.side * rr.ROW_Y_M)
            rl.append("numerals")
    for rows in hashes.values():
        for y, side in rows:
            ys.append(y)
            yt.append(side * rr.HASH_Y_M)
            rl.append("hashes")
    return ys, yt, rl


def snap_yards(track, df, *, n_frames: int = 3):
    """Where the formation stands on the first posed frames, in yards from
    the nearest goal line (median player x)."""
    from nfl_gsplat.pose.place_on_field import ground_point

    xs = []
    for f in sorted(df["frame"].unique())[:n_frames]:
        intr, pose = track.at(int(f))
        K, R, t = intr.K(), pose.R, pose.t
        for row in df[df["frame"] == f].itertuples():
            try:
                xs.append(ground_point((0.5 * (row.bbox_x1 + row.bbox_x2), float(row.bbox_y2)),
                                       K, R, t)[0])
            except Exception:
                continue
    if not xs:
        return float("nan")
    return (yn.GOAL_LINE_X_M - abs(float(np.median(xs)))) / yn.YARD_TO_M


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
    ap.add_argument("--los-yards", type=float, default=None,
                    help="line of scrimmage in yards from the goal line, from the play "
                         "description (e.g. 'KC 2-20 BLT 24' -> 24); checked, not used")
    ap.add_argument("--force", action="store_true",
                    help="apply even when the two rulers disagree (say why in the log)")
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

    numerals, hashes = read_frames(video, track, frames, reader)
    readings = [r for rs in numerals.values() for r in rs]
    for f in frames:
        num = ", ".join(f"{r.numeral} on {r.x_m / yn.YARD_TO_M:+.0f}yd at y {r.y_m:+.1f}"
                        for r in numerals.get(f, []))
        hsh = " ".join(f"{y:+.2f}" for y, _s in hashes.get(f, []))
        print(f"frame {f:5d}: numerals [{num}]  hash rows [{hsh}]")
    if not readings:
        raise CalibrationError("no numeral read on any frame; the camera may not see "
                               "the number rows, or the calibration is far off")

    h_before, n_h = player_heights(track, df, frames)
    print(f"\nplayer height under the solved camera: {h_before:.2f} m ({n_h} boxes)")

    refined = track
    row_fit = None
    agree = True
    if args.rows:
        ys0, yt0, rl0 = row_readings(numerals, hashes)
        row_fit = rr.fit_rows(ys0, yt0, rulers=rl0)
        by = row_fit.by_ruler or {}
        print(f"rows: y_true = {row_fit.scale:.3f} * y + {row_fit.offset:+.2f}  "
              f"({row_fit.n_far} far, {row_fit.n_near} near readings, MAD {row_fit.residual_m:.2f} m)")
        print("       by ruler: " + ", ".join(f"{k} {v:.3f}" for k, v in by.items()))
        if len(by) == 2:
            a, b = by.values()
            agree = abs(a - b) <= RULER_TOL * max(a, b)
            print(f"       rulers {'agree' if agree else 'DISAGREE'} "
                  f"(|{a:.3f} - {b:.3f}| {'<=' if agree else '>'} {RULER_TOL:.0%})")
        else:
            agree = False
            print("       only one ruler read; not enough to apply a scale")
        refined = rr.refine_track(track, row_fit)
        num2, hsh2 = read_frames(video, refined, frames, reader)
        ys, yt, rl = row_readings(num2, hsh2)
        for name, true_y in (("numerals", rr.ROW_Y_M), ("hashes", rr.HASH_Y_M)):
            got = [abs(y) for y, r in zip(ys, rl) if r == name]
            print(f"       {name} re-read under the refined camera: median |y| "
                  f"{np.median(got) if got else float('nan'):.2f} m (rule book {true_y:.2f}), "
                  f"{len(got)} readings")
        h_after, _ = player_heights(refined, df, frames)
        print(f"player height under the refined camera: {h_after:.2f} m")
        readings = [r for rs in num2.values() for r in rs] or readings

    tf = yn.solve_transform(readings)
    if tf is None:
        raise CalibrationError("the numerals do not agree on a shift (one numeral, or "
                               "misreads); more frames or a view of two numerals needed")
    yards_in = (yn.GOAL_LINE_X_M - abs(tf.shift_m)) / yn.YARD_TO_M
    print(f"\nshift: {tf.shift_m:+.2f} m ({tf.shift_m / yn.YARD_TO_M:+.0f} yd); solved x = 0 "
          f"is the {yards_in:.0f}-yard line on the {'left' if tf.shift_m < 0 else 'right'} "
          f"half; votes {tf.votes:.1f} vs {tf.runner_up:.1f} over {tf.n_readings} readings")

    if args.los_yards is not None:
        got = snap_yards(yn.transform_track(refined, tf), df)
        print(f"line of scrimmage: formation at {got:.1f} yd from the nearest goal line; "
              f"play description says {args.los_yards:.0f} "
              f"({'ok' if abs(got - args.los_yards) <= 1.5 else 'MISMATCH'})")

    out = {"cam": args.cam, "shift_m": tf.shift_m, "turn": tf.turn, "votes": tf.votes,
           "runner_up": tf.runner_up, "n_readings": tf.n_readings,
           "row_scale": None if row_fit is None else
           {"scale": row_fit.scale, "offset": row_fit.offset,
            "residual_m": row_fit.residual_m, "by_ruler": row_fit.by_ruler,
            "rulers_agree": agree},
           "height_before_m": h_before, "frames": frames}
    (args.play_dir / "field_offset.json").write_text(json.dumps(out, indent=2))
    if not args.apply:
        print("written field_offset.json; --apply to rewrite cameras.npz")
        return
    if args.rows and not agree and not args.force:
        raise CalibrationError("refusing to apply the cross-field scale: the numeral and "
                               "hash rulers disagree (see 'by ruler'); --no-rows applies "
                               "the shift alone, --force overrides")
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
