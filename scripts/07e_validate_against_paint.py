#!/usr/bin/env python
"""Check the helmet-fitted cameras against the PAINT, which never entered them.

Every other check on these cameras uses the same tracking that fitted them.
Holding players out breaks the worst of that circularity but not all of it: the
tracking is still the only source of truth in the loop. The painted field is a
completely separate witness -- yard lines were never given to the solve, in any
form -- so if the cameras are right they must predict where the paint appears.

There is one unknown in the comparison, and it is the useful part. The solve's
world z = 0 is the HELMET plane, while paint is on the turf some distance below
it, and that distance is never determined by the fit. So this sweeps it. If the
cameras are right the residual has a clear minimum, and the minimum lands at a
believable mean helmet height rather than wherever it likes.

Measured on 57583/82 sideline: 28.9 px at 0.0 m, falling to 12.8 px at 1.5 m and
rising again by 2.1 m. A 1.5 m mean helmet height is right for a mix of standing
players and crouching linemen, and nothing in the fit was free to put it there.

The residual is not a calibration score -- it compares a projected line against
a detected SEGMENT's midpoint, which is a crude metric, and the segment's
midpoint need not lie where the projected samples do. Read the shape of the
curve and where it bottoms out, not its depth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration import field_detect
from nfl_gsplat.calibration.decompose_homography import projection_matrix
from nfl_gsplat.calibration.field_landmarks import YARD_TO_M
from nfl_gsplat.calibration.from_helmets import cameras_fixed_centre
from nfl_gsplat.data.align_video import VIDEO_FPS, helmet_boxes_by_frame
from nfl_gsplat.data.helmet_dataset import load_labels, load_tracking

WIDTH, HEIGHT = 1280, 720
# Painted every 5 yards, and the sampled span stops short of the sidelines so a
# line is compared over the part of it a camera actually sees.
YARD_LINES_M = np.arange(-50.0, 50.1, 5.0) * YARD_TO_M
SPAN_Y_M = 24.0
HEIGHTS_M = (0.0, 0.8, 1.2, 1.5, 1.8, 2.1)


def nearest_projected_line(P, x_m, drop_m, seg, samples: int = 40) -> float:
    """Distance in px from a detected segment's midpoint to a projected line."""
    ys = np.linspace(-SPAN_Y_M, SPAN_Y_M, samples)
    pts = np.c_[np.full_like(ys, x_m), ys, np.full_like(ys, -drop_m)]
    q = np.c_[pts, np.ones(len(pts))] @ P.T
    uv = q[:, :2] / q[:, 2:3]
    mid = np.array([(seg.p0[0] + seg.p1[0]) / 2.0,
                    (seg.p0[1] + seg.p1[1]) / 2.0])
    return float(np.min(np.linalg.norm(uv - mid, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--game", type=int, default=57583)
    ap.add_argument("--play", type=int, default=82)
    ap.add_argument("--view", default="Sideline", choices=("Sideline", "Endzone"))
    ap.add_argument("--offset", type=float, default=-0.04)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()

    import cv2

    labels = load_labels(args.root / "train_labels.csv")
    plays = load_tracking(args.root / "train_player_tracking.csv")
    track = plays[(args.game, args.play)]
    index = {p: i for i, p in enumerate(track.players)}

    def world_at(frame):
        return track.at(args.offset + frame / VIDEO_FPS)

    name = f"{args.game}_{args.play:06d}_{args.view}.mp4"
    byf = helmet_boxes_by_frame(labels[labels["video"] == name], index)
    byf = {f: v for i, (f, v) in enumerate(sorted(byf.items())) if i % 10 == 0}
    cams, centre, _mirrored = cameras_fixed_centre(byf, world_at, WIDTH, HEIGHT)
    print(f"camera centre {np.round(centre, 1)} m from {len(cams)} frames\n")

    cap = cv2.VideoCapture(str(args.root / "video" / name))
    detected = {}
    for frame in sorted(cams)[:args.frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        ok, img = cap.read()
        if ok:
            detected[frame] = field_detect.detect_field_features(img).yard_lines
    cap.release()

    print("turf below the helmet plane -> agreement with the painted lines:")
    for drop in HEIGHTS_M:
        errs = [min(nearest_projected_line(projection_matrix(*cams[f]), x, drop, s)
                    for x in YARD_LINES_M)
                for f, segs in detected.items() for s in segs]
        if not errs:
            continue
        e = np.asarray(errs)
        print(f"   {drop:.1f} m: median {np.median(e):6.1f} px   "
              f"within 15 px {np.mean(e < 15):3.0%}   (n={len(e)})")


if __name__ == "__main__":
    main()
