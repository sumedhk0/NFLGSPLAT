#!/usr/bin/env python
"""Refine a play-dir's sideline camera track to the painted yard lines.

After scripts/08b exports per-frame cameras (paint solve on some frames,
interpolated between), this puts every frame's camera on the paint:
calibration.refine_paint refines rotation and focal length per frame with
the centre fixed, then smooths the deltas along the track. Prints the grid
distance (calibration.grid_fit) at sampled frames before and after and the
frame-to-frame rotation jitter, and refuses to write when the refinement
does not lower the median grid distance.

Play 1: 7-19 px -> 3-6 px over the clip, jitter p95 0.03 -> 0.10 degrees.

    python scripts/08e_refine_cameras.py --play-dir <P> [--cam sideline] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration import grid_fit as gf
from nfl_gsplat.calibration.cameras_io import load_camera_track, write_camera_track
from nfl_gsplat.calibration.refine_paint import refine_track
from nfl_gsplat.errors import CalibrationError


def sampled_grid(video, track, frames, orient_tol=None):
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            out.append(float("nan"))
            continue
        out.append(gf.grid_distance_px(img, track.K[f], track.R[f], track.t[f], orient_tol_deg=orient_tol)[0])
    cap.release()
    return np.asarray(out)


def jitter_deg(track, frames):
    from scipy.spatial.transform import Rotation

    a = [np.degrees(np.linalg.norm(Rotation.from_matrix(track.R[f] @ track.R[f - 1].T).as_rotvec()))
         for f in frames[1:]]
    return float(np.median(a)), float(np.percentile(a, 95))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--orient-tol", type=float, default=None,
                    help="score segments of any orientation within this many degrees of their projected "
                         "line (the endzone view, whose yard lines run across the image); default: the "
                         "sideline's near-vertical filter")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the refined cameras to this npz instead of cameras.npz (measure first)")
    args = ap.parse_args()

    tracks = load_camera_track(args.play_dir / "cameras.npz")
    track = tracks[args.cam]
    video = args.play_dir / f"{args.cam}.mp4"
    frames = [int(f) for f in np.flatnonzero(track.conf > 0)]
    if not frames:
        raise CalibrationError(f"no {args.cam} frame with a camera")
    probe = [int(x) for x in np.linspace(frames[0] + 5, frames[-1] - 5, 9)]

    refined, _after = refine_track(video, track, log_every=0, orient_tol_deg=args.orient_tol)
    before = sampled_grid(video, track, probe, args.orient_tol)
    after = sampled_grid(video, refined, probe, args.orient_tol)
    print("grid px at sampled frames (before -> after): " + " ".join(
        f"{f}:{b:.0f}->{a:.0f}" for f, b, a in zip(probe, before, after)))
    jb, ja = jitter_deg(track, frames), jitter_deg(refined, frames)
    print(f"median grid {np.nanmedian(before):.1f} -> {np.nanmedian(after):.1f} px; "
          f"frame-to-frame rotation median/p95 {jb[0]:.3f}/{jb[1]:.3f} -> {ja[0]:.3f}/{ja[1]:.3f} deg")
    if not (np.nanmedian(after) < np.nanmedian(before)):
        raise CalibrationError("refinement did not lower the grid distance; cameras.npz untouched")
    if args.dry_run:
        print("dry run; cameras.npz untouched")
        return
    if args.out is not None:
        tracks[args.cam] = refined
        write_camera_track(args.out, tracks, fps=59.94)
        print(f"refined {args.cam} written to {args.out}; cameras.npz untouched")
        return
    backup = args.play_dir / "cameras_unrefined.npz"
    if not backup.exists():
        shutil.copy(args.play_dir / "cameras.npz", backup)
    tracks[args.cam] = refined
    write_camera_track(args.play_dir / "cameras.npz", tracks, fps=59.94)
    print(f"cameras.npz rewritten ({args.cam} refined to the paint; original in {backup.name})")


if __name__ == "__main__":
    main()
