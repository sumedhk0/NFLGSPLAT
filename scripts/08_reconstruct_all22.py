#!/usr/bin/env python
"""Reconstruct one production All-22 play end to end, from the video alone.

Every stage of this has been measured on its own; this is where they meet on the
footage the project exists for. The chain is:

    paint    -> sideline camera candidates    calibrate_clip, no tracking; kept
                                              only if they can see the field
    YOLO     -> players per view              feet = bottom-centre of the box
    players  -> endzone camera                from_players: the sideline puts
                                              each player on the turf, the
                                              endzone sees the same people, and
                                              that is a plane homography
    agreement -> which sideline candidate     the pair that reconciles the most
                                              players, in metres
    placement -> 3D positions                 ground point per player

WHY THE ENDZONE IS NOT SOLVED FROM PAINT. It cannot be, and this was measured
rather than assumed: twenty-four paint solves of the endzone clip produced two
cameras, both impossible, and neither reconciled a single player with any
sideline candidate. Looking down the field, the hash marks are receding strings
of dots rather than rows, and the far half is a few dozen pixels.

THERE IS NO GROUND TRUTH HERE, which is the point. So the result is checked
against things that must be true regardless:

    reconciled    two independent cameras put the same players in the same
                  places -- about 22 of them, not 3
    coverage      an All-22 camera sees most of the field, not a fifth of it
    on the field  placements land inside the boundary
    head height   with feet on the turf, each view's head ray implies a height;
                  it must come out near 1.85 m in BOTH views, and nothing in the
                  pipeline forces it to
    spacing       real players stand about 1-3 m apart

The two clips are assumed to start together; they are the same play at the same
length. Nothing here verifies the offset, and an offset sweep on this play found
none.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.calibrate_clip import candidates_for_video
from nfl_gsplat.calibration.coverage import field_coverage
from nfl_gsplat.calibration.from_players import solve_second_view
from nfl_gsplat.calibration.joint_views import match_count
from nfl_gsplat.calibration.player_scale import implied_heights
from nfl_gsplat.errors import CalibrationError

EXPECTED_PLAYER_M = 1.85
FIELD_HALF_X_M = 56.0
FIELD_HALF_Y_M = 26.0


def boxes_by_frame(model, path, frames, *, conf=0.15):
    """Person boxes on the requested frames, plus the frame size."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        res = model.predict(img, classes=[0], conf=conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        out[int(f)] = res.boxes.xyxy.cpu().numpy()
    cap.release()
    return out, w, h


def feet_of(boxes):
    return {f: np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])
            for f, b in boxes.items()}


def nearest(cams, f):
    return cams[min(cams, key=lambda k: abs(k - f))]


def fov_deg(K, width):
    return float(2.0 * np.degrees(np.arctan(width / (2.0 * K[0, 0]))))


def frame_count(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("data/all22/bal_at_kc_2024_wk1"))
    ap.add_argument("--sideline", default="001_Sideline_KC_2-20_BLT_24.mp4")
    ap.add_argument("--endzone", default="002_Endzone_KC_2-20_BLT_24.mp4")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--calib-frames", type=int, default=26)
    ap.add_argument("--play-frames", type=int, default=16)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--out", type=Path,
                    default=Path("C:/Users/sumedh/diag/all22_reconstruction.npz"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    side_path = args.root / args.sideline
    end_path = args.root / args.endzone

    print(f"Sideline: {args.sideline}")
    cands = candidates_for_video(side_path, attempts=args.attempts,
                                 n_frames=args.calib_frames, model=model)
    if not cands:
        raise SystemExit("no sideline camera could see the field")
    print(f"   {len(cands)} candidate cameras that can see the field")

    # The same moments in both clips, away from the ends of the play.
    total = min(frame_count(side_path), frame_count(end_path))
    want = np.linspace(int(0.2 * total), int(0.8 * total),
                       args.play_frames).astype(int)

    boxes_s, _w_s, _h_s = boxes_by_frame(model, side_path, want)
    boxes_e, w_e, h_e = boxes_by_frame(model, end_path, want)
    feet_s, feet_e = feet_of(boxes_s), feet_of(boxes_e)
    print(f"   detections per frame: sideline "
          f"{np.median([len(b) for b in boxes_s.values()]):.0f}, endzone "
          f"{np.median([len(b) for b in boxes_e.values()]):.0f}")

    # Choose the sideline candidate by what the OTHER view can reconcile.
    best = None
    for i, cand in enumerate(cands):
        cams_s = cand["cams"]
        try:
            K, R, t, inliers = solve_second_view(cams_s, feet_s, feet_e,
                                                 w_e, h_e)
        except CalibrationError as exc:
            print(f"   [{i}] centre {np.round(cand['centre'], 1)} "
                  f"{cand['quality']['fov_deg']:.1f} deg: {str(exc)[:70]}")
            continue
        cam_e = (K, R, t)
        per = [match_count(nearest(cams_s, f), feet_s[f], cam_e, feet_e[f])[0]
               for f in want if f in feet_s and f in feet_e]
        score = float(np.median(per)) if per else 0.0
        print(f"   [{i}] centre {np.round(cand['centre'], 1)} "
              f"{cand['quality']['fov_deg']:.1f} deg, sees "
              f"{100 * cand['quality']['coverage']:.0f}% -> endzone "
              f"{np.round(-R.T @ t, 1)} {fov_deg(K, w_e):.1f} deg, "
              f"{inliers} inliers, reconciles {score:.0f}/frame")
        if best is None or score > best[0]:
            best = (score, cand, cam_e, per)
    if best is None:
        raise SystemExit("no sideline candidate let the players calibrate "
                         "the endzone")
    score, cand, cam_e, per = best
    cams_s = cand["cams"]
    K_e, R_e, t_e = cam_e

    print("\nchosen cameras")
    print(f"   sideline {np.round(cand['centre'], 1)} m, "
          f"{cand['quality']['fov_deg']:.1f} deg, sees "
          f"{100 * cand['quality']['coverage']:.0f}% of the field")
    cov_e = field_coverage(K_e, R_e, t_e, w_e, h_e)
    print(f"   endzone  {np.round(-R_e.T @ t_e, 1)} m, "
          f"{fov_deg(K_e, w_e):.1f} deg, sees {100 * cov_e:.0f}% of the field")

    # Place every reconciled player, and check what needs no ground truth.
    placed_all, heights_s, heights_e, spacing = [], [], [], []
    for f in want:
        if f not in feet_s or f not in feet_e:
            continue
        n, placed = match_count(nearest(cams_s, f), feet_s[f], cam_e, feet_e[f])
        if n:
            placed_all.append(placed)
            if n > 2:
                d = np.linalg.norm(placed[:, None] - placed[None], axis=2)
                np.fill_diagonal(d, np.inf)
                spacing.append(np.median(d.min(1)))
        heights_s.extend(implied_heights(*nearest(cams_s, f), boxes_s[f]))
        heights_e.extend(implied_heights(K_e, R_e, t_e, boxes_e[f]))

    if not placed_all:
        raise SystemExit("the chosen pair placed nobody")
    allxy = np.concatenate(placed_all)
    inside = ((np.abs(allxy[:, 0]) <= FIELD_HALF_X_M)
              & (np.abs(allxy[:, 1]) <= FIELD_HALF_Y_M))
    hs, he = np.asarray(heights_s), np.asarray(heights_e)
    hs = hs[(hs > 0.5) & (hs < 4.0)]
    he = he[(he > 0.5) & (he < 4.0)]

    print("\nchecks that need no ground truth")
    print(f"   reconciled        median {score:.0f} players per frame  {per}")
    print(f"   on the field      {inside.mean():5.0%} of placements")
    print(f"   player height     sideline median {np.median(hs):4.2f} m, "
          f"endzone median {np.median(he):4.2f} m  (expected ~{EXPECTED_PLAYER_M})")
    print(f"   footprint         x {allxy[:, 0].min():6.1f}..{allxy[:, 0].max():6.1f}"
          f"   y {allxy[:, 1].min():6.1f}..{allxy[:, 1].max():6.1f} m")
    if spacing:
        print(f"   nearest neighbour median {np.median(spacing):4.2f} m "
              "(real players stand about 1-3 m apart)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fr = sorted(cams_s)
    np.savez(args.out,
             side_frames=np.array(fr),
             side_K=np.array([cams_s[f][0] for f in fr]),
             side_R=np.array([cams_s[f][1] for f in fr]),
             side_t=np.array([cams_s[f][2] for f in fr]),
             end_K=K_e, end_R=R_e, end_t=t_e,
             frames=np.array([f for f in want if f in feet_s and f in feet_e]),
             reconciled=np.array(per),
             placed=allxy)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
