#!/usr/bin/env python
"""The football's flight read off the sideline film, and a check of ball.json against it.

    C:/venvs/smplx312/Scripts/python scripts/09a_ball_in_film.py --play-dir P [--start 500 --end 620] [--sheet OUT.png]

WHY. 08y places the ball from three hand-typed inputs (release, catch, receiver) and for a day on play 1
(2026-09-18) the receiver was the wrong man; nothing in the pipeline looked at the film for the ball. The
ball in flight is a 10-20 px blob that moves against the camera-compensated background, outside every
player's box, along a near-straight image line at 10-40 px per frame -- and the film shows it on about
half the flight frames (play 1: 539-558 of 527-562; near the hands and the pocket it sits inside a box).

HOW. For each frame pair (f-1, f): an ECC affine alignment of the earlier frame onto the later (the
camera pans a few px a frame), the absolute difference, a threshold, connected components of 8-400 px
outside the dilated player boxes and outside the broadcast graphics; then tracking.ball_film.fit_flight
(RANSAC line, grown along a quadratic, dense inliers, a box at each end) and name_ends (walk the track
into the passer's box and, among the boxes at the other end, his teammate's: in tight coverage the
defender's box holds the catch point too). Prints the flight, the ends and, when ball.json exists, whether its release/catch/receiver agree:
the receiver named here must be ball.json's, or the ball is going to the wrong man.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.errors import SetupError  # noqa: E402
from nfl_gsplat.tracking.ball_film import fit_flight, name_ends, nearest_box, track_at  # noqa: E402

THRESH = 28          # grey-level difference a moving blob must exceed
PAD = 12             # px a player box is grown by before a blob inside it is dropped
MIN_AREA, MAX_AREA, MAX_SIDE = 8, 120, 24      # play 1: the ball 10-50 px, a far player's moving limbs bigger


def candidates(video: Path, f0: int, f1: int, boxes: dict, *, thresh: int = THRESH, pad: int = PAD, masks=()) -> dict:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
    ok, prev = cap.read()
    if not ok:
        raise SetupError(f"09a: cannot read frame {f0 - 1} of {video}")
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
    out = {}
    for f in range(f0, f1 + 1):
        ok, im = cap.read()
        if not ok:
            break
        cur = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(cur, prev, warp, cv2.MOTION_AFFINE, crit, None, 5)
            prev_w = cv2.warpAffine(prev, warp, (cur.shape[1], cur.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            prev_w = prev
        d = cv2.absdiff(cur, prev_w)
        for r0, r1, c0, c1 in masks:
            d[r0:r1, c0:c1] = 0
        m = cv2.morphologyEx((d > thresh).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        bx = np.array([b[1:] for b in boxes.get(f, [])], float).reshape(-1, 4)
        cs = []
        for i in range(1, n):
            _x, _y, w, h, area = stats[i]
            if not (MIN_AREA <= area <= MAX_AREA) or max(w, h) > MAX_SIDE or max(w, h) / max(1, min(w, h)) > 4:
                continue
            cx, cy = cent[i]
            if len(bx) and np.any((cx >= bx[:, 0] - pad) & (cx <= bx[:, 2] + pad) & (cy >= bx[:, 1] - pad) & (cy <= bx[:, 3] + pad)):
                continue
            cs.append((float(cx), float(cy), int(area)))
        out[f] = cs
        prev = cur
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--start", type=int, default=None, help="first frame scanned (default: the snap from play_end.json)")
    ap.add_argument("--end", type=int, default=None, help="last frame scanned (default: the play's end)")
    ap.add_argument("--thresh", type=int, default=THRESH)
    ap.add_argument("--sheet", type=Path, default=None, help="a contact sheet of the flight with the track drawn")
    ap.add_argument("--ball", type=Path, default=None, help="the ball path to check (default <play-dir>/ball.json)")
    args = ap.parse_args()
    P = args.play_dir
    pe = json.loads((P / "play_end.json").read_text()) if (P / "play_end.json").exists() else {}
    f0 = args.start if args.start is not None else int(pe.get("snap", 0)) + 1
    f1 = args.end if args.end is not None else int(pe.get("end", f0 + 300))
    t = pd.read_parquet(P / "tracks.parquet")
    t = t[(t["cam"] == "sideline") & (t["track_id"] >= 0)]
    boxes = {int(f): [(int(r.global_player_id), float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2))
                      for r in g.itertuples()] for f, g in t.groupby("frame")}
    h = int(cv2.VideoCapture(str(P / "sideline.mp4")).get(cv2.CAP_PROP_FRAME_HEIGHT))
    masks = [(0, 150, 0, 4000), (h - 60, h, 0, 4000), (0, 220, 0, 420)]      # score bug, near sideline, the NFL PRO graphic
    cands = candidates(P / "sideline.mp4", f0, f1, boxes, thresh=args.thresh, masks=masks)
    n_c = sum(len(v) for v in cands.values())
    print(f"09a: {len(cands)} frames {f0}-{f1} scanned, {n_c} moving blobs outside the boxes")
    ident = P / "identity_resolved.pkl"
    team_of = {}
    if ident.exists():
        import pickle

        team_of = {int(k): getattr(v, "team", None) for k, v in pickle.load(open(ident, "rb")).get("merged", {}).items()}
    fl = fit_flight(cands, boxes)
    if fl is None:
        print("no flight: nothing moves like a ball outside the boxes (the pass may stay inside boxes, or the threshold is off)")
        return
    ends = name_ends(fl, boxes, teams=team_of or None)
    x0, y0 = track_at(fl, fl["frames"][0]); x1, y1 = track_at(fl, fl["frames"][-1])
    print(f"flight seen on {fl['n']} frames {fl['frames'][0]}-{fl['frames'][-1]} at {fl['speed']:.1f} px/frame, "
          f"from ({x0:.0f}, {y0:.0f}) to ({x1:.0f}, {y1:.0f}); frames with a blob on the line: {fl['frames']}")
    print(f"ends: leaves box of id {ends['passer']} at {ends['release']} (release); enters box of id {ends['receiver']} at {ends['catch']} (catch)"
          + (f"; other boxes holding the catch point: {ends['others']}" if ends.get('others') else ""))
    bj = args.ball if args.ball is not None else P / "ball.json"
    if bj.exists():
        b = json.loads(bj.read_text())
        rel, cat, rec, pas = b.get("release"), b.get("catch"), b.get("receiver"), b.get("passer")
        print(f"{bj.name}: release {rel}, catch {cat}, passer {pas}, receiver {rec}")
        verdict = []
        if ends["receiver"] is not None and rec is not None:
            verdict.append("receiver AGREES" if int(ends["receiver"]) == int(rec) else f"receiver DISAGREES: the film's flight enters id {ends['receiver']}, ball.json says {rec}")
        if cat is not None:
            x, y = track_at(fl, int(cat))
            pid, d = nearest_box(boxes.get(int(cat), []), x, y)
            own = [bx for bx in boxes.get(int(cat), []) if bx[0] == rec]
            d_own = nearest_box(own, x, y)[1] if own else float("inf")
            verdict.append(f"at ball.json's catch frame {cat} the film's track is at ({x:.0f}, {y:.0f}): nearest box id {pid} at {d:.0f} px, "
                           f"ball.json's receiver {rec} at {d_own:.0f} px")
        if ends["release"] is not None and rel is not None:
            verdict.append(f"release {'AGREES' if abs(int(ends['release']) - int(rel)) <= 3 else 'DISAGREES'} ({ends['release']} vs {rel})")
        if ends["catch"] is not None and cat is not None:
            verdict.append(f"catch {'AGREES' if abs(int(ends['catch']) - int(cat)) <= 4 else 'DISAGREES'} ({ends['catch']} vs {cat})")
        for v in verdict:
            print("  " + v)
    if args.sheet:
        cap = cv2.VideoCapture(str(P / "sideline.mp4"))
        fa = (ends["release"] or fl["frames"][0]) - 2; fb = (ends["catch"] or fl["frames"][-1]) + 2
        xs = [track_at(fl, f)[0] for f in range(fa, fb + 1)]; ys = [track_at(fl, f)[1] for f in range(fa, fb + 1)]
        c0, c1 = int(max(0, min(xs) - 160)), int(min(1920, max(xs) + 160)); r0, r1 = int(max(0, min(ys) - 140)), int(min(1080, max(ys) + 140))
        tiles = []
        seen = set(fl["frames"])
        for f in range(fa, fb + 1, max(1, (fb - fa) // 11)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
            if not ok:
                continue
            x, y = track_at(fl, f)
            cv2.circle(im, (int(x), int(y)), 16, (0, 255, 255) if f in seen else (0, 165, 255), 2)
            crop = im[r0:r1, c0:c1].copy()
            cv2.putText(crop, f"{f}{'' if f in seen else ' (extrapolated)'}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(crop)
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.sheet), sheet)
        print(f"wrote {args.sheet} {sheet.shape}")


if __name__ == "__main__":
    main()
