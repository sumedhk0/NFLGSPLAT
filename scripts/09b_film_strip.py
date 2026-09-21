#!/usr/bin/env python
"""Zoomed strips of the RAW footage with the tracker's boxes and ids drawn: the ruler for identity questions.

WHY. Every identity call made on 2026-09-19 (who the centre is under three endzone ids, which of four KC ids sit on
jerseys 65 and 74, which Raven goes on under 40 -> 198 -> 206, whether a held man is still where his hold draws him)
was settled by looking at the raw frames with the boxes and ids drawn, never by a ruler on the timeline. This is that
tool, kept: a fixed image window (a box, or an id's box on a frame) cropped and zoomed over a list of frames, every
box on each frame drawn thin and labelled with its global id (same team as the anchor id yellow, other team magenta,
unknown grey), the anchor window in red. The endzone clip runs clip_offset.json frames off the sideline's, so a
timeline frame f is endzone footage frame f + offset (05q's convention).

USAGE (nflgsplat env, reads tracks.parquet and the clips only):
  python scripts/09b_film_strip.py --play-dir P --cam sideline --id 204 --at 499 --frames 499 515 531 547 --out diag/x.png
  python scripts/09b_film_strip.py --play-dir P --cam endzone --box 737,417,838,639 --frames 500 515 530 --out diag/y.png
  --radius 150 --zoom 2.5 (crop half-size in px, zoom factor); --cols 4
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--cam", default="sideline", choices=["sideline", "endzone"])
    ap.add_argument("--id", type=int, default=None, help="anchor the window on this global id's box ...")
    ap.add_argument("--at", type=int, default=None, help="... on this frame (default: the first of --frames)")
    ap.add_argument("--box", default=None, help="or anchor on an image box x1,y1,x2,y2")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--radius", type=int, default=150)
    ap.add_argument("--zoom", type=float, default=2.5)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    P = a.play_dir
    off_file = P / "clip_offset.json"
    offset = {"sideline": 0, "endzone": int(json.loads(off_file.read_text())["offset"]) if off_file.exists() else 0}[a.cam]
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["cam"] == a.cam]
    teams = {}
    ident = P / "identity_resolved.pkl"
    if ident.exists():
        try:
            teams = {int(p): v.team for p, v in pickle.load(open(ident, "rb"))["merged"].items()}
        except Exception as e:  # noqa: BLE001
            print(f"no teams from {ident.name}: {e}", file=sys.stderr)
    if a.box:
        bx = [int(v) for v in a.box.split(",")]
    elif a.id is not None:
        f0 = a.at if a.at is not None else a.frames[0]
        g = df[(df["frame"] == f0) & (df["global_player_id"] == a.id)]
        if not len(g):
            raise SystemExit(f"id {a.id} has no {a.cam} box on frame {f0}")
        r = g.iloc[0]
        bx = [int(r.bbox_x1), int(r.bbox_y1), int(r.bbox_x2), int(r.bbox_y2)]
    else:
        raise SystemExit("give --id (with --at) or --box")
    anchor_team = teams.get(a.id) if a.id is not None else None
    cap = cv2.VideoCapture(str(P / f"{a.cam}.mp4"))
    cx, cy = (bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2
    R, Z = a.radius, a.zoom
    tiles = []
    for f in a.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f + offset)
        ok, img = cap.read()
        if not ok:
            print(f"no footage frame {f + offset}", file=sys.stderr)
            continue
        x0, y0 = max(0, cx - R), max(0, cy - R)
        crop = cv2.resize(img[y0:y0 + 2 * R, x0:x0 + 2 * R].copy(), None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
        for r in df[df["frame"] == f].itertuples():
            q = int(r.global_player_id)
            col = (0, 255, 255) if anchor_team and teams.get(q) == anchor_team else ((255, 0, 255) if teams.get(q) else (200, 200, 200))
            p1 = (int((r.bbox_x1 - x0) * Z), int((r.bbox_y1 - y0) * Z)); p2 = (int((r.bbox_x2 - x0) * Z), int((r.bbox_y2 - y0) * Z))
            cv2.rectangle(crop, p1, p2, col, 1)
            cv2.putText(crop, str(q), (p1[0], max(14, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        p1 = (int((bx[0] - x0) * Z), int((bx[1] - y0) * Z)); p2 = (int((bx[2] - x0) * Z), int((bx[3] - y0) * Z))
        cv2.rectangle(crop, p1, p2, (0, 0, 255), 2)
        cv2.putText(crop, f"{a.cam} f{f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        T = int(round(2 * R * Z))                        # cv2.resize rounds its own size: pad to one fixed tile
        pad = np.zeros((T, T, 3), np.uint8)
        h, w = min(T, crop.shape[0]), min(T, crop.shape[1])
        pad[:h, :w] = crop[:h, :w]
        tiles.append(pad)
    if not tiles:
        raise SystemExit("no tiles")
    cols = min(a.cols, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i * cols:(i + 1) * cols]) for i in range(rows)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), sheet)
    print(f"wrote {a.out} {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
