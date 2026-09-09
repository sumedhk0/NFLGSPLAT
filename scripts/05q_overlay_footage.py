#!/usr/bin/env python
"""The avatars on the All-22 footage: skeletons projected into the sideline camera, per frame and per player.

WHY. Every ruler so far scored the caches; the user scores the clip
against the footage, and on v24 still sees arms all over the place. This
puts the two side by side: for each frame, the timeline's bodies (the same
states 05k renders, placed the same way) are projected through the
sideline camera onto the footage frame as bone skeletons in team colour,
with the 2-D keypoints as dots; a player strip shows one body over
consecutive frames, footage crop with the skeleton, so a limb that swings
where the real one does not is visible at the frame it happens.

USAGE (smplx env):
  python scripts/05q_overlay_footage.py --play-dir P --frames 100 200 300 --out diag/overlay
  python scripts/05q_overlay_footage.py --play-dir P --player 5 --start 120 --count 8 --step 2 --out diag/overlay
  python scripts/05q_overlay_footage.py --play-dir P --video --stride 2 --out diag/overlay   (an mp4 of every rendered frame)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.pose.forward_kinematics import SMPLX_BODY_PARENTS
from nfl_gsplat.render.play_timeline import load_play_timeline

COLOUR = {"KC": (40, 40, 230), "BAL": (240, 240, 240), "?": (0, 220, 220)}
KP_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]


def placed_joints(state, model):
    """The 22 body joints of a state's body in the world, placed as placed_vertices places it."""
    with torch.no_grad():
        res = model(betas=torch.tensor(state.betas[None, :model.num_betas].astype(np.float32)),
                    body_pose=torch.tensor(state.body_pose.reshape(1, -1).astype(np.float32)),
                    global_orient=torch.tensor(state.global_orient.reshape(1, 3).astype(np.float32)))
    verts = res.vertices[0].numpy().astype(np.float64)
    joints = res.joints[0].numpy().astype(np.float64)[:22]
    shift = np.array([state.xy[0] - joints[0, 0], state.xy[1] - joints[0, 1], -verts[:, 2].min()])
    return joints + shift


def project(track, f, X):
    K, R, t = track.K[f], track.R[f], track.t[f]
    p = (K @ (R @ np.asarray(X, float).T + t[:, None])).T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def draw_skeleton(img, uv, depth, colour, thickness=2):
    for j, par in enumerate(SMPLX_BODY_PARENTS):
        if par < 0 or j >= len(uv) or depth[j] <= 0 or depth[par] <= 0:
            continue
        a, b = uv[par], uv[j]
        if np.isfinite(a).all() and np.isfinite(b).all():
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, thickness, cv2.LINE_AA)
    for j in range(len(uv)):
        if depth[j] > 0 and np.isfinite(uv[j]).all():
            cv2.circle(img, (int(uv[j][0]), int(uv[j][1])), 2, colour, -1, cv2.LINE_AA)


def draw_keypoints(img, kp, colour=(0, 255, 0)):
    xy = kp[["x", "y"]].to_numpy(float)
    conf = kp["conf"].to_numpy(float)
    joint = kp["joint"].to_numpy(int)
    by = {int(j): (xy[i], conf[i]) for i, j in enumerate(joint)}
    for a, b in KP_EDGES:
        if a in by and b in by and by[a][1] >= 0.3 and by[b][1] >= 0.3:
            cv2.line(img, tuple(int(v) for v in by[a][0]), tuple(int(v) for v in by[b][0]), colour, 1, cv2.LINE_AA)
    for j, (p, c) in by.items():
        if c >= 0.3:
            cv2.circle(img, (int(p[0]), int(p[1])), 2, colour, -1, cv2.LINE_AA)


def main() -> None:
    import smplx

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="frames to overlay (full frame PNGs)")
    ap.add_argument("--player", type=int, default=None, help="a strip of this id over consecutive frames")
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--video", action="store_true", help="every rendered frame to an mp4")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--no-keypoints", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    args.out.mkdir(parents=True, exist_ok=True)
    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral", num_betas=10, use_pca=False,
                         batch_size=1)
    tl, tracks, df, frames_all, poses = load_play_timeline(P, model)
    track = tracks[args.cam]
    import pickle

    ident = pickle.load(open(P / "identity_resolved.pkl", "rb")).get("merged", {}) if (P / "identity_resolved.pkl").exists() else {}
    team_of = {int(p): m.team for p, m in ident.items()}
    kdf = None
    if not args.no_keypoints and (P / "keypoints_2d.parquet").exists():
        kdf = pd.read_parquet(P / "keypoints_2d.parquet")
        kdf = kdf[kdf["cam"] == args.cam]
    cap = cv2.VideoCapture(str(P / f"{args.cam}.mp4"))
    boxes = df[df["cam"] == args.cam].set_index(["frame", "track_id"])

    def frame_image(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        return img if ok else None

    def overlay(f, only=None):
        img = frame_image(f)
        if img is None or f >= len(track.conf) or track.conf[f] <= 0:
            return None, {}
        proj = {}
        for s in tl.states.get(f, []):
            if only is not None and s.pid != only:
                continue
            J = placed_joints(s, model)
            uv, depth = project(track, f, J)
            proj[s.pid] = (uv, depth)
            draw_skeleton(img, uv, depth, COLOUR.get(team_of.get(s.pid, "?"), COLOUR["?"]))
            if depth[15] > 0 and np.isfinite(uv[15]).all():
                cv2.putText(img, str(s.pid), (int(uv[15][0]) + 4, int(uv[15][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 0), 1, cv2.LINE_AA)
        if kdf is not None:
            k = kdf[kdf["frame"] == f]
            for pid, g in k.groupby("global_player_id"):
                if only is not None and int(pid) != only:
                    continue
                draw_keypoints(img, g)
        cv2.putText(img, f"f{f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img, proj

    if args.player is not None:
        pid = args.player
        tiles = []
        for f in range(args.start, args.start + args.count * args.step, args.step):
            img, proj = overlay(f, only=pid)
            if img is None:
                continue
            if (f, pid) in boxes.index:
                b = boxes.loc[(f, pid)]
                cx, cy = int((b.bbox_x1 + b.bbox_x2) / 2), int((b.bbox_y1 + b.bbox_y2) / 2)
            elif pid in proj and np.isfinite(proj[pid][0][0]).all():
                cx, cy = int(proj[pid][0][0][0]), int(proj[pid][0][0][1])
            else:
                continue
            h = 260
            x0, y0 = max(0, min(cx - h // 2, img.shape[1] - h)), max(0, min(cy - h // 2, img.shape[0] - h))
            crop = img[y0:y0 + h, x0:x0 + h].copy()
            crop = cv2.resize(crop, (390, 390), interpolation=cv2.INTER_CUBIC)
            cv2.putText(crop, f"f{f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(crop)
        if not tiles:
            raise SystemExit(f"id {pid}: no frames")
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        out = args.out / f"player_{pid}_f{args.start}.jpg"
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {out} ({len(tiles)} frames)")
        return

    frames = args.frames if args.frames else (frames_all[:: max(1, args.stride)] if args.video else frames_all[::80])
    writer = None
    for f in frames:
        img, _ = overlay(f)
        if img is None:
            continue
        if args.video:
            if writer is None:
                writer = cv2.VideoWriter(str(args.out / f"overlay_{args.cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         59.94 / max(1, args.stride), (img.shape[1], img.shape[0]))
            writer.write(img)
        else:
            cv2.imwrite(str(args.out / f"overlay_f{f:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if writer is not None:
        writer.release()
        print(f"wrote {args.out / f'overlay_{args.cam}.mp4'} ({len(frames)} frames)")
    else:
        print(f"wrote {len(frames)} overlay frames to {args.out}")


if __name__ == "__main__":
    main()
