#!/usr/bin/env python
"""Build the pseudo-labelled 2D pose dataset for one play (pose.pseudo_labels) in YOLO-pose format, with a held-out
frame set, and print what the labels are made of.

USAGE (smplx312 venv: the pose caches are numpy-1 pickles, the body model is needed):
  python scripts/09g_pose_labels.py --play-dir P --out DATASET_DIR [--lo 215 --hi 615] [--stride 2] [--val-every 5]

Writes DATASET_DIR/images/{train,val}/<cam>_<frame>.jpg, labels/{train,val}/<cam>_<frame>.txt (one line per tracked
box: class, box, 17 keypoints with visibility), data.yaml, and labels.parquet (cam, clip_frame, pid, joint, u, v, vis,
source) for the rulers. Every ``val_every``-th sideline frame (and its endzone frame) is validation: never trained on,
and the set the fine-tuned detector is scored on against these labels and against the baseline detector.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.pose import pseudo_labels as pl  # noqa: E402
from nfl_gsplat.render.play_timeline import clip_offset  # noqa: E402

BODY = "data/body_models"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=int, default=215)
    ap.add_argument("--hi", type=int, default=615)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--agree-px", type=float, default=pl.AGREE_PX)
    ap.add_argument("--anchor-conf", type=float, default=pl.ANCHOR_CONF)
    ap.add_argument("--no-images", action="store_true", help="labels and parquet only")
    a = ap.parse_args()
    import smplx
    import torch

    P, out = Path(a.play_dir), Path(a.out)
    model = smplx.create(BODY, model_type="smplx", gender="neutral", num_betas=10, use_pca=False, batch_size=1)
    tracks = load_camera_track(P / "cameras.npz")
    refit = pickle.load(open(P / "poses_refit.json", "rb"))["frames"]
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    offset = clip_offset(P)
    cams = ["sideline", "endzone"]
    clip_of = {"sideline": lambda f: f, "endzone": lambda f: f + offset}      # sideline f sits beside endzone f + offset
    det_at = {(str(r.cam), int(r.frame), int(r.global_player_id)): None for r in kdf[["cam", "frame", "global_player_id"]].drop_duplicates().itertuples()}
    kg = kdf.groupby(["cam", "frame", "global_player_id"])
    boxes = {(str(r.cam), int(r.frame), int(r.global_player_id)): (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2) for r in df.itertuples()}

    def det(cam, cf, pid):
        key = (cam, cf, pid)
        if key not in det_at:
            return None
        g = kg.get_group(key).sort_values("joint")
        if len(g) != pl.N_COCO:
            return None
        return g[["x", "y"]].to_numpy(float), g["conf"].to_numpy(float)

    def joints(rec):
        with torch.no_grad():
            res = model(betas=torch.tensor(np.asarray(rec["betas"])[None, :10].astype(np.float32)),
                        body_pose=torch.tensor(np.asarray(rec["body_pose"]).reshape(1, -1).astype(np.float32)),
                        global_orient=torch.tensor(np.asarray(rec["global_orient"]).reshape(1, 3).astype(np.float32)),
                        transl=torch.tensor(np.asarray(rec["transl"]).reshape(1, 3).astype(np.float32)))
        return res.joints[0, :22].numpy().astype(np.float64)

    # the cache's own keyframes in the window (the fits sit on even frames; a range from an odd start missed them all)
    frames = sorted(int(f) for f in refit if a.lo <= int(f) <= a.hi and (int(f) - a.lo) % a.stride == 0 or a.lo <= int(f) <= a.hi and a.stride == 1)
    if not frames:
        frames = sorted(int(f) for f in refit if a.lo <= int(f) <= a.hi)
    rows = []
    per_image: dict = {}                       # (cam, clip_frame) -> list of (box, uv, vis)
    counts = {s: 0 for s in pl.SOURCES}
    for f in frames:
        recs = refit.get(f, {})
        for pid, rec in recs.items():
            pid = int(pid)
            proj, dets = {}, {}
            for cam in cams:
                cf = clip_of[cam](f)
                tr = tracks[cam]
                if cf < 0 or cf >= len(tr.conf) or tr.conf[cf] <= 0:
                    continue
                intr, pose = tr.at(cf)
                proj[cam] = pl.coco_projection(joints(rec), intr.K(), np.asarray(pose.R, float), np.asarray(pose.t, float))
                d = det(cam, cf, pid)
                if d is not None:
                    dets[cam] = d
            if not proj:
                continue
            lab = pl.label_frame(proj, dets, agree_px=a.agree_px, anchor_conf=a.anchor_conf)
            for cam in lab.uv:
                cf = clip_of[cam](f)
                box = boxes.get((cam, cf, pid))
                if box is None:
                    continue
                per_image.setdefault((cam, cf), []).append((box, lab.uv[cam], lab.vis[cam]))
                for k in range(pl.N_COCO):
                    src = pl.SOURCES[lab.source[cam][k]]
                    counts[src] += 1
                    rows.append((cam, cf, pid, k, lab.uv[cam][k, 0], lab.uv[cam][k, 1], int(lab.vis[cam][k]), src))
    # tracked boxes with no fit: box-only instances (all keypoints unlabelled), so the detector still learns the box
    for (cam, cf, pid), box in boxes.items():
        if (cam, cf) in per_image and all(b is not box for b, _u, _v in per_image[(cam, cf)]):
            if not any(np.allclose(b, box) for b, _u, _v in per_image[(cam, cf)]):
                per_image[(cam, cf)].append((box, np.full((pl.N_COCO, 2), np.nan), np.zeros(pl.N_COCO, int)))
    ldf = pd.DataFrame(rows, columns=["cam", "clip_frame", "pid", "joint", "u", "v", "vis", "source"])
    out.mkdir(parents=True, exist_ok=True)
    ldf.to_parquet(out / "labels.parquet", index=False)
    total = sum(counts.values())
    print(f"labelled keypoint slots: {total}; " + ", ".join(f"{s} {n} ({100 * n / max(1, total):.1f} %)" for s, n in counts.items()))
    cross = ldf[ldf.source == "fit_cross"]
    print(f"fit_cross labels by camera: {cross.groupby('cam').size().to_dict()}; by joint: {cross.groupby('joint').size().to_dict()}")
    # the split: every val_every-th sideline frame, with its endzone frame
    val_side = {f for i, f in enumerate(frames) if i % a.val_every == a.val_every - 1}
    is_val = lambda cam, cf: (cf if cam == "sideline" else cf - offset) in val_side
    n_img = {"train": 0, "val": 0}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    caps = {cam: cv2.VideoCapture(str(P / f"{cam}.mp4")) for cam in cams}
    size = {}
    for cam in cams:
        size[cam] = (int(caps[cam].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[cam].get(cv2.CAP_PROP_FRAME_HEIGHT)))
    for (cam, cf), inst in sorted(per_image.items()):
        split = "val" if is_val(cam, cf) else "train"
        W, H = size[cam]
        name = f"{cam}_{cf:05d}"
        lines = [pl.yolo_pose_line(box, uv, vis, W, H) for box, uv, vis in inst]
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
        if not a.no_images:
            caps[cam].set(cv2.CAP_PROP_POS_FRAMES, cf)
            ok, bgr = caps[cam].read()
            if ok:
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_img[split] += 1
    for cap in caps.values():
        cap.release()
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [17, 3]\n"
        f"flip_idx: {pl.COCO_FLIP}\nnames:\n  0: player\n")
    print(f"images: train {n_img['train']}, val {n_img['val']} (val = every {a.val_every}th sideline frame and its endzone frame); "
          f"instances {sum(len(v) for v in per_image.values())}; wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
