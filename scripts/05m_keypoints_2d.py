"""05m: 2-D keypoints per tracked person in each view (YOLOv8-pose).

WHY. The 3-D joints have come from a monocular regressor per view, with
the calibrated cameras used only afterwards to place two depth guesses.
With 2-D keypoints in both views the cameras can triangulate the joints
directly (pose.triangulate); this stage supplies the 2-D evidence.

WHAT. Runs the pose detector on every frame of each camera's video,
assigns each detection to the tracked box it overlaps most (IoU >= 0.5,
so the keypoints inherit the global player id), and writes a long
parquet: frame, cam, global_player_id, joint (COCO index), x, y, conf.

USAGE:
  python scripts/05m_keypoints_2d.py --play-dir data/all22/<game>/play_001
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
COCO_JOINTS = 17
MIN_IOU = 0.5


def iou_matrix(a, b):
    """``[A, B]`` IoU of two box sets (x1, y1, x2, y2)."""
    a = np.asarray(a, float)[:, None, :]
    b = np.asarray(b, float)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def assign(det_boxes, track_boxes, *, min_iou: float = MIN_IOU):
    """Greedy one-to-one assignment detection -> track index by IoU (or -1)."""
    if not len(det_boxes) or not len(track_boxes):
        return np.full(len(det_boxes), -1, int)
    m = iou_matrix(det_boxes, track_boxes)
    out = np.full(len(det_boxes), -1, int)
    used = set()
    for flat in np.argsort(-m, axis=None):
        i, j = divmod(int(flat), m.shape[1])
        if m[i, j] < min_iou:
            break
        if out[i] >= 0 or j in used:
            continue
        out[i] = j
        used.add(j)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=Path("yolov8x-pose.pt"))
    ap.add_argument("--cams", nargs="+", default=["sideline", "endzone"])
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--out", type=Path, default=None, help="default <play-dir>/keypoints_2d.parquet")
    ap.add_argument("--limit", type=int, default=0, help="frames per camera (0 = all)")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    P = args.play_dir
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    model = YOLO(str(args.weights))
    rows = []
    t0 = time.time()
    for cam in args.cams:
        video = P / f"{cam}.mp4"
        if not video.exists():
            print(f"no {video}", file=sys.stderr)
            continue
        dv = df[df["cam"] == cam]
        by_frame = {int(f): g for f, g in dv.groupby("frame")}
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.limit:
            n = min(n, args.limit)
        matched = unmatched = 0
        for f in range(n):
            ok, bgr = cap.read()
            if not ok:
                break
            g = by_frame.get(f)
            if g is None or not len(g):
                continue
            res = model.predict(bgr, imgsz=args.imgsz, verbose=False, conf=0.25)[0]
            if res.keypoints is None or res.boxes is None or not len(res.boxes):
                continue
            det_boxes = res.boxes.xyxy.cpu().numpy()
            kp = res.keypoints.xy.cpu().numpy()                      # [D, 17, 2]
            kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None \
                else np.ones(kp.shape[:2], np.float32)
            tb = g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
            pids = g["global_player_id"].to_numpy()
            j = assign(det_boxes, tb)
            for d in range(len(det_boxes)):
                if j[d] < 0:
                    unmatched += 1
                    continue
                matched += 1
                pid = int(pids[j[d]])
                for k in range(COCO_JOINTS):
                    rows.append((f, cam, pid, k, float(kp[d, k, 0]), float(kp[d, k, 1]), float(kc[d, k])))
        cap.release()
        _LOG.info("%s: %d detections matched to tracks, %d unmatched, %.0f s", cam, matched, unmatched,
                  time.time() - t0)
    out = args.out or P / "keypoints_2d.parquet"
    kdf = pd.DataFrame(rows, columns=["frame", "cam", "global_player_id", "joint", "x", "y", "conf"])
    kdf.to_parquet(out, index=False)
    print(f"keypoints: {len(kdf) // COCO_JOINTS} person-frames over {kdf.frame.nunique()} frames -> {out} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
