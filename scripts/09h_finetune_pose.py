#!/usr/bin/env python
"""Fine-tune the 2D pose detector on one play's pseudo-labelled dataset (09g) and score it on the held-out frames
against the baseline detector, by label source.

USAGE (nflgsplat venv, GPU):
  python scripts/09h_finetune_pose.py --dataset P/pose_ds --weights yolov8x-pose.pt --out P/pose_ft [--epochs 30]
      [--imgsz 1920] [--batch 2] [--freeze 10] [--score-only]

The ruler (printed and written to <out>/score.json): on the validation images, the keypoint error to the label per
camera and per label source -- det (the detector's own where confident), fit_self (the fit anchored by this camera),
fit_cross (anchored by the other camera: the labels the baseline detector could not have produced). A fine-tune that
does not raise PCK on fit_cross without lowering it on det has learned nothing worth keeping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.pose import pseudo_labels as pl  # noqa: E402


def score(weights: Path, dataset: Path, *, imgsz: int, conf: float = 0.25) -> dict:
    """Run ``weights`` on the validation images, match detections to the label boxes by IoU, and measure the keypoint
    error to the labels by source. Returns {cam: {source: {n, pck10, pck20, median_px}}}."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    lab = pd.read_parquet(dataset / "labels.parquet")
    lab["name"] = lab["cam"] + "_" + lab["clip_frame"].map(lambda f: f"{f:05d}")
    boxes_by = {}
    for split_dir in [dataset / "labels" / "val"]:
        for txt in split_dir.glob("*.txt"):
            boxes_by[txt.stem] = txt
    rows = []
    for img in sorted((dataset / "images" / "val").glob("*.jpg")):
        name = img.stem
        cam = name.split("_")[0]
        res = model.predict(str(img), imgsz=imgsz, verbose=False, conf=conf)[0]
        if res.keypoints is None or res.boxes is None or not len(res.boxes):
            continue
        det_boxes = res.boxes.xyxy.cpu().numpy()
        kp = res.keypoints.xy.cpu().numpy()
        H, W = res.orig_shape
        # the label instances: parse the YOLO lines back (box normalised) and pair each with its pid via labels.parquet
        L = lab[lab["name"] == name]
        for pid, g in L.groupby("pid"):
            gv = g[g["vis"] > 0]
            if gv.empty:
                continue
            # the label box for this pid comes from the same tracks the 09g wrote; recover it from the keypoints' hull
            # plus the parquet's box is not stored, so match by the labelled keypoints: the detection whose keypoints
            # are nearest on the det/fit_self joints
            best, best_d = None, 1e9
            for d in range(len(det_boxes)):
                x1, y1, x2, y2 = det_boxes[d]
                inside = ((gv["u"] >= x1 - 20) & (gv["u"] <= x2 + 20) & (gv["v"] >= y1 - 20) & (gv["v"] <= y2 + 20)).mean()
                if inside > 0.6:
                    dd = np.median(np.hypot(kp[d][gv["joint"].to_numpy()][:, 0] - gv["u"].to_numpy(),
                                            kp[d][gv["joint"].to_numpy()][:, 1] - gv["v"].to_numpy()))
                    if dd < best_d:
                        best, best_d = d, dd
            if best is None:
                continue
            for r in gv.itertuples():
                e = float(np.hypot(kp[best][r.joint, 0] - r.u, kp[best][r.joint, 1] - r.v))
                rows.append((cam, r.source, e))
    t = pd.DataFrame(rows, columns=["cam", "source", "err"])
    out = {}
    for (cam, src), g in t.groupby(["cam", "source"]):
        out.setdefault(cam, {})[src] = {"n": int(len(g)), "pck10": float((g["err"] <= 10).mean()),
                                        "pck20": float((g["err"] <= 20).mean()), "median_px": float(g["err"].median())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", default="yolov8x-pose.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=10, help="backbone layers frozen")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--score-only", action="store_true", help="score --weights on the val frames, no training")
    a = ap.parse_args()
    dataset, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.score_only:
        s = score(Path(a.weights), dataset, imgsz=a.imgsz)
        print(json.dumps(s, indent=1))
        (out / f"score_{Path(a.weights).stem}.json").write_text(json.dumps(s, indent=1))
        return
    from ultralytics import YOLO

    base = score(Path(a.weights), dataset, imgsz=a.imgsz)
    (out / "score_baseline.json").write_text(json.dumps(base, indent=1))
    print("baseline:", json.dumps(base))
    model = YOLO(str(a.weights))
    model.train(data=str(dataset / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, freeze=a.freeze,
                lr0=a.lr0, project=str(out), name="train", exist_ok=True, plots=False, verbose=False,
                fliplr=0.5, mosaic=0.0, degrees=0.0, scale=0.2, translate=0.05, hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,
                close_mosaic=0, workers=2, patience=10)
    best = out / "train" / "weights" / "best.pt"
    ft = score(best, dataset, imgsz=a.imgsz)
    (out / "score_finetuned.json").write_text(json.dumps(ft, indent=1))
    print("fine-tuned:", json.dumps(ft))
    for cam in ft:
        for src in ft[cam]:
            b = base.get(cam, {}).get(src, {})
            print(f"  {cam:8s} {src:9s} PCK@10 {100 * b.get('pck10', float('nan')):5.1f} -> {100 * ft[cam][src]['pck10']:5.1f} %   "
                  f"median {b.get('median_px', float('nan')):5.1f} -> {ft[cam][src]['median_px']:5.1f} px  (n {ft[cam][src]['n']})")


if __name__ == "__main__":
    main()
