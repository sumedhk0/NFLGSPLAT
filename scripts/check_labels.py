"""Validate hand-labeled field landmarks WITHOUT training a model.

For each labeled frame: fit a field→image homography directly from the clicked
landmarks, report the reprojection residual, and render the predicted field grid
(cyan) over the frame. If the grid tracks the painted lines/hashes/numbers, the
labels (esp. the number anchors) are geometrically sound and well-conditioned —
i.e. the whole approach works, before any GPU training.

    python scripts/check_labels.py <labels.json> <frames_dir> [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from nfl_gsplat.calibration.field_homography import fit_plane_homography
from nfl_gsplat.calibration.field_landmarks import (
    HALF_WIDTH_M, HASH_OFFSET_M, NFL_LANDMARKS, YARD_LINE_SPACING_M,
)


def _overlay(img, H):
    """Draw the predicted field (yard lines sideline-to-sideline + hash rows)."""
    out = img.copy()

    def to_img(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), H).reshape(2)
        return (int(round(p[0])), int(round(p[1])))

    for k in range(-10, 11):                      # yard lines every 5 yd
        X = k * YARD_LINE_SPACING_M
        cv2.line(out, to_img(X, +HALF_WIDTH_M), to_img(X, -HALF_WIDTH_M),
                 (255, 200, 0), 1, cv2.LINE_AA)
    xspan = 10 * YARD_LINE_SPACING_M
    for Y in (+HASH_OFFSET_M, -HASH_OFFSET_M):    # two hash rows
        cv2.line(out, to_img(-xspan, Y), to_img(+xspan, Y), (255, 120, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Validate hand-labeled landmarks via homography + overlay")
    ap.add_argument("labels", help="labels.json from label_landmarks.py")
    ap.add_argument("frames_dir", help="directory with the labeled frame PNGs")
    ap.add_argument("--out-dir", default=None, help="where to write overlays (default: <frames_dir>/../check)")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir) if args.out_dir else frames_dir.parent / "check"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.labels).read_text())

    residuals = []
    for rec in data["frames"]:
        pts = rec["points"]
        names = [p["name"] for p in pts]
        world = np.array([NFL_LANDMARKS[n][:2] for n in names], dtype=np.float64)
        image = np.array([p["uv"] for p in pts], dtype=np.float64)
        n_num = sum("number" in n for n in names)
        if len(pts) < 4:
            print(f"{rec['file']}: only {len(pts)} pts (need >=4) — skipped")
            continue
        H = fit_plane_homography(world, image)
        if H is None:
            print(f"{rec['file']}: homography failed")
            continue
        proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), H).reshape(-1, 2)
        res = np.linalg.norm(proj - image, axis=1)
        residuals.append(float(np.median(res)))
        print(f"{rec['file']}: {len(pts)} pts ({n_num} number) | "
              f"median {np.median(res):.2f}px  max {res.max():.1f}px")
        img = cv2.imread(str(frames_dir / rec["file"]))
        if img is not None:
            vis = _overlay(img, H)
            for (u, v) in image:
                cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 255), 2)
            cv2.imwrite(str(out_dir / f"check_{rec['file']}"), vis)

    if residuals:
        print(f"\n{len(residuals)} frames | overall median residual {np.median(residuals):.2f}px")
        print(f"overlays (cyan = predicted field) → {out_dir}")
        print("If the cyan grid tracks the painted lines/hashes/numbers, the labels "
              "are sound and well-conditioned — the approach works.")


if __name__ == "__main__":
    main()
