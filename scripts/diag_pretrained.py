"""Acceptance diagnostic for the pretrained-hybrid path (run locally).

    python scripts/diag_pretrained.py <frames_dir> <roboflow_kps.json> ^
        --territory away --out-dir C:\\Users\\<you>\\diag\\pretrained_overlays

Frames_dir holds frames named f%05d.png (as written by eval/precompute
sampling); indices are parsed from filenames to look up cached keypoints.
Overlays go OUTSIDE the repo. Grid on painted lines = ship it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from nfl_gsplat.calibration.field_detect import detect_field_features
from nfl_gsplat.calibration.field_landmarks import (
    HALF_WIDTH_M, HASH_OFFSET_M, NFL_LANDMARKS, YARD_LINE_SPACING_M,
)
from nfl_gsplat.calibration.fuse_pretrained import fuse_frame
from nfl_gsplat.calibration.roboflow_kps import load_kps_json


def _grid(img, Hm):
    out = img.copy()

    def to_img(X, Y):
        p = cv2.perspectiveTransform(np.array([[[X, Y]]], np.float64), Hm).reshape(2)
        return (int(round(p[0])), int(round(p[1])))

    for k in range(-10, 11):
        X = k * YARD_LINE_SPACING_M
        cv2.line(out, to_img(X, +HALF_WIDTH_M), to_img(X, -HALF_WIDTH_M),
                 (255, 200, 0), 1, cv2.LINE_AA)
    for Y in (+HASH_OFFSET_M, -HASH_OFFSET_M):
        cv2.line(out, to_img(-10 * YARD_LINE_SPACING_M, Y),
                 to_img(10 * YARD_LINE_SPACING_M, Y), (255, 120, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir")
    ap.add_argument("kps_json")
    ap.add_argument("--territory", default="away", choices=["home", "away"])
    ap.add_argument("--out-dir", required=True,
                    help="overlay output dir OUTSIDE the repo")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kps_by_frame = load_kps_json(args.kps_json)

    for fp in sorted(Path(args.frames_dir).glob("f*.png")):
        m = re.match(r"f(\d+)", fp.stem)
        if not m:
            continue
        fidx = int(m.group(1))
        kps = kps_by_frame.get(fidx, [])
        img = cv2.imread(str(fp))
        if img is None or not kps:
            print(f"{fp.name}: skipped (no image or no cached kps)")
            continue
        Hh, Ww = img.shape[:2]
        feats = detect_field_features(img)
        corrs = fuse_frame(feats.yard_lines, feats.hashes, kps,
                           territory=args.territory, image_size=(Ww, Hh))
        if len(corrs) < 4:
            print(f"{fp.name}: only {len(corrs)} correspondences — gap")
            continue
        world = np.array([NFL_LANDMARKS[n][:2] for (n, _uv) in corrs], np.float64)
        uv = np.array([p for (_n, p) in corrs], np.float64)
        Hm, mask = cv2.findHomography(world, uv, cv2.RANSAC, 5.0)
        if Hm is None:
            print(f"{fp.name}: homography failed")
            continue
        inl = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), Hm).reshape(-1, 2)
        res = np.linalg.norm(proj - uv, axis=1)
        print(f"{fp.name}: {len(corrs)} corrs, inliers {int(inl.sum())}/{len(corrs)}, "
              f"median {np.median(res[inl]):.2f}px")
        over = _grid(img, Hm)
        for (gu, gv) in uv:
            cv2.circle(over, (int(gu), int(gv)), 6, (0, 200, 0), 2)
        cv2.imwrite(str(out_dir / f"fused_{fp.name}"), over)
    print(f"\noverlays -> {out_dir}")


if __name__ == "__main__":
    main()
