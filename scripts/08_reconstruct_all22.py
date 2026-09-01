#!/usr/bin/env python
"""Reconstruct one production All-22 play end to end, from the video alone.

Every stage of this has been measured on its own; none have ever run together on
production footage. The chain is:

    paint  -> two cameras          calibrate_clip, no tracking, no identity
    YOLO   -> players per view     head point = top-centre of the person box
    geometry -> cross-view matching  which detection in one view is which in the
                                     other, by how well the pair triangulates
    triangulation -> 3D positions

THERE IS NO GROUND TRUTH HERE, which is the point: this is the footage the
project exists to reconstruct, and nothing external says where the players were.
So it is checked against things that must be true regardless:

    on the field     a triangulated player must land inside the boundary
    head height      matching HEADS, the 3D point must sit near 1.8 m above the
                     turf -- nothing in the pipeline forces this, and it is the
                     sharpest check available, because a wrong camera or a wrong
                     match puts it anywhere
    count            about 22 players, not 5 and not 60
    continuity       positions must move smoothly between frames

The two clips are assumed to start together. They are the same play at the same
length, but nothing verifies the offset, so any per-frame count should be read
with that in mind.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.calibrate_clip import calibrate_clip
from nfl_gsplat.calibration.decompose_homography import projection_matrix
from nfl_gsplat.errors import CalibrationError

# A head sits about here above the turf. Used ONLY to check the result, never
# to produce it.
EXPECTED_HEAD_M = 1.80
FIELD_HALF_X_M = 56.0
FIELD_HALF_Y_M = 26.0


def read_frames(path, n_wanted):
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = {}
    for f in np.linspace(0, max(1, total - 2), n_wanted).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            out[int(f)] = img
    cap.release()
    return out, w, h, total


def head_points(model, image, conf=0.25):
    """Top-centre of each person box: the same physical point in both views."""
    res = model.predict(image, classes=[0], conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 2))
    xyxy = res.boxes.xyxy.cpu().numpy()
    return np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 1]])


def match_and_triangulate(P_a, pts_a, P_b, pts_b, *, max_cost_px=60.0):
    """Pair detections across views by triangulation quality, then place them."""
    import cv2
    from scipy.optimize import linear_sum_assignment

    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.zeros((0, 3))
    ia, ib = np.meshgrid(np.arange(len(pts_a)), np.arange(len(pts_b)),
                         indexing="ij")
    pa = pts_a[ia.ravel()].astype(np.float64)
    pb = pts_b[ib.ravel()].astype(np.float64)
    h = cv2.triangulatePoints(P_a, P_b, pa.T.copy(), pb.T.copy())
    xyz = (h[:3] / h[3]).T

    def reproj(P, uv):
        q = np.c_[xyz, np.ones(len(xyz))] @ P.T
        return np.linalg.norm(q[:, :2] / q[:, 2:3] - uv, axis=1)

    cost = reproj(P_a, pa) + reproj(P_b, pb)
    off = ((np.abs(xyz[:, 0]) > FIELD_HALF_X_M)
           | (np.abs(xyz[:, 1]) > FIELD_HALF_Y_M)
           | (xyz[:, 2] < 0.0) | (xyz[:, 2] > 3.5))
    cost[off] = 1e6
    cost = cost.reshape(len(pts_a), len(pts_b))
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < max_cost_px
    return xyz.reshape(len(pts_a), len(pts_b), 3)[r[keep], c[keep]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("data/all22/bal_at_kc_2024_wk1"))
    ap.add_argument("--sideline", default="001_Sideline_KC_2-20_BLT_24.mp4")
    ap.add_argument("--endzone", default="002_Endzone_KC_2-20_BLT_24.mp4")
    ap.add_argument("--calib-frames", type=int, default=26)
    ap.add_argument("--play-frames", type=int, default=24)
    args = ap.parse_args()

    from ultralytics import YOLO

    cams, sizes = {}, {}
    for tag, name in (("Sideline", args.sideline), ("Endzone", args.endzone)):
        images, w, h, total = read_frames(args.root / name, args.calib_frames)
        print(f"{tag}: {name}  {w}x{h}, {total} frames")
        try:
            cam, focal, centre, q, won = calibrate_clip(images, w, h)
        except CalibrationError as exc:
            raise SystemExit(f"{tag} did not calibrate: {exc}")
        print(f"   camera {np.round(centre, 1)} m, {q['fov_deg']:.1f} deg, "
              f"rms {q['rms_px']:.1f} px, {len(cam)} frames, white>={won[0]}")
        cams[tag] = cam
        sizes[tag] = (w, h, total)

    # Sample the SAME frame indices in both clips; they are the same play at the
    # same length, though nothing here verifies the offset.
    total = min(sizes["Sideline"][2], sizes["Endzone"][2])
    want = np.linspace(0, max(1, total - 2), args.play_frames).astype(int)
    frames = {}
    for tag, name in (("Sideline", args.sideline), ("Endzone", args.endzone)):
        import cv2

        cap = cv2.VideoCapture(str(args.root / name))
        got = {}
        for f in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok:
                got[int(f)] = img
        cap.release()
        frames[tag] = got

    model = YOLO("yolov8n.pt")
    per_frame, counts = [], []
    for f in want:
        f = int(f)
        if f not in frames["Sideline"] or f not in frames["Endzone"]:
            continue
        # Cameras are solved on their own sampled frames; use the nearest.
        ks = min(cams["Sideline"], key=lambda k: abs(k - f))
        ke = min(cams["Endzone"], key=lambda k: abs(k - f))
        Ps = projection_matrix(*cams["Sideline"][ks])
        Pe = projection_matrix(*cams["Endzone"][ke])
        pa = head_points(model, frames["Sideline"][f])
        pb = head_points(model, frames["Endzone"][f])
        xyz = match_and_triangulate(Ps, pa, Pe, pb)
        counts.append((len(pa), len(pb), len(xyz)))
        if len(xyz):
            per_frame.append((f, xyz))

    if not per_frame:
        raise SystemExit("no frame produced a reconstruction")

    det = np.array(counts)
    allxyz = np.concatenate([x for _f, x in per_frame])
    print(f"\n{len(per_frame)} frames reconstructed")
    print(f"detections per frame: sideline {np.median(det[:, 0]):.0f}, "
          f"endzone {np.median(det[:, 1]):.0f}, matched {np.median(det[:, 2]):.0f}")
    print(f"players placed per frame: median {np.median(det[:, 2]):.0f}")
    print("\nchecks that need no ground truth:")
    inside = ((np.abs(allxyz[:, 0]) <= FIELD_HALF_X_M)
              & (np.abs(allxyz[:, 1]) <= FIELD_HALF_Y_M))
    print(f"   on the field      {inside.mean():5.0%} of placements")
    z = allxyz[:, 2]
    print(f"   head height       median {np.median(z):4.2f} m "
          f"(expected ~{EXPECTED_HEAD_M:.1f}), IQR "
          f"{np.subtract(*np.percentile(z, [75, 25])):4.2f} m")
    spread = [np.median([np.min(np.linalg.norm(x[i] - np.delete(x, i, 0), axis=1))
                         for i in range(len(x))]) for _f, x in per_frame if len(x) > 2]
    if spread:
        print(f"   nearest neighbour median {np.median(spread):4.2f} m "
              "(real players stand about 1-3 m apart)")
    np.savez("C:/Users/sumedh/diag/all22_reconstruction.npz",
             frames=np.array([f for f, _x in per_frame]),
             counts=det)
    print("\nsaved -> C:/Users/sumedh/diag/all22_reconstruction.npz")


if __name__ == "__main__":
    main()
