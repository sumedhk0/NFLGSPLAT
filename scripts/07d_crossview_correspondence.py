#!/usr/bin/env python
"""Can geometry alone match the same player across the two cameras?

This re-opens a question the project already answered NO to. Cross-view
matching was measured three ways here and failed every time -- 4.5 m, 3.31 m,
and a scatter the size of the spacing between players -- and that failure is
why the whole identity effort turned to jersey numbers instead.

But every one of those measurements used cameras this project now knows were
wrong. With cameras fitted from tracking the same rig triangulates to 0.15 m,
which is far tighter than the ~1.5 m that separates players. So the earlier
conclusion may have been a verdict on the cameras rather than on geometry, and
that is worth knowing before more effort goes into working around it.

THE TEST. Take the labelled boxes in both views, THROW THE IDENTITIES AWAY,
and try to pair them up by geometry alone: every sideline box against every
endzone box, scored by how well the pair triangulates, then a global assignment.
Score the recovered pairing against the identities that were discarded.

Reported against two baselines so the number means something:
    chance      pairing at random, 1/N
    epipolar    the pairing implied by geometry, which is what is being tested

Run with --detections to repeat it on the dataset's own detector output rather
than ground-truth boxes, which adds real misses and false positives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.decompose_homography import projection_matrix
from nfl_gsplat.calibration.from_helmets import cameras_fixed_centre
from nfl_gsplat.data.align_video import VIDEO_FPS, helmet_boxes_by_frame
from nfl_gsplat.data.helmet_dataset import load_labels, load_tracking
from nfl_gsplat.errors import CalibrationError

WIDTH, HEIGHT = 1280, 720

# A pair is only allowed if it triangulates to somewhere a helmet could be:
# on the field, and near the helmet plane the cameras were fitted to.
MAX_HEIGHT_M = 1.2
FIELD_HALF_X_M = 60.0
FIELD_HALF_Y_M = 30.0


def pair_cost(P_a, uv_a, P_b, uv_b):
    """Reprojection cost of every sideline box against every endzone box.

    Triangulating a MISMATCHED pair still returns a point -- two rays that do
    not meet still have a closest approach -- so the cost has to be how badly
    that point reprojects, not whether one exists.
    """
    import cv2

    n, m = len(uv_a), len(uv_b)
    ia, ib = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    pa = uv_a[ia.ravel()].astype(np.float64)
    pb = uv_b[ib.ravel()].astype(np.float64)
    h = cv2.triangulatePoints(P_a, P_b, pa.T.copy(), pb.T.copy())
    xyz = (h[:3] / h[3]).T

    def reproj(P, uv):
        q = np.c_[xyz, np.ones(len(xyz))] @ P.T
        return np.linalg.norm(q[:, :2] / q[:, 2:3] - uv, axis=1)

    cost = reproj(P_a, pa) + reproj(P_b, pb)
    impossible = (
        (np.abs(xyz[:, 2]) > MAX_HEIGHT_M)
        | (np.abs(xyz[:, 0]) > FIELD_HALF_X_M)
        | (np.abs(xyz[:, 1]) > FIELD_HALF_Y_M))
    cost[impossible] = 1e6
    return cost.reshape(n, m)


def match_frame(P_a, uv_a, cols_a, P_b, uv_b, cols_b):
    """``(n_correct, n_pairs)`` from a global assignment on the geometric cost."""
    from scipy.optimize import linear_sum_assignment

    if len(uv_a) == 0 or len(uv_b) == 0:
        return 0, 0
    cost = pair_cost(P_a, uv_a, P_b, uv_b)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < 1e6
    r, c = r[keep], c[keep]
    return int((cols_a[r] == cols_b[c]).sum()), int(len(r))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--alignment", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    align_path = args.alignment or (args.root / "alignment.json")
    out_path = args.out or (args.root / "correspondence.json")
    alignment = json.loads(align_path.read_text(encoding="utf-8"))
    labels = load_labels(args.root / "train_labels.csv")
    plays = load_tracking(args.root / "train_player_tracking.csv")

    usable = [r for r in alignment.values() if r["offset_s"] is not None]
    if args.limit:
        usable = usable[:args.limit]
    print(f"{len(usable)} aligned plays\n")

    out, totals = {}, [0, 0, 0.0]
    for i, rec in enumerate(usable, 1):
        game, play, offset = rec["game_key"], rec["play_id"], rec["offset_s"]
        track = plays.get((game, play))
        if track is None:
            continue
        index = {p: j for j, p in enumerate(track.players)}

        def world_at(frame, _t=track, _o=offset):
            return _t.at(_o + frame / VIDEO_FPS)

        try:
            views = {}
            for view in ("Sideline", "Endzone"):
                sub = labels[labels["video"] == f"{game}_{play:06d}_{view}.mp4"]
                byf = helmet_boxes_by_frame(sub, index)
                byf = {f: v for j, (f, v) in enumerate(sorted(byf.items()))
                       if j % args.stride == 0
                       and track.covers(offset + f / VIDEO_FPS)}
                cams, _c, mirrored = cameras_fixed_centre(
                    byf, world_at, WIDTH, HEIGHT)
                if mirrored:
                    raise CalibrationError("mirrored solve")
                views[view] = (byf, cams)
        except CalibrationError as exc:
            out[f"{game}_{play}"] = {"failed": str(exc)[:120]}
            print(f"[{i:3d}] {game}/{play:<6d} FAILED {str(exc)[:70]}", flush=True)
            continue

        (byf_s, cam_s), (byf_e, cam_e) = views["Sideline"], views["Endzone"]
        ok = n = 0
        chance = []
        for frame in sorted(set(cam_s) & set(cam_e)):
            if frame not in byf_s or frame not in byf_e:
                continue
            Ks, Rs, ts = cam_s[frame]
            Ke, Re, te = cam_e[frame]
            uv_s, cols_s = byf_s[frame]
            uv_e, cols_e = byf_e[frame]
            c, t = match_frame(projection_matrix(Ks, Rs, ts), uv_s, cols_s,
                               projection_matrix(Ke, Re, te), uv_e, cols_e)
            ok += c
            n += t
            if t:
                chance.append(1.0 / max(len(cols_e), 1))
        if not n:
            continue
        acc = ok / n
        rec_out = {"pairs": n, "correct": ok, "accuracy": round(acc, 4),
                   "chance": round(float(np.mean(chance)), 4)}
        out[f"{game}_{play}"] = rec_out
        totals[0] += ok
        totals[1] += n
        totals[2] += float(np.mean(chance)) * n
        print(f"[{i:3d}] {game}/{play:<6d} correspondence {acc:6.1%}  "
              f"({ok}/{n} pairs, chance {np.mean(chance):.1%})", flush=True)

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if totals[1]:
        print(f"\noverall: {totals[0]}/{totals[1]} = "
              f"{totals[0] / totals[1]:.1%} correct, "
              f"chance {totals[2] / totals[1]:.1%}  ->  {out_path}")


if __name__ == "__main__":
    main()
