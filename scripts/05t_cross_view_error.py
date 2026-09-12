#!/usr/bin/env python
"""Score every cached body in the camera it was NOT fitted to. The only ruler that sees depth.

    C:/venvs/smplx312/Scripts/python scripts/05t_cross_view_error.py --play-dir P [--csv out.csv]

WHY. A body fitted to one camera's keypoints can match that camera to three pixels while standing
metres from where it belongs and facing the wrong way, because depth along the camera ray and
rotation about it are exactly what a single view cannot see. Measured on play 1's v34 cache
(2026-09-12), scoring each body against the keypoints of the camera it was fitted to:

    sideline: 5099 body-frames, rms p50 3.4 px, p90 11.0, only 16 frames (0.3 %) over 20 px

which says the fit is nearly perfect and is worthless as evidence. The same bodies in the OTHER
camera, on the 1812 frames it can see them:

    endzone: p50 8.7 px, p90 25.8, p99 341.5; 197 frames (10.9 %) over 20 px, 129 over 100

One player carried most of the tail: id 19 at 4.0 px in the sideline and 142.2 px in the endzone,
his hips 149 px sideways from his own endzone box -- sideways in the endzone being depth along the
sideline ray. His two cameras' ankle keypoints agreed to 0.36 m the whole time, so the information
to place him existed and was not used: ankle_anchor only trusts the triangulated ankle midpoint
when the two views agree to 0.3 m, and a player just the wrong side of that cliff is placed on a
point that slides along one camera's ray.

So this is the standing ruler for placement, as the census is the ruler for identity and the
footage overlay for everything: per camera, the reprojection of the SAME cached bodies, and the
tail per player. A correction to placement has to move the camera that was not fitted.

Diagnostic only: writes nothing but the optional CSV. numpy 1 (smplx312), it reads the pose caches.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.pose.coco import coco_to_body
from nfl_gsplat.pose.fit_mono2d import project
from nfl_gsplat.pose.forward_kinematics import fk_forward, load_smplx_skeleton
from nfl_gsplat.pose.fuse_smplx import _pack_params
from nfl_gsplat.pose.keypoint_filter import reject_outliers
from nfl_gsplat.render.play_timeline import clip_offset

MIN_JOINTS = 6


def boxes_by_frame(kdf, cam: str) -> dict:
    """``{(frame, id): the 17 COCO rows}``, the highest-confidence box where a camera has several."""
    out = {}
    for (f, p), g in kdf[kdf["cam"] == cam].groupby(["frame", "global_player_id"]):
        if len(g) == 17:
            out[(int(f), int(p))] = g.sort_values("joint")
        elif len(g) and len(g) % 17 == 0:
            chunks = [g.iloc[i:i + 17] for i in range(0, len(g), 17)]
            chunks = [c for c in chunks if sorted(c["joint"].tolist()) == list(range(17))]
            if chunks:
                out[(int(f), int(p))] = max(chunks, key=lambda c: float(c["conf"].sum())).sort_values("joint")
    return out


def rms_in(track, rows, J, *, frame: int, min_conf: float):
    """Reprojection rms (px) of a placed body's joints against one camera's keypoints, or None when
    that camera cannot see the frame or has too few confident joints."""
    if rows is None or frame < 0 or frame >= len(track.conf) or track.conf[frame] <= 0:
        return None
    uv, conf = coco_to_body(rows[["x", "y"]].to_numpy(float), rows["conf"].to_numpy(float), min_conf=min_conf)
    use = np.asarray(conf, float) >= min_conf
    if use.sum() < MIN_JOINTS:
        return None
    intr, pose = track.at(frame)
    pix, _depth = project(intr.K(), np.asarray(pose.R, float), np.asarray(pose.t, float), J[use])
    err = np.linalg.norm(pix - np.asarray(uv, float)[use], axis=1)
    return float(np.sqrt(np.mean(err * err)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--refit", type=Path, default=None, help="default <play-dir>/poses_refit.json")
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--cam", default="sideline", help="the camera the bodies were fitted to")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-frames", type=int, default=20, help="players with fewer paired frames are not listed")
    ap.add_argument("--csv", type=Path, default=None, help="write one row per body-frame")
    ap.add_argument("--no-keypoint-filter", action="store_true")
    args = ap.parse_args()
    P = args.play_dir

    tracks = load_camera_track(P / "cameras.npz")
    other = "endzone" if args.cam == "sideline" else "sideline"
    if other not in tracks:
        raise SystemExit(f"{P} has no {other} camera: there is no second view to score against")
    offset = clip_offset(P)
    kall = pd.read_parquet(P / "keypoints_2d.parquet")
    if not args.no_keypoint_filter:
        kall, n_rej = reject_outliers(kall)
        print(f"{n_rej} keypoints thrown out by the temporal filter, as the fit throws them out")
    ka, kb = boxes_by_frame(kall, args.cam), boxes_by_frame(kall, other)
    blob = pickle.load(open(args.refit or P / "poses_refit.json", "rb"))

    per: dict = {}
    for f, recs in blob["frames"].items():
        for pid, rec in recs.items():
            per.setdefault(int(pid), {})[int(f)] = rec

    rows = []
    for pid, recs in sorted(per.items()):
        if args.ids and pid not in args.ids:
            continue
        betas = np.asarray(next(iter(recs.values()))["betas"], float)
        rest, _ = load_smplx_skeleton(args.body_models, betas=betas)
        forward = fk_forward(rest)
        for f, rec in sorted(recs.items()):
            J = forward(_pack_params(np.asarray(rec["body_pose"], float).reshape(-1),
                                     np.asarray(rec["global_orient"], float).reshape(3),
                                     np.asarray(rec["transl"], float).reshape(3)))
            own = rms_in(tracks[args.cam], ka.get((f, pid)), J, frame=f, min_conf=args.min_conf)
            cross = rms_in(tracks[other], kb.get((f + offset, pid)), J, frame=f + offset, min_conf=args.min_conf)
            if own is None and cross is None:
                continue
            rows.append((pid, f, np.nan if own is None else own, np.nan if cross is None else cross))
    if not rows:
        raise SystemExit("no body-frame could be scored in either camera")

    a = np.array([[r[2], r[3]] for r in rows], float)
    both = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
    print(f"{len(a)} body-frames scored, {int(both.sum())} of them visible to both cameras "
          f"({other} frame = {args.cam} frame {offset:+d})")
    print(f"  {args.cam:8s} (fitted to it):        p50 {np.nanmedian(a[:, 0]):6.1f} px, "
          f"p90 {np.nanpercentile(a[:, 0], 90):6.1f}")
    print(f"  {other:8s} (sees what it cannot):  p50 {np.nanmedian(a[:, 1]):6.1f} px, "
          f"p90 {np.nanpercentile(a[:, 1], 90):6.1f}, p99 {np.nanpercentile(a[:, 1], 99):6.1f}")
    for th in (10, 20, 30, 50, 100):
        print(f"    over {th:3d} px: {other} {int((a[both, 1] > th).sum()):5d} "
              f"({100.0 * (a[both, 1] > th).mean():4.1f}%), {args.cam} {int((a[both, 0] > th).sum()):5d} "
              f"({100.0 * (a[both, 0] > th).mean():4.1f}%)")
    print(f"per player on the paired frames, worst {other} first:")
    print(f"  id   frames   {args.cam} p50/p90    {other} p50/p90")
    table = []
    for pid in sorted({r[0] for r in rows}):
        v = np.array([[r[2], r[3]] for r in rows if r[0] == pid], float)
        m = np.isfinite(v[:, 0]) & np.isfinite(v[:, 1])
        if m.sum() >= args.min_frames:
            table.append((pid, int(m.sum()), np.median(v[m, 0]), np.percentile(v[m, 0], 90),
                          np.median(v[m, 1]), np.percentile(v[m, 1], 90)))
    for pid, n, o5, o9, c5, c9 in sorted(table, key=lambda r: -r[4]):
        print(f"  {pid:3d} {n:7d}   {o5:7.1f}/{o9:7.1f}   {c5:7.1f}/{c9:7.1f}")
    if args.csv:
        pd.DataFrame(rows, columns=["global_player_id", "frame", f"rms_{args.cam}", f"rms_{other}"]).to_csv(
            args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
