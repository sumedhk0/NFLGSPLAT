#!/usr/bin/env python
"""Is a badly reprojecting body in the wrong POSE, the wrong PLACE, or facing the wrong WAY?

    C:/venvs/smplx312/Scripts/python scripts/05s_probe_translation.py --play-dir P --ids 9 --frames 250 292

WHY. Play 1's motion man reprojects 19-30 px off his own keypoints through frames 258-270 at EVERY
joint, shoulders at 0.99 confidence included. Three things could do that and they are told apart by
what you are allowed to move:

  re-solve transl alone          the pose and heading are right, the PLACEMENT is wrong
  re-solve global_orient + transl the body pose is right, the HEADING is wrong
  turn the body 180 deg, re-solve the front/back ambiguity a single camera cannot see
  none of them helps             the pose itself is wrong

Measured on play 1 (2026-09-12): translation alone recovered a third of it (rms 29.5 -> 26.2 px at
frame 266), and the cache's heading spun +80 deg across 258-270 (93 -> -172 deg) while the same
man's two-view records at 271-281 sat steady at 71 deg and his ground point walked straight in +y
the whole time. So the last column matters most: a running man faces roughly where he is going, and
this prints how far each frame's heading is from the direction his own ground point is moving.

Diagnostic only: writes nothing. numpy 1 (smplx312), because it reads the pose caches.
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
from nfl_gsplat.render.play_timeline import ankle_ground, clip_offset, ground_positions

FPS = 59.94


def best_box_rows(g):
    """One box's 17 COCO rows for a (frame, id), the highest-confidence box when there are several
    (the same rule the fit uses -- scripts/05p_refit_mono.best_box_rows)."""
    if len(g) == 17:
        return g.sort_values("joint")
    if len(g) % 17 or len(g) == 0:
        return None
    chunks = [g.iloc[i:i + 17] for i in range(0, len(g), 17)]
    chunks = [c for c in chunks if sorted(c["joint"].tolist()) == list(range(17))]
    if not chunks:
        return None
    return max(chunks, key=lambda c: float(c["conf"].sum())).sort_values("joint")


def heading_deg(go) -> float:
    """Where the body faces on the ground plane, degrees (SMPL-X faces +z in model axes)."""
    from scipy.spatial.transform import Rotation

    v = Rotation.from_rotvec(np.asarray(go, float).reshape(3)).apply([0.0, 0.0, 1.0])[:2]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def wrap_deg(d: float) -> float:
    """An angle difference folded into (-180, 180]."""
    return float((d + 540.0) % 360.0 - 180.0)


def turned(go, deg: float):
    """``global_orient`` turned ``deg`` about the world vertical -- the front/back ambiguity a single
    camera cannot resolve (a symmetric body projects nearly the same either way)."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_rotvec([0.0, 0.0, np.radians(deg)]) * Rotation.from_rotvec(np.asarray(go, float).reshape(3))
    return r.as_rotvec()


def reproj_px(forward, params, cam, uv, use):
    """Median and rms pixel error of the body joints against the keypoints, for the used joints."""
    J = forward(params)
    pix, _depth = project(*cam, J[use])
    err = np.linalg.norm(pix - np.asarray(uv, float)[use], axis=1)
    return float(np.median(err)), float(np.sqrt(np.mean(err * err)))


def resolve(forward, params, cam, uv, conf, use, *, free_orient: bool, free_z: bool):
    """``(params, shift_m)``: the same body_pose placed (and optionally turned) where it best fits
    the keypoints. Only translation -- and global_orient with ``free_orient`` -- may move."""
    from scipy.optimize import least_squares

    p0 = np.asarray(params, float).copy()
    n_t = 3 if free_z else 2
    go0 = p0[63:66].copy()
    t0 = p0[-3:].copy()
    w = np.asarray(conf, float)[use][:, None]
    tgt = np.asarray(uv, float)[use]

    def unpack(d):
        p = p0.copy()
        p[-3:-3 + n_t] = t0[:n_t] + d[:n_t]
        if free_orient:
            p[63:66] = go0 + d[n_t:]
        return p

    def resid(d):
        J = forward(unpack(d))
        pix, _ = project(*cam, J[use])
        return (w * (pix - tgt)).reshape(-1)

    sol = least_squares(resid, np.zeros(n_t + (3 if free_orient else 0)), method="trf", max_nfev=80)
    return unpack(sol.x), float(np.linalg.norm(sol.x[:2]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--refit", type=Path, default=None)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs=2, default=None, help="print per frame within this range")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--free-z", action="store_true", help="let the re-solves change the height too")
    ap.add_argument("--min-speed", type=float, default=1.5,
                    help="a body moving slower than this (m/s) does not have to face anywhere in particular, "
                         "so it is left out of the heading-against-travel summary")
    ap.add_argument("--no-keypoint-filter", action="store_true")
    args = ap.parse_args()
    P = args.play_dir

    tracks = load_camera_track(P / "cameras.npz")
    tr = tracks[args.cam]
    kall = pd.read_parquet(P / "keypoints_2d.parquet")
    if not args.no_keypoint_filter:
        kall, n_rej = reject_outliers(kall)
        print(f"{n_rej} keypoints thrown out by the temporal filter, as the fit throws them out")
    blob = pickle.load(open(args.refit or P / "poses_refit.json", "rb"))
    offset = clip_offset(P)
    other = "endzone" if args.cam == "sideline" else "sideline"

    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    ankles = ankle_ground(kall, tracks)
    gnd_own = ground_positions(df[df["cam"] == args.cam], tracks, ankles=ankles)

    ks = {(int(f), int(p)): best_box_rows(g)
          for (f, p), g in kall[kall["cam"] == args.cam].groupby(["frame", "global_player_id"])}
    ko = {(int(f), int(p)) for (f, p), _g in kall[kall["cam"] == other].groupby(["frame", "global_player_id"])}

    recs: dict = {}
    for f, r in blob["frames"].items():
        for pid, rec in r.items():
            recs.setdefault(int(pid), {})[int(f)] = rec

    for pid in args.ids:
        per = recs.get(pid)
        if not per:
            print(f"id {pid}: no records in the cache")
            continue
        betas = np.asarray(next(iter(per.values()))["betas"], float)
        rest, _ = load_smplx_skeleton(args.body_models, betas=betas)
        forward = fk_forward(rest)
        # where this body is going, from its own ground track (central difference over the records)
        fs = sorted(per)
        g_of = {f: gnd_own.get(f, {}).get(pid) for f in fs}
        travel: dict = {}
        for i, f in enumerate(fs):
            a = g_of[fs[i - 1]] if i else None
            b = g_of[fs[i + 1]] if i + 1 < len(fs) else None
            if a is None or b is None:
                continue
            dt = (fs[i + 1] - fs[i - 1]) / FPS
            d = np.asarray(b, float) - np.asarray(a, float)
            sp = float(np.linalg.norm(d) / dt)
            travel[f] = (float(np.degrees(np.arctan2(d[1], d[0]))), sp)

        rows = []
        lo, hi = args.frames if args.frames else (10 ** 9, -10 ** 9)
        print(f"\nid {pid}: {len(per)} records, betas[0:3] {np.round(betas[:3], 2)}")
        if args.frames:
            print("frame  n   rms px: cache  +transl  +orient   flip180   heading  travel  speed  |head-travel|  2nd cam")
        for f in fs:
            rec = per[f]
            g = ks.get((f, pid))
            if g is None or f >= len(tr.conf) or tr.conf[f] <= 0:
                continue
            uv, conf = coco_to_body(g[["x", "y"]].to_numpy(float), g["conf"].to_numpy(float), min_conf=args.min_conf)
            use = np.asarray(conf, float) >= args.min_conf
            if use.sum() < 6:
                continue
            intr, pose = tr.at(f)
            cam = (intr.K(), np.asarray(pose.R, float), np.asarray(pose.t, float))
            go = np.asarray(rec["global_orient"], float).reshape(3)
            params = _pack_params(np.asarray(rec["body_pose"], float).reshape(-1), go,
                                  np.asarray(rec["transl"], float).reshape(3))
            _b50, brms = reproj_px(forward, params, cam, uv, use)
            p_t, sh_t = resolve(forward, params, cam, uv, conf, use, free_orient=False, free_z=args.free_z)
            _t50, trms = reproj_px(forward, p_t, cam, uv, use)
            p_o, _sh_o = resolve(forward, params, cam, uv, conf, use, free_orient=True, free_z=args.free_z)
            _o50, orms = reproj_px(forward, p_o, cam, uv, use)
            p_f = params.copy()
            p_f[63:66] = turned(go, 180.0)
            p_f, _sh_f = resolve(forward, p_f, cam, uv, conf, use, free_orient=False, free_z=args.free_z)
            _f50, frms = reproj_px(forward, p_f, cam, uv, use)
            h = heading_deg(go)
            tv, sp = travel.get(f, (np.nan, np.nan))
            dh = abs(wrap_deg(h - tv)) if np.isfinite(tv) else np.nan
            rows.append((f, brms, trms, orms, frms, dh, sp, sh_t))
            if lo <= f <= hi:
                print(f"{f:5d} {int(use.sum()):3d}   {brms:11.1f} {trms:8.1f} {orms:8.1f} {frms:9.1f}   "
                      f"{h:7.1f} {tv:7.1f} {sp:6.2f} {dh:13.1f}   "
                      f"{'yes' if (f + offset, pid) in ko else ' - '}")
        if not rows:
            continue
        a = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows], float)
        fr = np.array([r[0] for r in rows])
        print(f"  WHOLE TRACK ({len(rows)} frames): rms p50 cache {np.median(a[:, 0]):5.1f} -> transl "
              f"{np.median(a[:, 1]):5.1f} -> orient+transl {np.median(a[:, 2]):5.1f} px "
              f"(p90 {np.percentile(a[:, 0], 90):5.1f} -> {np.percentile(a[:, 1], 90):5.1f} -> "
              f"{np.percentile(a[:, 2], 90):5.1f}); flip180 p50 {np.median(a[:, 3]):5.1f}; "
              f"transl shift p50 {np.median(a[:, 6]):.2f} m")
        m = np.isfinite(a[:, 4]) & (a[:, 5] >= args.min_speed)
        if m.any():
            print(f"  HEADING vs TRAVEL on {int(m.sum())} frames over {args.min_speed} m/s: |diff| p50 "
                  f"{np.median(a[m, 4]):5.1f} deg, p90 {np.percentile(a[m, 4], 90):5.1f}, "
                  f"over 90 deg (facing away from where he runs) {100 * np.mean(a[m, 4] > 90):.0f}%; "
                  f"flip180 beats the cache on {100 * np.mean(a[m, 3] < a[m, 0]):.0f}% of them")
        if args.frames:
            w = (fr >= lo) & (fr <= hi)
            if w.any():
                print(f"  WINDOW {lo}-{hi} ({int(w.sum())} frames): rms p50 cache {np.median(a[w, 0]):5.1f} -> "
                      f"transl {np.median(a[w, 1]):5.1f} -> orient+transl {np.median(a[w, 2]):5.1f} px; "
                      f"flip180 {np.median(a[w, 3]):5.1f}; |head-travel| p50 "
                      f"{np.nanmedian(a[w, 4]):5.1f} deg")


if __name__ == "__main__":
    main()
