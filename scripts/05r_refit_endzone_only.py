#!/usr/bin/env python
"""Fit bodies the ENDZONE camera alone sees to its 2-D keypoints (pose.fit_mono2d), where no pose record reaches.

WHY. The one-view refit (05p) fits the sideline's keypoints; a body only the endzone camera sees has no record of its
own there, and the timeline SLERPs across the gap from the nearest record. Play 1 (2026-09-25): Madubuike (id 4)
crawls on the turf through his 443-528 sideline hole and was drawn upright, then lunging; 262 of 4,684 live drawn
body-frames had no record of their id within 6 frames, 169 of them endzone-only. The endzone regressor's records
(05c) hold a set man's stance and are used before the snap (timeline.ENDZONE_POSES); six frames apart they contort a
moving man. This fits the endzone keypoints frame by frame with 05p's own fit, so a moving man gets a fit.

WHAT. For each id (``--ids``, default every id) and each SIDELINE frame on ``--frames`` (stride ``--stride``) where
the refit cache has no record of the id within ``--reach`` frames and the endzone keypoints have ``--min-joints``
confident joints at that frame's CLIP frame (sideline frame + clip offset): the endzone camera at the clip frame, the
endzone ground point there, the endzone regressor's nearest record of the id's endzone TRACK as the start pose (its
orientation put in world axes through that frame's endzone camera), "lying" where the endzone box is wider than
``--lying-aspect`` of its height and the sideline has no box of the id. 05p's fit job runs the sequence; the records
land in a copy of the cache at their SIDELINE frames (fused/one-view records win where they exist).

USAGE (smplx312):
  python scripts/05r_refit_endzone_only.py --play-dir P --refit P/poses_refit.json \
      --keypoints P/keypoints_2d_ft2.parquet --frames 395 607 --ids 4 --out P/poses_refit_ez.json [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.pose.coco import coco_to_body
from nfl_gsplat.pose.fit_mono2d import Mono2DConfig, merge_into_refit
from nfl_gsplat.pose.keypoint_filter import reject_outliers
from nfl_gsplat.render.play_timeline import ankle_ground, clip_offset, ground_positions

_spec = importlib.util.spec_from_file_location("refit_mono_05p", Path(__file__).with_name("05p_refit_mono.py"))
_p05 = importlib.util.module_from_spec(_spec)
sys.modules["refit_mono_05p"] = _p05                       # spawned workers unpickle _p05._fit_job by this name
_spec.loader.exec_module(_p05)


def select_frames(rec_frames: list, frames: list, reach: int) -> list:
    """The candidate sideline ``frames`` with no record (``rec_frames``, sorted) within ``reach`` frames."""
    rf = np.asarray(sorted(rec_frames), int)
    out = []
    for f in frames:
        if len(rf) and int(np.min(np.abs(rf - int(f)))) <= reach:
            continue
        out.append(int(f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--refit", type=Path, default=None, help="the pose cache to fill (default <play-dir>/poses_refit.json)")
    ap.add_argument("--keypoints", type=Path, default=None, help="default <play-dir>/keypoints_2d.parquet")
    ap.add_argument("--poses-endzone", type=Path, default=None, help="default <play-dir>/poses_endzone.json (05c)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frames", type=int, nargs=2, required=True, metavar=("LO", "HI"), help="sideline frames")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--reach", type=int, default=3, help="a record of the id this close already poses the frame")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-joints", type=int, default=8)
    ap.add_argument("--reproj-px-max", type=float, default=20.0)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--max-gap", type=int, default=12)
    ap.add_argument("--lying-aspect", type=float, default=0.7)
    ap.add_argument("--max-width-ratio", type=float, default=1.5,
                    help="skip a frame whose endzone box is wider than this times the id's median endzone box width "
                         "(a box merged with the man beside him: play 1 id 4 at 516-522 fitted #65's keypoints)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--no-roster-betas", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    t0 = time.time()
    tracks = load_camera_track(P / "cameras.npz")
    tr = tracks["endzone"]
    off = clip_offset(P) or 0                                   # clip frame = sideline frame + off
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    e = df[df["cam"] == "endzone"]
    side_keys = set(zip(df.loc[df["cam"] == "sideline", "frame"].astype(int), df.loc[df["cam"] == "sideline", "global_player_id"].astype(int)))
    kdf_all = pd.read_parquet(args.keypoints or P / "keypoints_2d.parquet")
    kdf_all, n_rej = reject_outliers(kdf_all)
    kdf = kdf_all[kdf_all["cam"] == "endzone"]
    ankles_all = ankle_ground(kdf_all, tracks)
    ground = ground_positions(e, tracks, ankles=ankles_all)     # clip frames
    refit_path = args.refit or P / "poses_refit.json"
    blob = pickle.load(open(refit_path, "rb"))
    rec_of: dict = {}
    for f, recs in blob["frames"].items():
        for pid in recs:
            rec_of.setdefault(int(pid), []).append(int(f))
    ez = pickle.load(open(args.poses_endzone or P / "poses_endzone.json", "rb"))
    tid_of = {(int(f), int(g)): int(t) for f, t, g in zip(e["frame"], e["track_id"], e["global_player_id"])}
    ez_by_tid: dict = {}
    for fc, recs in ez["frames"].items():
        for tid, r in recs.items():
            ez_by_tid.setdefault(int(tid), {})[int(fc)] = r
    box = {(int(r.frame), int(r.global_player_id)): (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2) for r in e.itertuples()}
    med_w = (e["bbox_x2"] - e["bbox_x1"]).groupby(e["global_player_id"]).median().to_dict()
    n_merged = 0
    lo, hi = args.frames
    cand = list(range(lo - (lo % args.stride), hi + 1, args.stride))
    jobs = []
    for pid, g in kdf.groupby("global_player_id"):
        pid = int(pid)
        if args.ids is not None and pid not in args.ids:
            continue
        frames_s = select_frames(rec_of.get(pid, []), cand, args.reach)
        if not frames_s:
            continue
        by_clip = {int(f): rows for f, rows in g.groupby("frame")}
        frames, uv, conf, cams, gnd, init_bp, init_go, overrides = [], [], [], [], [], [], [], []
        betas_seen = []
        for fs in frames_s:
            fc = fs + off
            rows = by_clip.get(fc)
            if rows is None or len(rows) != 17 or fc >= len(tr.conf) or tr.conf[fc] <= 0 or pid not in ground.get(fc, {}):
                continue
            rows = rows.sort_values("joint")
            u, c = coco_to_body(rows[["x", "y"]].to_numpy(float), rows["conf"].to_numpy(float), min_conf=args.min_conf)
            if int((c >= args.min_conf).sum()) < args.min_joints:
                continue
            intr, pose = tr.at(fc)
            K, R, t = intr.K(), np.asarray(pose.R, float), np.asarray(pose.t, float)
            bp = go = None
            tid = tid_of.get((fc, pid))
            recs = ez_by_tid.get(tid, {}) if tid is not None else {}
            if recs:
                fr = min(recs, key=lambda x: abs(x - fc))
                r = recs[fr]
                bp = np.asarray(r["body_pose"], float).reshape(-1)
                _i, pose_r = tr.at(fr)
                go = (Rotation.from_matrix(np.asarray(pose_r.R, float).T)
                      * Rotation.from_rotvec(np.asarray(r["global_orient"], float).reshape(3))).as_rotvec()
                betas_seen.append(np.asarray(r["betas"], float)[:10])
            b = box.get((fc, pid))
            if b is not None and med_w.get(pid) and (b[2] - b[0]) > args.max_width_ratio * float(med_w[pid]):
                n_merged += 1
                continue
            lying = (b is not None and (fs, pid) not in side_keys
                     and (b[3] - b[1]) / max(1.0, b[2] - b[0]) < args.lying_aspect)
            frames.append(fs); uv.append(u); conf.append(c); cams.append((K, R, t))
            gnd.append(np.asarray(ground[fc][pid], float)); init_bp.append(bp); init_go.append(go)
            overrides.append({"lying": True} if lying else None)
        if len(frames) < 2:
            continue
        betas = np.mean(betas_seen, axis=0) if betas_seen else np.zeros(10)
        if not args.no_roster_betas:
            betas = _p05.roster_betas(args, P, pid, betas)
        has_init = all(x is not None for x in init_bp)
        jobs.append({"pid": pid, "frames": np.asarray(frames), "uv": np.stack(uv), "conf": np.stack(conf), "cams": cams,
                     "ground": np.stack(gnd), "betas": betas,
                     "init_bp": np.stack(init_bp) if has_init else None, "init_go": np.stack(init_go) if has_init else None,
                     "rec_frames": np.asarray([]), "rec_bp": [], "body_models": args.body_models, "max_gap": args.max_gap,
                     "prev_seq": [None] * len(frames), "cfg_overrides": overrides, "blend_seq": [None] * len(frames),
                     "reproj_px_max": args.reproj_px_max, "truth": None,
                     "cfg": {"min_conf": args.min_conf, "min_joints": args.min_joints, "max_iter": args.max_iter}})
    n_frames = sum(len(j["frames"]) for j in jobs)
    print(f"{len(jobs)} ids with endzone keypoints and no record within {args.reach} frames on {lo}-{hi}: {n_frames} "
          f"frames to fit ({sum(sum(1 for o in j['cfg_overrides'] if o) for j in jobs)} on the ground; {n_merged} skipped "
          f"on a merged box); {n_rej} keypoint outliers rejected")
    if not jobs:
        raise SystemExit("nothing to fit")
    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_p05._fit_job, jobs, chunksize=1)
    else:
        results = [_p05._fit_job(j) for j in jobs]
    fits, betas_of = {}, {j["pid"]: j["betas"] for j in jobs}
    for pid, frames, params, valid, rep, before, _spb, _spa, _val in sorted(results, key=lambda r: r[0]):
        fits[pid] = (frames, params, valid)
        print(f"  id {pid:3d}: {int(valid.sum()):3d}/{len(frames):3d} frames ({frames[0]}-{frames[-1]}), reproj "
              f"{np.nanmedian(before):5.1f} -> {np.nanmedian(rep[valid]) if valid.any() else float('nan'):5.1f} px")
    print(f"{time.time() - t0:.0f} s")
    if args.dry_run:
        return
    merged, added = merge_into_refit(blob, fits, betas_of, source="endzone-mono2d")
    for pid, (frames, _params, valid) in fits.items():          # the pose only: the loader keeps its own placement
        for f, ok in zip(frames, valid):
            r = merged["frames"].get(int(f), {}).get(int(pid))
            if ok and r is not None and int(pid) not in blob["frames"].get(int(f), {}):
                r["no_place"] = True
    out = args.out or P / "poses_refit_ez.json"
    with open(out, "wb") as fh:
        pickle.dump(merged, fh)
    print(f"wrote {out}: {added} endzone one-view records added")


if __name__ == "__main__":
    main()
