#!/usr/bin/env python
"""Refit the one-view bodies to the sideline's 2-D keypoints (pose.fit_mono2d).

WHY. The fused refit (05f) covers the players both cameras see; the rest
render from the monocular regressor's pose (05c), which sits near the mean
pose: on play 1 v14 their joints moved 0.25 m/s in the body frame against
1.0 m/s for triangulated bodies -- mannequins gliding. The 2-D keypoints
(05m) exist for every tracked person and hold the articulation.

WHAT. For every player the sideline keypoints see, every frame the fused
refit has no record for: least squares over SMPL-X (body_pose, orient,
transl) to the keypoints through that frame's sideline camera, feet on the
turf, the pelvis over the box-bottom ground point, started from the
regressor's pose and heading (nearest 05c record). Records are merged into
05f's cache -- the fused record wins where both exist -- so the timeline
draws them world-placed like the triangulated bodies.

    poses_refit.json        -> poses_refit_fused.json   (the input kept, fused only)
    poses_refit.json        <- fused + one-view records ("mono" key marks it)

A frame whose fit reprojects worse than --reproj-px-max keeps the
regressor's pose (no record written). Printed per player: frames fitted,
reprojection before (the regressor's pose placed as the timeline places it)
and after, body-frame joint speed before and after; then the totals.

Runs in the smplx env (the SMPL-X model gives the rest skeleton per shape).
"""
from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.pose.coco import coco_to_body
from nfl_gsplat.pose.fit_mono2d import (Mono2DConfig, body_frame_speeds, fit_sequence_2d, merge_into_refit,
                                        rigid_start_2d)
from nfl_gsplat.pose.forward_kinematics import fk_forward, load_smplx_skeleton
from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig, _pack_params
from nfl_gsplat.render.play_timeline import ground_positions

FPS = 59.94


def _fit_job(job):
    """One player: (pid, frames, params, valid, reproj, before_reproj, speeds_before, speeds_after)."""
    cfg = Mono2DConfig(**job["cfg"])
    base = SMPLXFitConfig()
    rest, _ = load_smplx_skeleton(job["body_models"], betas=job["betas"])
    forward = fk_forward(rest)
    frames = job["frames"]
    init_bp, init_go = job["init_bp"], job["init_go"]
    # before: the regressor's pose and heading placed as the timeline places them
    before = np.full(len(frames), np.nan)
    for i in range(len(frames)):
        try:
            _, e = rigid_start_2d(rest, job["ground"][i], job["cams"][i], job["uv"][i], job["conf"][i], forward,
                                  base, None if init_bp is None else init_bp[i], min_conf=cfg.min_conf,
                                  init_orient=None if init_go is None else init_go[i], heading_span_deg=0.0)
            before[i] = e
        except ValueError:
            pass
    params, valid, rep = fit_sequence_2d(job["uv"], job["conf"], job["cams"], job["ground"], rest, forward,
                                         cfg=cfg, base_cfg=base, init_body_pose_seq=init_bp,
                                         init_orient_seq=init_go, frames=frames, max_gap=job["max_gap"])
    valid &= np.nan_to_num(rep, nan=np.inf) <= job["reproj_px_max"]
    sp_after = np.array([])
    if valid.sum() >= 2:
        sp_after = body_frame_speeds(params[valid], np.asarray(frames)[valid], forward, fps=FPS, base_cfg=base)
    sp_before = np.array([])
    rf, rbp = job["rec_frames"], job["rec_bp"]
    if len(rf) >= 2:
        sp_before = body_frame_speeds([_pack_params(b, np.zeros(3), np.zeros(3)) for b in rbp], rf, forward,
                                      fps=FPS, base_cfg=base)
    return (job["pid"], frames, params, valid, rep, before, sp_before, sp_after)


def main() -> None:
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--keypoints", type=Path, default=None, help="default <play-dir>/keypoints_2d.parquet")
    ap.add_argument("--poses", type=Path, default=None,
                    help="05c sideline cache; default <play-dir>/poses_sideline.json")
    ap.add_argument("--refit", type=Path, default=None,
                    help="05f cache to merge into; default <play-dir>/poses_refit.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: the --refit path (its input kept as poses_refit_fused.json)")
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--stride", type=int, default=2,
                    help="fit every n-th frame (the timeline interpolates; 05k renders at 2)")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-joints", type=int, default=6)
    ap.add_argument("--reproj-px-max", type=float, default=20.0)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--max-gap", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="only these player ids (a probe)")
    ap.add_argument("--dry-run", action="store_true", help="fit and report, write nothing")
    args = ap.parse_args()
    P = args.play_dir
    t0 = time.time()

    tracks = load_camera_track(P / "cameras.npz")
    tr = tracks[args.cam]
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    ground = ground_positions(df, tracks)
    kdf = pd.read_parquet(args.keypoints or P / "keypoints_2d.parquet")
    kdf = kdf[kdf["cam"] == args.cam]
    side = pickle.load(open(args.poses or P / "poses_sideline.json", "rb"))
    if side["cam"] != args.cam:
        raise SystemExit(f"the pose cache is the {side['cam']} camera's, not {args.cam}")
    refit_path = args.refit or P / "poses_refit.json"
    fused_backup = P / "poses_refit_fused.json"
    if refit_path.exists():
        blob = pickle.load(open(refit_path, "rb"))
        if "mono" in blob:
            if not fused_backup.exists():
                raise SystemExit(f"{refit_path} already holds one-view records and {fused_backup.name} is missing")
            print(f"{refit_path.name} already merged; starting from {fused_backup.name}")
            refit_path = fused_backup
            blob = pickle.load(open(refit_path, "rb"))
    else:
        print(f"no fused refit at {refit_path}: every keypointed frame is one-view")
        blob = {"cam": "fused", "world": True, "appearance_cam": args.cam, "stride": int(side["stride"]),
                "frames": {}}
    fused_frames = {int(f): set(int(p) for p in recs) for f, recs in blob["frames"].items()}

    # the regressor's records per player: {pid: {frame: rec}}
    recs_of: dict[int, dict[int, dict]] = {}
    for f, recs in side["frames"].items():
        for pid, r in recs.items():
            recs_of.setdefault(int(pid), {})[int(f)] = r

    jobs = []
    for pid, g in kdf.groupby("global_player_id"):
        pid = int(pid)
        if args.ids is not None and pid not in args.ids:
            continue
        rec = recs_of.get(pid, {})
        rec_frames = sorted(rec)
        betas = (np.mean([np.asarray(rec[f]["betas"], float) for f in rec_frames], axis=0) if rec_frames
                 else np.zeros(10))
        frames, uv, conf, cams, gnd, init_bp, init_go = [], [], [], [], [], [], []
        for f, rows in g.groupby("frame"):
            f = int(f)
            if f % args.stride or pid in fused_frames.get(f, set()) or len(rows) != 17:
                continue
            if f >= len(tr.conf) or tr.conf[f] <= 0 or pid not in ground.get(f, {}):
                continue
            rows = rows.sort_values("joint")
            u, c = coco_to_body(rows[["x", "y"]].to_numpy(float), rows["conf"].to_numpy(float),
                                min_conf=args.min_conf)
            if int((c >= args.min_conf).sum()) < args.min_joints:
                continue
            intr, pose = tr.at(f)
            K, R, t = intr.K(), np.asarray(pose.R, float), np.asarray(pose.t, float)
            if rec_frames:
                fr = min(rec_frames, key=lambda x: abs(x - f))
                r = rec[fr]
                bp = np.asarray(r["body_pose"], float).reshape(-1)
                _intr_r, pose_r = tr.at(fr)
                # the regressor's orientation is in camera axes; world = R^T (placement_transform)
                go = (Rotation.from_matrix(np.asarray(pose_r.R, float).T)
                      * Rotation.from_rotvec(np.asarray(r["global_orient"], float).reshape(3))).as_rotvec()
            else:
                bp, go = None, None
            frames.append(f)
            uv.append(u)
            conf.append(c)
            cams.append((K, R, t))
            gnd.append(np.asarray(ground[f][pid], float))
            init_bp.append(bp)
            init_go.append(go)
        if len(frames) < 2:
            continue
        has_init = all(b is not None for b in init_bp)
        jobs.append({"pid": pid, "frames": np.asarray(frames), "uv": np.stack(uv), "conf": np.stack(conf),
                     "cams": cams, "ground": np.stack(gnd), "betas": betas,
                     "init_bp": np.stack(init_bp) if has_init else None,
                     "init_go": np.stack(init_go) if has_init else None,
                     "rec_frames": np.asarray(rec_frames),
                     "rec_bp": [np.asarray(rec[f]["body_pose"], float).reshape(-1) for f in rec_frames],
                     "body_models": args.body_models, "max_gap": args.max_gap,
                     "reproj_px_max": args.reproj_px_max,
                     "cfg": {"min_conf": args.min_conf, "min_joints": args.min_joints, "max_iter": args.max_iter}})
    n_frames = sum(len(j["frames"]) for j in jobs)
    print(f"{len(jobs)} players with {args.cam} keypoints outside the fused refit, {n_frames} frames to fit "
          f"(stride {args.stride}), {args.workers} workers")
    if not jobs:
        raise SystemExit("nothing to fit")

    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_fit_job, jobs, chunksize=1)
    else:
        results = [_fit_job(j) for j in jobs]

    fits, betas_of = {}, {j["pid"]: j["betas"] for j in jobs}
    all_before, all_after, sp_b, sp_a = [], [], [], []
    nan = float("nan")
    for pid, frames, params, valid, rep, before, spb, spa in sorted(results, key=lambda r: r[0]):
        fits[pid] = (frames, params, valid)
        all_before.extend(before[valid].tolist())
        all_after.extend(rep[valid].tolist())
        sp_b.extend(spb.reshape(-1).tolist())
        sp_a.extend(spa.reshape(-1).tolist())
        after_px = np.nanmedian(rep[valid]) if valid.any() else nan
        print(f"  id {pid:3d}: {int(valid.sum()):3d}/{len(frames):3d} frames, reproj {np.nanmedian(before):5.1f} -> "
              f"{after_px:5.1f} px, body-frame speed p50 {np.median(spb) if len(spb) else nan:.2f} -> "
              f"{np.median(spa) if len(spa) else nan:.2f} m/s")
    n_ok = sum(int(v.sum()) for _, _, v in fits.values())
    print(f"fitted {n_ok}/{n_frames} frames on {len(fits)} players; reprojection median "
          f"{np.nanmedian(all_before):.1f} -> {np.nanmedian(all_after):.1f} px; body-frame joint speed p50 "
          f"{np.median(sp_b):.2f} -> {np.median(sp_a):.2f} m/s (p90 {np.percentile(sp_b, 90):.2f} -> "
          f"{np.percentile(sp_a, 90):.2f}); {time.time() - t0:.0f} s")
    if args.dry_run:
        return
    merged, added = merge_into_refit(blob, fits, betas_of)
    out = args.out or (args.refit or P / "poses_refit.json")
    if refit_path.exists() and refit_path.resolve() != fused_backup.resolve():
        pickle.dump(blob, open(fused_backup, "wb"))
        print(f"kept the fused-only cache as {fused_backup.name}")
    with open(out, "wb") as fh:
        pickle.dump(merged, fh)
    print(f"wrote {out}: {added} one-view records added over {len(merged['frames'])} frames")


if __name__ == "__main__":
    main()
