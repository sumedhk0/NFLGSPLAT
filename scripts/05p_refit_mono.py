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
from nfl_gsplat.pose.keypoint_filter import GATE_PX, reject_outliers
from nfl_gsplat.pose.fit_mono2d import (Mono2DConfig, blend_params, body_frame_speeds, fit_sequence_2d,
                                        merge_into_refit, rigid_start_2d, tilt_rad)
from nfl_gsplat.pose.forward_kinematics import fk_forward, load_smplx_skeleton
from nfl_gsplat.pose.fuse_smplx import SMPLXFitConfig, _pack_params
from nfl_gsplat.render.play_timeline import ground_positions

FPS = 59.94
# Inside a player's fused span (frames the triangulation dropped) the one-view
# fit is a refinement of the interpolated two-view fit: with the default
# weights (place 1, init 0.05) the boundary jump stayed 0.37 m / 18 deg
# (play 1, 187 boundaries); these hold the pelvis on the fused path and the
# pose near the fused one, the keypoints correcting within that.
INSIDE_PLACE_WEIGHT = 10.0
INSIDE_INIT_WEIGHT = 1.0
EDGE_INIT_WEIGHT = 0.5


def heading_diff_deg(go_a, go_b):
    """Angle between two bodies' facing directions on the ground plane (SMPL-X faces +z in model axes)."""
    from scipy.spatial.transform import Rotation

    fa = Rotation.from_rotvec(np.asarray(go_a, float)).apply([0.0, 0.0, 1.0])[:2]
    fb = Rotation.from_rotvec(np.asarray(go_b, float)).apply([0.0, 0.0, 1.0])[:2]
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(fa @ fb / (na * nb), -1.0, 1.0))))


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
    starts = [None] * len(frames)
    for i in range(len(frames)):
        try:
            p0, e = rigid_start_2d(rest, job["ground"][i], job["cams"][i], job["uv"][i], job["conf"][i], forward,
                                   base, None if init_bp is None else init_bp[i], min_conf=cfg.min_conf,
                                   init_orient=None if init_go is None else init_go[i], heading_span_deg=0.0)
            before[i] = e
            starts[i] = p0
        except ValueError:
            pass
    params, valid, rep = fit_sequence_2d(job["uv"], job["conf"], job["cams"], job["ground"], rest, forward,
                                         cfg=cfg, base_cfg=base, init_body_pose_seq=init_bp,
                                         init_orient_seq=init_go, frames=frames, max_gap=job["max_gap"],
                                         prev_seq=job.get("prev_seq"), cfg_overrides=job.get("cfg_overrides"))
    valid &= np.nan_to_num(rep, nan=np.inf) <= job["reproj_px_max"]
    # cross-fade into a bordering two-view block: w = 1 at the block's edge, 0 a max-gap away
    blend = job.get("blend_seq")
    if blend is not None:
        for i in range(len(frames)):
            if valid[i] and blend[i] is not None:
                anchor, w = blend[i]
                params[i] = blend_params(params[i], anchor, w, base_cfg=base)
    sp_after = np.array([])
    if valid.sum() >= 2:
        sp_after = body_frame_speeds(params[valid], np.asarray(frames)[valid], forward, fps=FPS, base_cfg=base)
    sp_before = np.array([])
    rf, rbp = job["rec_frames"], job["rec_bp"]
    if len(rf) >= 2:
        sp_before = body_frame_speeds([_pack_params(b, np.zeros(3), np.zeros(3)) for b in rbp], rf, forward,
                                      fps=FPS, base_cfg=base)
    truth = job.get("truth")
    val = None
    if truth is not None:
        # against the fused refit on the same frames: joint error (m), tilt, heading
        rows = []
        for i, f in enumerate(frames):
            tp = truth.get(int(f))
            if tp is None or not valid[i]:
                continue
            Jt = forward(tp)
            Jf = forward(params[i])
            row = [np.linalg.norm(Jf - Jt, axis=1).mean(), tilt_rad(params[i][63:66]), tilt_rad(tp[63:66]), np.nan,
                   np.linalg.norm((Jf - Jf[0]) - (Jt - Jt[0]), axis=1).mean(), np.linalg.norm(Jf[0] - Jt[0]),
                   heading_diff_deg(params[i][63:66], tp[63:66])]
            if starts[i] is not None:
                J0 = forward(starts[i])
                row[3] = np.linalg.norm((J0 - J0[0]) - (Jt - Jt[0]), axis=1).mean()
            rows.append(row)
        val = np.array(rows) if rows else np.zeros((0, 7))
    return (job["pid"], frames, params, valid, rep, before, sp_before, sp_after, val)


def _two_view_job(job):
    """One two-camera player: (pid, frames, params, valid, reproj, reproj per view)."""
    cfg = Mono2DConfig(**job["cfg"])
    base = SMPLXFitConfig()
    rest, _ = load_smplx_skeleton(job["body_models"], betas=job["betas"])
    forward = fk_forward(rest)
    params, valid, rep = fit_sequence_2d(job["uv"], job["conf"], job["cams"], job["ground"], rest, forward,
                                         cfg=cfg, base_cfg=base, init_body_pose_seq=job["init_bp"],
                                         init_orient_seq=job["init_go"], frames=job["frames"], max_gap=job["max_gap"])
    valid &= np.nan_to_num(rep, nan=np.inf) <= job["reproj_px_max"]
    # reprojection per view on the valid frames, for the report
    per_view = [[], []]
    from nfl_gsplat.pose.fit_mono2d import project
    for i in range(len(job["frames"])):
        if not valid[i]:
            continue
        J = forward(params[i])
        for v, (K, R, t) in enumerate(job["cams"][i]):
            use = np.asarray(job["conf"][i][v], float) >= cfg.min_conf
            if use.sum():
                pix, _ = project(K, R, t, J[use])
                per_view[v].append(float(np.median(np.linalg.norm(pix - np.asarray(job["uv"][i][v], float)[use], axis=1))))
    return (job["pid"], job["frames"], params, valid, rep, per_view)


def two_view_pass(args, P, tracks, df, ground, blob):
    """Refit the two-camera players to both cameras' keypoints; the records replace 05f's in ``blob``."""
    from scipy.spatial.transform import Rotation

    kall = pd.read_parquet(args.keypoints or P / "keypoints_2d.parquet")
    if not args.no_keypoint_filter:
        kall, _ = reject_outliers(kall)
    cams_present = sorted(kall["cam"].unique())
    if len(cams_present) < 2:
        print("two-view: keypoints from one camera only; nothing to do")
        return blob
    other = [c for c in cams_present if c != args.cam][0]
    tr_a, tr_b = tracks[args.cam], tracks[other]
    offset = 0
    if (P / "poses_tri.json").exists():
        offset = int(pickle.load(open(P / "poses_tri.json", "rb")).get("offset", 0))
    side = pickle.load(open(args.poses or P / "poses_sideline.json", "rb"))
    recs_of: dict = {}
    for f, recs in side["frames"].items():
        for pid, r in recs.items():
            recs_of.setdefault(int(pid), {})[int(f)] = r
    ka = {(int(f), int(p)): g.sort_values("joint") for (f, p), g in kall[kall["cam"] == args.cam].groupby(["frame", "global_player_id"])}
    kb = {(int(f), int(p)): g.sort_values("joint") for (f, p), g in kall[kall["cam"] == other].groupby(["frame", "global_player_id"])}
    pids = sorted({p for (_, p) in ka} & {p for (_, p) in kb})
    jobs = []
    for pid in pids:
        if args.ids is not None and pid not in args.ids:
            continue
        rec = recs_of.get(pid, {})
        rec_frames = sorted(rec)
        betas = (np.mean([np.asarray(rec[f]["betas"], float) for f in rec_frames], axis=0) if rec_frames
                 else np.zeros(10))
        frames, uv, conf, cams, gnd, init_bp, init_go = [], [], [], [], [], [], []
        for f in sorted(f for (f, p) in ka if p == pid):
            if f % args.stride or (f + offset, pid) not in kb or pid not in ground.get(f, {}):
                continue
            if f >= len(tr_a.conf) or tr_a.conf[f] <= 0 or f + offset >= len(tr_b.conf) or tr_b.conf[f + offset] <= 0:
                continue
            ga, gb = ka[(f, pid)], kb[(f + offset, pid)]
            if len(ga) != 17 or len(gb) != 17:
                continue
            ua, ca = coco_to_body(ga[["x", "y"]].to_numpy(float), ga["conf"].to_numpy(float), min_conf=args.min_conf)
            ub, cb = coco_to_body(gb[["x", "y"]].to_numpy(float), gb["conf"].to_numpy(float), min_conf=args.min_conf)
            if int((ca >= args.min_conf).sum()) + int((cb >= args.min_conf).sum()) < args.min_joints:
                continue
            ia, pa = tr_a.at(f)
            ib, pb = tr_b.at(f + offset)
            cam_a = (ia.K(), np.asarray(pa.R, float), np.asarray(pa.t, float))
            cam_b = (ib.K(), np.asarray(pb.R, float), np.asarray(pb.t, float))
            if rec_frames:
                fr = min(rec_frames, key=lambda x: abs(x - f))
                r = rec[fr]
                bp = np.asarray(r["body_pose"], float).reshape(-1)
                _i, pr = tr_a.at(fr)
                go = (Rotation.from_matrix(np.asarray(pr.R, float).T)
                      * Rotation.from_rotvec(np.asarray(r["global_orient"], float).reshape(3))).as_rotvec()
            else:
                bp, go = None, None
            frames.append(f)
            uv.append([ua, ub])
            conf.append([ca, cb])
            cams.append([cam_a, cam_b])
            gnd.append(np.asarray(ground[f][pid], float))
            init_bp.append(bp)
            init_go.append(go)
        if len(frames) < 8:
            continue
        has_init = all(b is not None for b in init_bp)
        jobs.append({"pid": pid, "frames": np.asarray(frames), "uv": uv, "conf": conf, "cams": cams,
                     "ground": np.stack(gnd), "betas": betas,
                     "init_bp": np.stack(init_bp) if has_init else None,
                     "init_go": np.stack(init_go) if has_init else None,
                     "body_models": args.body_models, "max_gap": args.max_gap,
                     "reproj_px_max": args.reproj_px_max,
                     "cfg": {"min_conf": args.min_conf, "min_joints": args.min_joints, "max_iter": args.max_iter,
                             "tilt_weight": args.tilt_weight, "tilt_free_deg": args.tilt_free_deg,
                             "place_weight": TWO_VIEW_PLACE_WEIGHT}})
    n_frames = sum(len(j["frames"]) for j in jobs)
    print(f"two-view: {len(jobs)} players with keypoints in both cameras, {n_frames} frames (endzone offset {offset:+d}, "
          f"stride {args.stride}), {args.workers} workers")
    if not jobs:
        return blob
    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_two_view_job, jobs, chunksize=1)
    else:
        results = [_two_view_job(j) for j in jobs]
    fits, betas_of = {}, {j["pid"]: j["betas"] for j in jobs}
    ra, rb = [], []
    n_ok = 0
    for pid, frames, params, valid, rep, per_view in sorted(results, key=lambda r: r[0]):
        fits[pid] = (frames, params, valid)
        ra.extend(per_view[0])
        rb.extend(per_view[1])
        n_ok += int(valid.sum())
    print(f"two-view: fitted {n_ok}/{n_frames} frames on {len(fits)} players; reprojection median {args.cam} "
          f"{np.median(ra):.1f} px, {other} {np.median(rb):.1f} px")
    # these records replace 05f's for the same players (the fused cache loses those players entirely,
    # so the one-view pass treats their remaining frames as one-view)
    frames = {int(f): {int(p): r for p, r in recs.items() if int(p) not in fits} for f, recs in blob["frames"].items()}
    frames = {f: r for f, r in frames.items() if r}
    new_blob = dict(blob)
    new_blob["frames"] = frames
    merged, added = merge_into_refit(new_blob, fits, betas_of, source="two-view keypoints")
    merged.pop("mono", None)
    print(f"two-view: {added} records replace 05f's for {len(fits)} players")
    return merged


TWO_VIEW_PLACE_WEIGHT = 0.3


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
    ap.add_argument("--no-keypoint-filter", action="store_true",
                    help="skip the temporal outlier rejection on the keypoints (pose.keypoint_filter)")
    ap.add_argument("--one-view-only", action="store_true",
                    help="ignore the fused (05f) cache: every player is fitted to the sideline keypoints alone "
                         "(an experiment: the pairing's ~1 m ambiguity corrupts two-view poses and placement)")
    ap.add_argument("--two-view", action="store_true",
                    help="fit the two-camera players to BOTH cameras' keypoints (no triangulation) and write them as "
                         "the fused records, replacing 05f's for those players; then the one-view pass as usual")
    ap.add_argument("--validate", action="store_true",
                    help="fit the frames the fused refit COVERS and score against it (joint error, tilt); writes nothing")
    ap.add_argument("--tilt-weight", type=float, default=Mono2DConfig.tilt_weight)
    ap.add_argument("--tilt-free-deg", type=float, default=Mono2DConfig.tilt_free_deg)
    args = ap.parse_args()
    P = args.play_dir
    t0 = time.time()

    tracks = load_camera_track(P / "cameras.npz")
    tr = tracks[args.cam]
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    # This camera's own ground points: ground_positions averages both cameras
    # where both see an id, and the fused offset carried beyond a span was
    # measured against that average and applied to the sideline-only point
    # afterwards (the endzone's share along x is what stayed as a 0.37 m jump).
    ground = ground_positions(df[df["cam"] == args.cam], tracks)
    kdf_all = pd.read_parquet(args.keypoints or P / "keypoints_2d.parquet")
    if not args.no_keypoint_filter:
        kdf_all, n_rej = reject_outliers(kdf_all)
        print(f"keypoint outliers rejected: {n_rej} of {int((kdf_all['conf'] > 0).sum() + n_rej)} confident "
              f"(a one-frame jump past {GATE_PX:.0f} px from its neighbours' midpoints)")
    kdf = kdf_all[kdf_all["cam"] == args.cam]
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
    if args.one_view_only and not args.validate:
        fused_backup_two = P / "poses_refit_fused_05f.json"
        if refit_path.exists() and not fused_backup_two.exists():
            import shutil
            shutil.copy(refit_path, fused_backup_two)
            print(f"kept 05f's cache as {fused_backup_two.name}")
        blob = {"cam": "fused", "world": True, "appearance_cam": args.cam, "stride": int(side["stride"]), "frames": {}}
        print("one-view only: the fused cache is ignored")
    if args.two_view and not args.validate:
        fused_backup_two = P / "poses_refit_fused_05f.json"
        if refit_path.exists() and not fused_backup_two.exists():
            import shutil
            shutil.copy(refit_path, fused_backup_two)
            print(f"kept 05f's cache as {fused_backup_two.name}")
        blob = two_view_pass(args, P, tracks, df, ground, blob)
    fused_frames = {int(f): set(int(p) for p in recs) for f, recs in blob["frames"].items()}
    # the fused records per player, for continuity across and beyond their span
    fused_of: dict[int, dict[int, np.ndarray]] = {}
    fused_pelvis: dict[int, dict[int, np.ndarray]] = {}       # the pelvis on the field: transl + rest pelvis
    rest_pelvis_cache: dict = {}
    for f, recs in blob["frames"].items():
        for pid, r in recs.items():
            fused_of.setdefault(int(pid), {})[int(f)] = _pack_params(
                np.asarray(r["body_pose"], float).reshape(-1), np.asarray(r["global_orient"], float).reshape(3),
                np.asarray(r["transl"], float).reshape(3))
            key = tuple(np.round(np.asarray(r["betas"], float)[:10], 3))
            if key not in rest_pelvis_cache:
                rest_pelvis_cache[key] = load_smplx_skeleton(args.body_models, betas=np.asarray(r["betas"], float))[0][0][:2]
            fused_pelvis.setdefault(int(pid), {})[int(f)] = np.asarray(r["transl"], float)[:2] + rest_pelvis_cache[key]
    if args.validate:
        args.dry_run = True

    # the regressor's records per player: {pid: {frame: rec}}
    recs_of: dict[int, dict[int, dict]] = {}
    for f, recs in side["frames"].items():
        for pid, r in recs.items():
            recs_of.setdefault(int(pid), {})[int(f)] = r

    jobs = []
    n_short_gap = 0
    for pid, g in kdf.groupby("global_player_id"):
        pid = int(pid)
        if args.ids is not None and pid not in args.ids:
            continue
        rec = recs_of.get(pid, {})
        rec_frames = sorted(rec)
        betas = (np.mean([np.asarray(rec[f]["betas"], float) for f in rec_frames], axis=0) if rec_frames
                 else np.zeros(10))
        frames, uv, conf, cams, gnd, init_bp, init_go = [], [], [], [], [], [], []
        truth = {}
        for f, rows in g.groupby("frame"):
            f = int(f)
            covered = pid in fused_frames.get(f, set())
            if f % args.stride or covered != args.validate or len(rows) != 17:
                continue
            if args.validate:
                r = blob["frames"][f][pid]
                truth[f] = _pack_params(np.asarray(r["body_pose"], float).reshape(-1),
                                        np.asarray(r["global_orient"], float).reshape(3),
                                        np.asarray(r["transl"], float).reshape(3))
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
        # Continuity with the player's two-view fit (fused records win where they
        # exist): inside the fused span the ground point is the fused pelvis
        # interpolated, the start pose the nearest fused one, and the warm start at
        # a block boundary the fused params; beyond the span the box point carries
        # the span end's offset from the fused pelvis, decayed over --max-gap frames.
        prev_seq = [None] * len(frames)
        overrides = [None] * len(frames)
        blend_seq = [None] * len(frames)
        fr = fused_of.get(pid, {})
        if fr and not args.validate:
            ff = np.array(sorted(fr))
            # A short gap in the fused coverage is interpolated by the timeline
            # (poses and, place_from_refit, the translation): a one-view fit in
            # it pops -- measured on play 1, second differences at mixed
            # fused/one-view triples 227 mm (hands/feet p90 778) against 18
            # (140) at fused-only ones, whatever the weights. Only gaps longer
            # than --max-gap are filled.
            keep = []
            for i, f in enumerate(frames):
                k = int(np.searchsorted(ff, f))
                in_short_gap = 0 < k < len(ff) and (ff[k] - ff[k - 1] - 1) <= args.max_gap
                if in_short_gap:
                    n_short_gap += 1
                else:
                    keep.append(i)
            if len(keep) < 2:
                continue
            frames = [frames[i] for i in keep]
            uv, conf, cams, gnd = [uv[i] for i in keep], [conf[i] for i in keep], [cams[i] for i in keep], [gnd[i] for i in keep]
            init_bp, init_go = [init_bp[i] for i in keep], [init_go[i] for i in keep]
            prev_seq, overrides, blend_seq = [None] * len(frames), [None] * len(frames), [None] * len(frames)
            fp = fused_pelvis[pid]
            fx = np.stack([fp[f] for f in ff])                 # the fused PELVIS, which the fit places at gnd
            gnd_arr = np.stack(gnd)
            for i, f in enumerate(frames):
                k = int(np.argmin(np.abs(ff - f)))
                near = int(ff[k])
                if ff[0] < f < ff[-1]:
                    gnd_arr[i] = np.array([np.interp(f, ff, fx[:, 0]), np.interp(f, ff, fx[:, 1])])
                    init_bp[i] = fr[near][:63]
                    init_go[i] = fr[near][63:66]
                    # inside the span the keypoints refine the fused fit, not replace it
                    overrides[i] = {"place_weight": INSIDE_PLACE_WEIGHT, "init_weight": INSIDE_INIT_WEIGHT}
                else:
                    edge = int(ff[0]) if f < ff[0] else int(ff[-1])
                    # the fused pelvis vs this camera's box point at the span's end
                    off = fp[edge] - ground.get(edge, {}).get(pid, fp[edge])
                    w = max(0.0, 1.0 - abs(f - edge) / float(args.max_gap * 5))
                    gnd_arr[i] = gnd_arr[i] + w * np.asarray(off, float)
                    # the corrected point is trusted as far as the correction reaches
                    overrides[i] = {"place_weight": 1.0 + (INSIDE_PLACE_WEIGHT - 1.0) * w}
                    if abs(f - edge) <= args.max_gap:
                        init_bp[i] = fr[edge][:63]
                        init_go[i] = fr[edge][63:66]
                        overrides[i]["init_weight"] = EDGE_INIT_WEIGHT
                # a block boundary: the previous fitted frame is not this one's neighbour
                if abs(near - f) <= args.stride and (i == 0 or frames[i - 1] < near):
                    prev_seq[i] = fr[near]
                # within a max-gap of a fused block: cross-fade toward its edge record
                d = abs(near - f)
                if d <= args.max_gap:
                    blend_seq[i] = (fr[near], 1.0 - d / float(args.max_gap + 1))
            gnd = list(gnd_arr)
        has_init = all(b is not None for b in init_bp)
        jobs.append({"pid": pid, "frames": np.asarray(frames), "uv": np.stack(uv), "conf": np.stack(conf),
                     "cams": cams, "ground": np.stack(gnd), "betas": betas,
                     "init_bp": np.stack(init_bp) if has_init else None,
                     "init_go": np.stack(init_go) if has_init else None,
                     "rec_frames": np.asarray(rec_frames),
                     "rec_bp": [np.asarray(rec[f]["body_pose"], float).reshape(-1) for f in rec_frames],
                     "body_models": args.body_models, "max_gap": args.max_gap, "prev_seq": prev_seq,
                     "cfg_overrides": overrides, "blend_seq": blend_seq,
                     "reproj_px_max": args.reproj_px_max, "truth": truth if args.validate else None,
                     "cfg": {"min_conf": args.min_conf, "min_joints": args.min_joints, "max_iter": args.max_iter,
                             "tilt_weight": args.tilt_weight, "tilt_free_deg": args.tilt_free_deg}})
    n_frames = sum(len(j["frames"]) for j in jobs)
    print(f"{len(jobs)} players with {args.cam} keypoints {'inside' if args.validate else 'outside'} the fused refit, "
          f"{n_frames} frames to fit (stride {args.stride}; {n_short_gap} in fused gaps of <= {args.max_gap} frames "
          f"left to the interpolation), {args.workers} workers"
          + (f"; tilt prior {args.tilt_weight} past {args.tilt_free_deg} deg" if args.tilt_weight > 0 else ""))
    if not jobs:
        raise SystemExit("nothing to fit")

    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_fit_job, jobs, chunksize=1)
    else:
        results = [_fit_job(j) for j in jobs]

    fits, betas_of = {}, {j["pid"]: j["betas"] for j in jobs}
    all_before, all_after, sp_b, sp_a, vals = [], [], [], [], []
    nan = float("nan")
    for pid, frames, params, valid, rep, before, spb, spa, val in sorted(results, key=lambda r: r[0]):
        if val is not None:
            vals.append(val)
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
    if args.validate:
        v = np.concatenate(vals) if vals else np.zeros((0, 7))
        print(f"VALIDATE against the fused refit on {len(v)} frames: pelvis-aligned joint error p50 "
              f"{np.nanmedian(v[:, 3]):.2f} m (regressor) -> {np.median(v[:, 4]):.2f} m (mono fit), p90 "
              f"{np.percentile(v[:, 4], 90):.2f}; with placement {np.median(v[:, 0]):.2f} m (pelvis off by "
              f"{np.median(v[:, 5]):.2f}); "
              f"tilt p50 fit {np.degrees(np.median(v[:, 1])):.0f} vs fused {np.degrees(np.median(v[:, 2])):.0f} deg, "
              f"|tilt diff| p50 {np.degrees(np.median(np.abs(v[:, 1] - v[:, 2]))):.0f} deg; "
              f"fit over 60 deg {100 * np.mean(v[:, 1] > np.radians(60)):.0f}% (fused {100 * np.mean(v[:, 2] > np.radians(60)):.0f}%); "
              f"heading |diff| p50 {np.nanmedian(v[:, 6]):.0f} deg, p90 {np.nanpercentile(v[:, 6], 90):.0f}, "
              f"over 90 deg (facing the wrong way) {100 * np.nanmean(v[:, 6] > 90):.0f}%")
        return
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
