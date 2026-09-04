#!/usr/bin/env python
"""High-fidelity render of a play: every player, every frame, upright, 1080p.

Replaces the world-mode preview for the deliverable. Per frame:

  timeline   render.timeline: every detected player's ground position (both
             views' feet fused, smoothed, short gaps bridged), pose from the
             fused refit / the single-view pose / the play's median stance,
             SLERP-interpolated to every frame, tilt clamped to 35 degrees
  bodies     SMPL-X forward per player, feet on the turf under the pelvis
  look       fitted appearance (05i) where a player has one, else the team
             colour; the procedural field in the rule-book frame
  render     compositing.splat_torch on the GPU, the whole scene, 1920x1080,
             sparse front-to-back compositing (no CUDA toolkit needed)
  output     PNG per rendered frame and an mp4 at the source rate / stride

Runs in the smplx env (SMPL-X forward; torch with CUDA).

    python scripts/05k_render_hifi.py --play-dir <P> --out-dir <P>/render_hifi
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)
BODY_RGB = (0.72, 0.62, 0.55)
TEAM_RGB = {"KC": (0.89, 0.09, 0.22), "BAL": (0.95, 0.95, 0.95), "ARI": (0.62, 0.11, 0.22),
            "SEA": (0.11, 0.22, 0.34)}


def ground_positions(df, tracks):
    """frame -> {pid: xy}: each view's foot through its camera, both averaged."""
    from nfl_gsplat.pose.place_on_field import ground_point

    out: dict[int, dict[int, list]] = {}
    for cam, sub in df.groupby("cam"):
        tr = tracks[cam]
        for f, rows in sub.groupby("frame"):
            f = int(f)
            if f >= len(tr.conf) or tr.conf[f] <= 0:
                continue
            intr, pose = tr.at(f)
            K, R, t = intr.K(), pose.R, pose.t
            for r in rows.itertuples():
                try:
                    g = ground_point((0.5 * (r.bbox_x1 + r.bbox_x2), float(r.bbox_y2)), K, R, t)
                except Exception:
                    continue
                if abs(g[0]) < 60 and abs(g[1]) < 30:
                    out.setdefault(f, {}).setdefault(int(r.track_id), []).append(np.asarray(g[:2], float))
    return {f: {pid: np.mean(v, axis=0) for pid, v in d.items()} for f, d in out.items()}


def poses_from_caches(refit, side_blob, tracks, model):
    """pid -> {frame: (body_pose, global_orient_world, betas, source)}."""
    import torch

    from nfl_gsplat.pose.place_on_field import placement_transform
    from scipy.spatial.transform import Rotation

    out: dict[int, dict[int, tuple]] = {}
    for f, recs in refit.items():
        for pid, r in recs.items():
            out.setdefault(int(pid), {})[int(f)] = (
                np.asarray(r["body_pose"], float).reshape(21, 3),
                np.asarray(r["global_orient"], float).reshape(3),
                np.asarray(r["betas"], float)[:10], "fused")
    if side_blob is not None:
        cam = side_blob["cam"]
        tr = tracks[cam]
        for f, recs in side_blob["frames"].items():
            f = int(f)
            if f >= len(tr.conf) or tr.conf[f] <= 0:
                continue
            intr, pose = tr.at(f)
            K, R, t = intr.K(), pose.R, pose.t
            for pid, r in recs.items():
                pid = int(pid)
                if f in out.get(pid, {}):
                    continue                                  # fused wins
                betas = np.asarray(r["betas"], np.float32)[None, :model.num_betas]
                body_pose = np.asarray(r["body_pose"], np.float32).reshape(1, -1)
                orient = np.asarray(r["global_orient"], np.float32).reshape(1, 3)
                with torch.no_grad():
                    res = model(betas=torch.tensor(betas), body_pose=torch.tensor(body_pose),
                                global_orient=torch.tensor(orient))
                joints = res.joints[0].numpy().astype(float)
                b = r["bbox"]
                foot = (0.5 * (b[0] + b[2]), float(b[3]))
                try:
                    rot_world, _off = placement_transform(joints, foot, K, R, t)
                except Exception:
                    continue
                go_cam = Rotation.from_rotvec(np.asarray(r["global_orient"], float).reshape(3))
                go_world = (Rotation.from_matrix(rot_world) * go_cam).as_rotvec()
                out.setdefault(pid, {})[f] = (np.asarray(r["body_pose"], float).reshape(21, 3),
                                              go_world, np.asarray(r["betas"], float)[:10], "sideline")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--poses-refit", type=Path, default=None)
    ap.add_argument("--poses-sideline", type=Path, default=None)
    ap.add_argument("--identity", type=Path, default=None)
    ap.add_argument("--appearance", type=Path, default=None, help="05i directory")
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--stride", type=int, default=2, help="render every n-th source frame")
    ap.add_argument("--fps", type=float, default=59.94)
    ap.add_argument("--field-res-m", type=float, default=0.12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="re-render frames whose PNG already exists (default: skip them)")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import pandas as pd
    import smplx
    import torch

    from nfl_gsplat.compositing import splat_torch as st
    from nfl_gsplat.compositing.merge_ply import batch_from_arrays
    from nfl_gsplat.compositing.mesh_to_gaussians import merge, mesh_to_gaussians
    from nfl_gsplat.compositing.preview_cpu import intrinsics, look_at
    from nfl_gsplat.field.procedural_field import render_field_texture, texture_to_gaussians
    from nfl_gsplat.render import timeline as tlm

    P = args.play_dir
    tracks = load_camera_track(P / "cameras.npz")
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    refit_path = args.poses_refit or (P / "poses_refit.json")
    side_path = args.poses_sideline or (P / "poses_sideline.json")
    refit = pickle.load(open(refit_path, "rb"))["frames"] if refit_path.exists() else {}
    side_blob = pickle.load(open(side_path, "rb")) if side_path.exists() else None
    if not refit and side_blob is None:
        raise SetupError("no pose cache: need poses_refit.json (05f) or poses_sideline.json (05c)")
    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=1)
    faces = model.faces.astype(np.int64)

    team_of: dict[int, str] = {}
    ident_path = args.identity or (P / "identity_resolved.pkl")
    if ident_path.exists():
        merged = pickle.load(open(ident_path, "rb")).get("merged", {})
        team_of = {int(pid): p.team for pid, p in merged.items() if getattr(p, "team", None)}
    fitted: dict[int, dict] = {}
    app_dir = args.appearance or (P / "appearance")
    if app_dir.exists():
        for path in app_dir.glob("appearance_*.npz"):
            z = np.load(path)
            fitted[int(path.stem.split("_")[1])] = {k: np.asarray(z[k], np.float32)
                                                    for k in ("colour", "log_scale_mult", "opacity_logit")}

    ground = ground_positions(df, tracks)
    frames_all = sorted(ground)
    frames = frames_all[:: max(1, args.stride)]
    if args.limit:
        frames = frames[: args.limit]
    poses = poses_from_caches(refit, side_blob, tracks, model)
    all_bp = [v[0] for d in poses.values() for v in d.values() if v[3] == "fused"]
    all_betas = [v[2] for d in poses.values() for v in d.values() if v[3] == "fused"]
    default_pose = tlm.median_pose(all_bp) if all_bp else np.zeros((21, 3))
    default_betas = np.median(np.stack(all_betas), axis=0) if all_betas else np.zeros(10)
    tl = tlm.build_timeline(frames_all, ground, poses, default_pose=default_pose,
                            default_betas=default_betas)
    print(f"timeline: {len(frames_all)} frames, {len(poses)} posed players, "
          f"{tl.n_default} default-posed, {tl.n_clamped} tilt-clamped states; "
          f"appearance for {len(fitted)} players, teams for {len(team_of)}")

    # Field once; virtual camera on the players' centroid over the play.
    f_xyz, f_rot, f_scale, f_opac, f_dc = texture_to_gaussians(
        render_field_texture(res_m=args.field_res_m), args.field_res_m)
    field = batch_from_arrays(f_xyz, f_rot, f_scale, f_opac, np.asarray(f_dc, np.float32)[:, :, None])
    centre = np.median(np.concatenate([np.stack([s.xy for s in tl.states[f]])
                                       for f in frames if f in tl.states]), axis=0)
    target = np.array([centre[0], centre[1], 1.0])
    eye = target + np.array([2.0, -34.0, 13.0])
    R_v, t_v = look_at(eye, target)
    K_v = intrinsics(args.width, args.height, fov_deg=55.0)
    print(f"camera on ({centre[0]:.1f}, {centre[1]:.1f}) m")

    def body_batch(s: tlm.PlayerState):
        with torch.no_grad():
            res = model(betas=torch.tensor(s.betas[None, :model.num_betas].astype(np.float32)),
                        body_pose=torch.tensor(s.body_pose.reshape(1, -1).astype(np.float32)),
                        global_orient=torch.tensor(s.global_orient.reshape(1, 3).astype(np.float32)))
        verts = res.vertices[0].numpy().astype(np.float64)
        verts = verts + np.array([s.xy[0], s.xy[1], -verts[:, 2].min()])   # feet on the turf
        # Fitted COLOUR only: the fit's scale and opacity were tuned to blurry
        # 140-px crops and read as translucent bodies at this distance; the
        # colours were the acceptance-tested part.
        colour = (fitted[s.pid]["colour"] if s.pid in fitted
                  else TEAM_RGB.get(team_of.get(s.pid, ""), BODY_RGB))
        return mesh_to_gaussians(verts, faces, colour=colour)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = []
    for i, f in enumerate(frames):
        out = args.out_dir / f"frame_{f:05d}.png"
        if args.resume and out.exists():
            written.append(out)                      # a stalled or interrupted run resumes here
            continue
        states = tl.states.get(f, [])
        scene = merge([field] + [body_batch(s) for s in states])
        sp = st.SceneParams.from_batch(scene, device=args.device)
        with torch.no_grad():
            img = st.render(sp, K_v, R_v, t_v, crop=(0, 0, args.width, args.height),
                            background=(0.06, 0.06, 0.08))
        imageio.imwrite(out, (255 * img.clamp(0, 1).cpu().numpy()).astype(np.uint8))
        written.append(out)
        if i % 20 == 0:
            _LOG.info("rendered %d/%d (%d bodies, %.2f s/frame)", i + 1, len(frames), len(states),
                      (time.time() - t0) / (i + 1))
    mp4 = args.out_dir / "play.mp4"
    with imageio.get_writer(mp4, fps=args.fps / max(1, args.stride), codec="libx264",
                            quality=8, macro_block_size=None) as w:
        for p in written:
            w.append_data(imageio.imread(p))
    print(f"wrote {len(written)} frames and {mp4} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
