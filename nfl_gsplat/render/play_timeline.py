"""Build a play's timeline from its play-dir (shared by 05k and 05i).

Everything here needs the play-dir's cameras, tracks and pose caches, and the
SMPL-X model for the single-view poses' world orientation; the pure parts
live in render.timeline.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.render import timeline as tlm


def ground_positions(df, tracks, *, with_views: bool = False):
    """frame -> {pid: xy}: each view's foot through its camera, both averaged.
    With ``with_views``, also frame -> {pid: (views seen,)}."""
    from nfl_gsplat.pose.place_on_field import ground_point

    out: dict[int, dict[int, list]] = {}
    seen: dict[int, dict[int, list]] = {}
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
                    seen.setdefault(f, {}).setdefault(int(r.track_id), []).append(str(cam))
    ground = {f: {pid: np.mean(v, axis=0) for pid, v in d.items()} for f, d in out.items()}
    if not with_views:
        return ground
    views = {f: {pid: tuple(sorted(set(v))) for pid, v in d.items()} for f, d in seen.items()}
    return ground, views


def poses_from_caches(refit, side_blob, tracks, model):
    """pid -> {frame: (body_pose, global_orient_world, betas, source)}."""
    import torch
    from scipy.spatial.transform import Rotation

    from nfl_gsplat.pose.place_on_field import placement_transform

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


def load_play_timeline(play_dir: Path, model, *, poses_refit=None, poses_sideline=None):
    """``(timeline, tracks, df, frames_all, poses)`` for a play-dir."""
    import pandas as pd

    P = Path(play_dir)
    tracks = load_camera_track(P / "cameras.npz")
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    refit_path = Path(poses_refit) if poses_refit else P / "poses_refit.json"
    side_path = Path(poses_sideline) if poses_sideline else P / "poses_sideline.json"
    refit = pickle.load(open(refit_path, "rb"))["frames"] if refit_path.exists() else {}
    side_blob = pickle.load(open(side_path, "rb")) if side_path.exists() else None
    if not refit and side_blob is None:
        raise SetupError("no pose cache: need poses_refit.json (05f) or poses_sideline.json (05c)")
    ground, views = ground_positions(df, tracks, with_views=True)
    frames_all = sorted(ground)
    poses = poses_from_caches(refit, side_blob, tracks, model)
    all_bp = [v[0] for d in poses.values() for v in d.values() if v[3] == "fused"]
    all_betas = [v[2] for d in poses.values() for v in d.values() if v[3] == "fused"]
    default_pose = tlm.median_pose(all_bp) if all_bp else np.zeros((21, 3))
    default_betas = np.median(np.stack(all_betas), axis=0) if all_betas else np.zeros(10)
    tl = tlm.build_timeline(frames_all, ground, poses, default_pose=default_pose,
                            default_betas=default_betas, views_by_frame=views)
    return tl, tracks, df, frames_all, poses


def placed_vertices(state: tlm.PlayerState, model):
    """World vertices of a state's body, feet on the turf under the pelvis."""
    import torch

    with torch.no_grad():
        res = model(betas=torch.tensor(state.betas[None, :model.num_betas].astype(np.float32)),
                    body_pose=torch.tensor(state.body_pose.reshape(1, -1).astype(np.float32)),
                    global_orient=torch.tensor(state.global_orient.reshape(1, 3).astype(np.float32)))
    verts = res.vertices[0].numpy().astype(np.float64)
    return verts + np.array([state.xy[0], state.xy[1], -verts[:, 2].min()])
