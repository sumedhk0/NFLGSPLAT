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
from nfl_gsplat.render.edge_rule import edge_clipped_ids
from nfl_gsplat.render.endzone_only_rule import beyond_sideline_span, endzone_only_ids
from nfl_gsplat.render.offfield_rule import behind_the_offence, sideline_dwellers, striped_ids
from nfl_gsplat.render.pair_rule import mispaired_ids
from nfl_gsplat.errors import SetupError
from nfl_gsplat.render import timeline as tlm


# The detector's box ends below the shoes: on play 1's sideline the box bottom
# sits 16.7 px below the lower ankle keypoint at the median (138 px boxes), of
# which 6 px is the ankle above the sole -- 11 px of margin, 0.078 of the box
# height, which put every body ~0.15 m toward the camera (the footage overlay,
# 2026-09-09: skeleton feet 15 px below the shoes on every player).
BOX_MARGIN_FRAC: float = 0.078
# The ankle keypoints' rays meeting the turf at ankle height are the better foot:
# against the two cameras' triangulated ankles on play 1 (2399 paired frames,
# 2026-09-09) the sideline's box point sits 0.31 m off at the median (0.54 pre-snap,
# the stances), its ankle ray 0.06 m (0.32 pre-snap); the endzone's box 0.44 m along
# its depth, its ankle ray 0.23 m.
ANKLE_Z_M: float = 0.08
ANKLE_MIN_CONF: float = 0.5


def clip_offset(play_dir) -> int:
    """05o's endzone clip offset (sideline f beside endzone f + offset), 0 without the file."""
    import json

    f = Path(play_dir) / "clip_offset.json"
    return int(json.loads(f.read_text())["offset"]) if f.exists() else 0


def _line_of_scrimmage(play_dir):
    """08n's line of scrimmage block from identity_resolved.pkl, or None when it has not run."""
    import pickle

    f = Path(play_dir) / "identity_resolved.pkl"
    if not f.exists():
        return None
    try:
        return pickle.load(open(f, "rb")).get("line_of_scrimmage")
    except Exception:                                    # noqa: BLE001 - a cache without the block
        return None


def ankle_ground(kdf, tracks, *, z: float = ANKLE_Z_M, min_conf: float = ANKLE_MIN_CONF, frame_shift=None) -> dict:
    """``{(cam, frame, pid): xy}``: the mean of a player's confident ankle keypoints (COCO 15,
    16) carried along their rays to the plane z = ``z``, per camera. ``frame_shift``
    ``{cam: n}`` says a table's frames were moved by n from the clip's (the camera pose is
    the clip frame's)."""
    import numpy as np

    k = kdf[(kdf["joint"].isin([15, 16])) & (kdf["conf"] >= min_conf)]
    out: dict = {}
    for (cam, f), g in k.groupby(["cam", "frame"]):
        cam, f = str(cam), int(f)
        if cam not in tracks:
            continue
        tr = tracks[cam]
        fc = f + (frame_shift or {}).get(cam, 0)
        if fc < 0 or fc >= len(tr.conf) or tr.conf[fc] <= 0:
            continue
        K, R, t = tr.K[fc], tr.R[fc], tr.t[fc]
        C = -R.T @ t
        Kinv = np.linalg.inv(K)
        for pid, gg in g.groupby("global_player_id"):
            uv = gg[["x", "y"]].to_numpy(float)
            d = (R.T @ (Kinv @ np.c_[uv, np.ones(len(uv))].T)).T
            ok = np.abs(d[:, 2]) > 1e-9
            if not ok.any():
                continue
            sc = (z - C[2]) / d[ok, 2]
            if (sc <= 0).any():
                continue
            pts = C[None, :2] + sc[:, None] * d[ok, :2]
            out[(cam, f, int(pid))] = pts.mean(axis=0)
    return out


def ground_positions(df, tracks, *, with_views: bool = False, margin_frac: float = BOX_MARGIN_FRAC, ankles=None,
                     frame_shift=None):
    """frame -> {pid: xy}: each view's foot (the box bottom less the detector's
    margin below the shoe, or the ankle keypoints' ground point from ``ankles``
    -- ankle_ground -- where the view has them) through its camera, both
    averaged. With ``with_views``, also frame -> {pid: (views seen,)}.
    ``frame_shift`` ``{cam: n}``: that camera's rows were moved by n from the
    clip's frames (the camera pose is the clip frame's)."""
    from nfl_gsplat.pose.place_on_field import ground_point

    out: dict[int, dict[int, list]] = {}
    seen: dict[int, dict[int, list]] = {}
    for cam, sub in df.groupby("cam"):
        tr = tracks[cam]
        shift = (frame_shift or {}).get(str(cam), 0)
        for f, rows in sub.groupby("frame"):
            f = int(f)
            fc = f + shift
            if fc < 0 or fc >= len(tr.conf) or tr.conf[fc] <= 0:
                continue
            intr, pose = tr.at(fc)
            K, R, t = intr.K(), pose.R, pose.t
            for r in rows.itertuples():
                g = None if ankles is None else ankles.get((str(cam), f, int(r.track_id)))
                if g is not None:
                    out.setdefault(f, {}).setdefault(int(r.track_id), []).append(np.asarray(g[:2], float))
                    seen.setdefault(f, {}).setdefault(int(r.track_id), []).append(str(cam))
                    continue
                try:
                    foot_v = float(r.bbox_y2) - margin_frac * float(r.bbox_y2 - r.bbox_y1)
                    g = ground_point((0.5 * (r.bbox_x1 + r.bbox_x2), foot_v), K, R, t)
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


# A record's pelvis further than this from the box-bottom point is the refit
# being wrong, not the box (2026-09-09, the footage overlay: refit-placed
# bodies 1.5-3 m toward the camera, feet 50-76 px below the real ones; the box
# point sits 0.52 m from a right triangulated pelvis at the median, 1.29 p90).
MAX_REFIT_SHIFT_M = 1.0


def place_from_refit(ground, refit, *, max_shift_m: float = MAX_REFIT_SHIFT_M, max_gap: int = 12, pelvis_xy=None):
    """``ground`` with every (frame, id) that has a refit record moved to the
    record's pelvis. ``pelvis_xy(rec) -> xy`` gives the record's pelvis on the
    field; without it the translation alone is used (the model's origin, which
    sits 0.35 m from the pelvis along the rest skeleton's down axis -- see
    placed_vertices). A shift beyond ``max_shift_m`` is a wrong record and is
    not applied. Returns ``(ground, shifts)``, ``shifts`` the metres moved."""
    out = {f: dict(d) for f, d in ground.items()}
    shifts = []
    accepted: set = set()
    for f, recs in refit.items():
        f = int(f)
        if f not in out:
            continue
        for pid, r in recs.items():
            pid = int(pid)
            if pid not in out[f]:
                continue
            xy = np.asarray(r["transl"], float)[:2] if pelvis_xy is None else np.asarray(pelvis_xy(r), float)[:2]
            d = float(np.hypot(*(xy - np.asarray(out[f][pid], float))))
            if np.isfinite(d) and d <= max_shift_m:
                out[f][pid] = xy
                shifts.append(d)
                accepted.add((f, pid))
    # Across a short gap in a player's records the translation is interpolated:
    # the box-bottom point in between sat ~0.5 m from the refit's pelvis and the
    # body dipped there and back (play 1: 132 gaps, p50 3 frames).
    by_pid: dict[int, list] = {}
    for f, pid in accepted:                                  # only between records that were placed
        by_pid.setdefault(pid, []).append(f)
    for pid, fs in by_pid.items():
        fs = sorted(fs)
        for a, b in zip(fs, fs[1:]):
            if 1 < b - a <= max_gap + 1:
                ra, rb = refit[a][pid], refit[b][pid]
                xa = np.asarray(ra["transl"], float)[:2] if pelvis_xy is None else np.asarray(pelvis_xy(ra), float)[:2]
                xb = np.asarray(rb["transl"], float)[:2] if pelvis_xy is None else np.asarray(pelvis_xy(rb), float)[:2]
                for f in range(a + 1, b):
                    if f in out and pid in out[f]:
                        w = (f - a) / float(b - a)
                        out[f][pid] = (1 - w) * xa + w * xb
    return out, np.asarray(shifts)


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


def load_play_timeline(play_dir: Path, model, *, poses_refit=None, poses_sideline=None,
                       stitch_ids: bool = False, place_from_refit_transl: bool = True):
    """``(timeline, tracks, df, frames_all, poses)`` for a play-dir. With
    ``stitch_ids`` the linker's fragments are joined by tracking.stitch
    (position and speed, in field metres) and every state carries the
    player id; the timeline's ``members`` maps it back to the fragments."""
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
    # The endzone clip's frames sit beside sideline frame f at f + offset (05o); the
    # timeline runs on sideline frames, so the endzone rows move to the sideline frame
    # they belong to (play 1: -15 frames -- an endzone-only body was drawn 0.25 s early
    # and a moving pair looked 2 m apart to the pair rule).
    offset = clip_offset(P)
    shift = {"endzone": offset} if offset else None
    if offset and "endzone" in set(df["cam"].unique()):
        df = df.copy()
        df.loc[df["cam"] == "endzone", "frame"] = df.loc[df["cam"] == "endzone", "frame"].astype(int) - offset
        print(f"endzone rows moved to their sideline frames (clip offset {offset:+d})")
    ankles = None
    if (P / "keypoints_2d.parquet").exists():
        kdf = pd.read_parquet(P / "keypoints_2d.parquet")
        if offset:
            kdf = kdf.copy()
            kdf.loc[kdf["cam"] == "endzone", "frame"] = kdf.loc[kdf["cam"] == "endzone", "frame"].astype(int) - offset
        ankles = ankle_ground(kdf, tracks, frame_shift=shift)
        print(f"ground from the ankle keypoints on {len(ankles)} (camera, frame, id); the box point elsewhere")
    # A pair whose two tracks are not one player: the sideline alone draws it
    # (pair_rule; the pairing's median-distance gate catches it upstream now).
    if {"sideline", "endzone"} <= set(df["cam"].unique()):
        bad = mispaired_ids(ground_positions(df[df["cam"] == "sideline"], tracks, ankles=ankles, frame_shift=shift),
                            ground_positions(df[df["cam"] == "endzone"], tracks, ankles=ankles, frame_shift=shift))
        if bad:
            print("mispaired ids drawn from the sideline alone: "
                  + ", ".join(f"{pid} ({d:.1f} m)" for pid, d in sorted(bad.items())))
            df = df[~((df["cam"] == "endzone") & df["track_id"].isin(list(bad)))]
    ground, views = ground_positions(df, tracks, with_views=True, ankles=ankles, frame_shift=shift)
    # The sideline's point places a body wherever the sideline sees it: the
    # two-camera average carried the endzone's depth error (1.9 m on id 2 of
    # play 1) and refused 9 % of the one-view records' placements against it;
    # the endzone's point stands only where the sideline has none.
    if "sideline" in tracks:
        for f, d in ground_positions(df[df["cam"] == "sideline"], tracks, ankles=ankles, frame_shift=shift).items():
            for pid, xy in d.items():
                ground.setdefault(f, {})[pid] = xy
    # A paired id lives on its sideline span: beyond it the endzone track
    # alone draws a second copy of a player (endzone_only_rule).
    if "sideline" in tracks:
        ground, n_beyond = beyond_sideline_span(ground, df, tracks["sideline"], gap=tlm.MAX_GAP_FRAMES)
        if n_beyond:
            print(f"frames beyond an id's sideline span left out: {n_beyond}")
    if place_from_refit_transl and refit:
        ground, shifts = place_from_refit(ground, refit, pelvis_xy=_pelvis_xy_fn(model))
        if len(shifts):
            print(f"placement from the refit for {len(shifts)} body-frames (median shift "
                  f"{np.median(shifts):.2f} m from the box-bottom point)")
    frames_all = sorted(ground)
    clipped = edge_clipped_ids(df, tracks, views)
    if clipped:
        print(f"edge-clipped one-view ids left out: {len(clipped)}")
    # One avatar per sideline track: an id the endzone alone sees is the
    # sideline's player unpaired, drawn twice (endzone_only_rule).
    ghosts = endzone_only_ids(df, views, ground=ground, sideline=tracks.get("sideline"))
    if ghosts:
        print(f"endzone-only ids left out: {len(ghosts)}")
    # Not players: staff at the boundary (position) and officials (stripes),
    # offfield_rule; measured on play 1 against the roster-named ids.
    dwellers = sideline_dwellers(ground)
    if dwellers:
        print(f"sideline dwellers left out: {len(dwellers)}")
    striped = set()
    side_video = P / "sideline.mp4"
    if side_video.exists():
        striped = striped_ids(df, side_video) - dwellers
        if striped:
            print(f"striped (officials) left out: {len(striped)}")
    # The officials the SIDELINE camera never sees: they stand behind the offence, where the
    # sideline's narrow lens does not look, so the endzone-only rule keeps them (it keeps a body
    # the sideline could not have seen, which is right for a wide receiver). Play 1 drew the
    # referee, the umpire and a marker 14.5 m behind the line, one of them named from the 83 on
    # his back. No offensive player lines up that deep.
    behind = set()
    ez_only = {int(p) for p, g in df[df["track_id"] >= 0].groupby("track_id") if set(g["cam"]) == {"endzone"}}
    los = _line_of_scrimmage(P)
    if los is not None and ez_only:
        behind = behind_the_offence(ground, los["x"], los["sign"], ids=ez_only)
        if behind:
            print(f"bodies behind the offence (officials) left out: {sorted(behind)}")
    clipped = set(clipped) | ghosts | dwellers | striped | behind
    poses = poses_from_caches(refit, side_blob, tracks, model)
    # Roster height is the one shape fact worth imposing: the regressor's
    # betas sit near neutral (1.72 m) and these players median 1.85 m.
    ident_path = P / "identity_resolved.pkl"
    heights = {}
    if ident_path.exists():
        from nfl_gsplat.render.roster_shape import (betas_for_height, betas_for_height_weight,
                                                    heights_from_identity, weights_from_identity)

        merged_ident = pickle.load(open(ident_path, "rb")).get("merged", {})
        heights = heights_from_identity(merged_ident)
        weights = weights_from_identity(merged_ident)
        n_adj = 0
        for pid, byf in poses.items():
            h = heights.get(int(pid))
            if h is None:
                continue
            cache: dict = {}
            kg = weights.get(int(pid))
            for f, rec in byf.items():
                key = tuple(np.round(np.asarray(rec[2], float), 3))
                if key not in cache:
                    cache[key] = (betas_for_height_weight(model, rec[2], h, kg) if kg
                                  else betas_for_height(model, rec[2], h))
                byf[f] = (rec[0], rec[1], cache[key], rec[3])
                n_adj += 1
        print(f"roster heights: {len(heights)} ids known ({len(weights)} with a weight), {n_adj} posed records set to them")
    members = {}
    if stitch_ids:
        from nfl_gsplat.tracking.stitch import stitch

        pos: dict[int, list] = {}
        for f in frames_all:
            for pid, xy in ground[f].items():
                pos.setdefault(int(pid), []).append((int(f), float(xy[0]), float(xy[1])))
        player_of = stitch(pos, fps=59.94)
        n_before = len(pos)
        ground, views, poses, members = tlm.relabel(ground, views, poses, player_of)
        print(f"stitch: {n_before} ids -> {len(members)} players")
    all_bp = [v[0] for d in poses.values() for v in d.values() if v[3] == "fused"]
    all_betas = [v[2] for d in poses.values() for v in d.values() if v[3] == "fused"]
    default_pose = tlm.median_pose(all_bp) if all_bp else np.zeros((21, 3))
    default_betas = np.median(np.stack(all_betas), axis=0) if all_betas else np.zeros(10)
    tl = tlm.build_timeline(frames_all, ground, poses, default_pose=default_pose,
                            default_betas=default_betas, views_by_frame=views,
                            exclude=clipped if not stitch_ids else None)
    tl.members = members
    return tl, tracks, df, frames_all, poses


def rest_pelvis_xy(model, betas):
    """The pelvis joint of the unposed body, model axes, xy: (0.003, -0.35) for
    the neutral shape. SMPL-X and pose.forward_kinematics both keep the
    pelvis at this point plus the translation whatever the orientation."""
    import torch

    with torch.no_grad():
        res = model(betas=torch.tensor(np.asarray(betas, np.float32)[None, :model.num_betas]))
    return res.joints[0, 0].numpy().astype(np.float64)[:2]


def _pelvis_xy_fn(model):
    cache: dict = {}

    def fn(rec):
        key = tuple(np.round(np.asarray(rec["betas"], float)[:10], 3))
        if key not in cache:
            cache[key] = rest_pelvis_xy(model, rec["betas"])
        return np.asarray(rec["transl"], float)[:2] + cache[key]

    return fn


def placed_vertices(state: tlm.PlayerState, model):
    """World vertices of a state's body: the PELVIS over ``state.xy``, feet on
    the turf. It used to put the model's origin at xy, which is 0.35 m from
    the pelvis along the rest skeleton's down axis (a world direction here,
    since the pelvis joint is not rotated by the orientation): every body
    without a refit record stood 0.35 m from its box-bottom point, and a
    body popped by that much at every record boundary (play 1, 2026-09-08)."""
    import torch

    with torch.no_grad():
        res = model(betas=torch.tensor(state.betas[None, :model.num_betas].astype(np.float32)),
                    body_pose=torch.tensor(state.body_pose.reshape(1, -1).astype(np.float32)),
                    global_orient=torch.tensor(state.global_orient.reshape(1, 3).astype(np.float32)))
    verts = res.vertices[0].numpy().astype(np.float64)
    pelvis = res.joints[0, 0].numpy().astype(np.float64)
    return verts + np.array([state.xy[0] - pelvis[0], state.xy[1] - pelvis[1], -verts[:, 2].min()])
