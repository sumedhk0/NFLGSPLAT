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
from nfl_gsplat.render.blind_axis import hold_blind_axis
from nfl_gsplat.render.depth_snap import snap_ground
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
# Where a view has no confident ankles the box point stands in, and the two disagree by that 0.31 m
# (1.2 m on a crouched man), so every switch between them was a hop and every ankle-less frame carried
# the bias. anchor_boxes_to_ankles moves a box point by the id's own median (ankle - box) offset over
# the ankle frames within this window, needing this many of them. Measured on play 1 (2026-09-15):
# 2462 box frames moved, offset p50 0.31 m / p90 0.73; live steps > 0.25 m/frame 21 -> 19, whole clip
# 214 -> 196, census 2.00 -> 1.94, root jitter p90 0.0294 -> 0.0286, the endzone reprojection of the
# lower joints unchanged (8.8 / 15.2 px) -- better on every ruler, worse on none.
ANKLE_ANCHOR_WINDOW: int = 15
ANKLE_ANCHOR_MIN_SUPPORT: int = 3


def clip_offset(play_dir) -> int:
    """05o's endzone clip offset (sideline f beside endzone f + offset), 0 without the file."""
    import json

    f = Path(play_dir) / "clip_offset.json"
    return int(json.loads(f.read_text())["offset"]) if f.exists() else 0


def _named(play_dir) -> dict:
    """pid -> True when identity_resolved.pkl gives the id a jersey number or a role."""
    import pickle

    f = Path(play_dir) / "identity_resolved.pkl"
    if not f.exists():
        return {}
    blob = pickle.load(open(f, "rb"))
    merged = blob.get("merged", {}) or {}
    roles = blob.get("roles", {}) or {}
    out = {}
    for pid, m in merged.items():
        out[int(pid)] = bool(getattr(m, "jersey", 0)) or bool(roles.get(pid) or roles.get(int(pid)))
    return out


def _teams(play_dir):
    """``{pid: team}`` from identity_resolved.pkl, empty without it."""
    import pickle

    f = Path(play_dir) / "identity_resolved.pkl"
    if not f.exists():
        return {}
    try:
        merged = pickle.load(open(f, "rb")).get("merged", {})
    except Exception:                                    # noqa: BLE001
        return {}
    return {int(k): getattr(v, "team", None) for k, v in merged.items()}


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


def box_ground(df, tracks, *, margin_frac: float = BOX_MARGIN_FRAC, frame_shift=None) -> dict:
    """``{(cam, frame, pid): xy}``: every row's box-bottom point on the turf -- ground_positions'
    fallback, keyed like ankle_ground so the two can be joined per (camera, frame, player)."""
    from nfl_gsplat.pose.place_on_field import ground_point

    out: dict = {}
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
                try:
                    foot_v = float(r.bbox_y2) - margin_frac * float(r.bbox_y2 - r.bbox_y1)
                    g = ground_point((0.5 * (r.bbox_x1 + r.bbox_x2), foot_v), K, R, t)
                except Exception:
                    continue
                if abs(g[0]) < 60 and abs(g[1]) < 30:
                    out[(str(cam), f, int(r.global_player_id))] = np.asarray(g[:2], float)
    return out


def anchor_boxes_to_ankles(box_pts: dict, ankle_pts: dict, *, window: int = ANKLE_ANCHOR_WINDOW,
                           min_support: int = ANKLE_ANCHOR_MIN_SUPPORT):
    """``(points, n_anchored)``: ``ankle_pts`` plus, for every ``box_pts`` key without an ankle point,
    the box point moved by that (camera, player)'s median ``ankle - box`` offset over its ankle frames
    within ``window`` frames, when at least ``min_support`` of them exist. Keys with no ankle frame
    near are left out, so ground_positions falls back to the raw box there as before.

    WHY. The ankle ray is the better foot (0.06 m vs the box's 0.31 m against the triangulated
    ankles), but the detector's ankles are confident on some frames and not others, and a body whose
    ground point alternates between the two sources hops by their disagreement -- 1.2 m on play 1's
    id 161 at frame 368 -> 369, one of the live-play steps that survived every teleport fix. The
    offset is a property of the box (how far the detector's bottom edge sits from the ankles for this
    man's stance), so it varies slowly and the id's own recent frames measure it."""
    by_id: dict = {}
    for key, b in box_pts.items():
        cam, f, pid = key
        a = ankle_pts.get(key)
        by_id.setdefault((cam, int(pid)), {})[int(f)] = (np.asarray(b, float), None if a is None else np.asarray(a, float) - b)
    out = dict(ankle_pts)
    n_anchored = 0
    for (cam, pid), byf in by_id.items():
        anks = sorted((f, off) for f, (_b, off) in byf.items() if off is not None)
        if not anks:
            continue
        fs = np.asarray([f for f, _ in anks])
        offs = np.stack([off for _, off in anks])
        for f, (b, off) in byf.items():
            if off is not None:
                continue
            lo, hi = np.searchsorted(fs, f - window), np.searchsorted(fs, f + window, side="right")
            if hi - lo < min_support:
                continue
            out[(cam, f, pid)] = b + np.median(offs[lo:hi], axis=0)
            n_anchored += 1
    return out, n_anchored


def ground_positions(df, tracks, *, with_views: bool = False, margin_frac: float = BOX_MARGIN_FRAC, ankles=None,
                     frame_shift=None):
    """frame -> {pid: xy}: each view's foot (the box bottom less the detector's
    margin below the shoe, or the ankle keypoints' ground point from ``ankles``
    -- ankle_ground -- where the view has them) through its camera, both
    averaged. With ``with_views``, also frame -> {pid: (views seen,)}.
    ``frame_shift`` ``{cam: n}``: that camera's rows were moved by n from the
    clip's frames (the camera pose is the clip frame's)."""
    from nfl_gsplat.pose.place_on_field import ground_point

    if "global_player_id" not in df.columns:
        raise SetupError(
            "ground_positions keys by global_player_id and this table has none. The tracker's id is NOT "
            "the key -- the two are equal only until a track is relabelled onto another player (08s), "
            "and falling back to track_id silently would put a relabelled body under an id no caller "
            "asks for.")
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
                # The key is the PLAYER, not this camera's tracker id. The two have always been equal
                # in this pipeline, so they were used interchangeably until 08s relabelled an endzone
                # track onto another player's global id -- after which those ids reported NO ground
                # points at all (id 11: 423 endzone rows, 405 keypoint boxes, 0 ground frames, against
                # control id 12 at 381/377/381), because their points landed under the old track id and
                # the ankle lookup missed every time: ankle_ground keys by global_player_id. Every
                # caller asks by player id -- 05p tests `pid not in ground.get(f, {})` with pid from the
                # keypoints, the depth snap matches endzone candidates the same way, the census counts
                # players -- so the docstring's "{pid: xy}" was the contract and track_id broke it.
                pid = int(r.global_player_id)
                g = None if ankles is None else ankles.get((str(cam), f, pid))
                if g is not None:
                    out.setdefault(f, {}).setdefault(pid, []).append(np.asarray(g[:2], float))
                    seen.setdefault(f, {}).setdefault(pid, []).append(str(cam))
                    continue
                try:
                    foot_v = float(r.bbox_y2) - margin_frac * float(r.bbox_y2 - r.bbox_y1)
                    g = ground_point((0.5 * (r.bbox_x1 + r.bbox_x2), foot_v), K, R, t)
                except Exception:
                    continue
                if abs(g[0]) < 60 and abs(g[1]) < 30:
                    out.setdefault(f, {}).setdefault(pid, []).append(np.asarray(g[:2], float))
                    seen.setdefault(f, {}).setdefault(pid, []).append(str(cam))
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


def play_snap(play_dir) -> int | None:
    """The snap frame from ``<play-dir>/play_end.json`` (08x), or None: before it nobody moves, which
    the span rule uses to tell an endzone-placed ghost from a set man."""
    import json

    f = Path(play_dir) / "play_end.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return int(d["snap"]) if d.get("snap") is not None else None


def load_play_timeline(play_dir: Path, model, *, poses_refit=None, poses_sideline=None,
                       stitch_ids: bool = False, place_from_refit_transl: bool = True,
                       no_depth_snap: bool = False, blind_axis: bool = False, span_gap: int | None = None,
                       span_hold_m: float | None = None, span_presnap: str | None = None,
                       hole_hold_m: float | None = -1.0, box_twin_iou: float | None = -1.0,
                       despike_m: float | None = -1.0, weak_kit_margin: float | None = -1.0):
    """``(timeline, tracks, df, frames_all, poses)`` for a play-dir. With
    ``stitch_ids`` the linker's fragments are joined by tracking.stitch
    (position and speed, in field metres) and every state carries the
    player id; the timeline's ``members`` maps it back to the fragments. ``span_gap`` is how many
    frames beyond its sideline span a two-view id is still drawn from the endzone alone
    (default timeline.MAX_GAP_FRAMES; a probe knob, see endzone_only_rule.beyond_sideline_span);
    ``span_hold_m`` drops those frames when they stand farther than this from the sideline's own
    point for the man (default endzone_only_rule.HOLD_M)."""
    import pandas as pd

    P = Path(play_dir)
    if hole_hold_m is not None and hole_hold_m < 0:
        from nfl_gsplat.render import endzone_only_rule as _ezr

        hole_hold_m = _ezr.HOLE_HOLD_M
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
        n_ankle = len(ankles)
        # ... and the box points near them carry the id's own ankle-box offset, so a body does not
        # hop when the detector's ankles come and go (ANKLE_ANCHOR_WINDOW)
        ankles, n_anchored = anchor_boxes_to_ankles(box_ground(df, tracks, frame_shift=shift), ankles)
        print(f"ground from the ankle keypoints on {n_ankle} (camera, frame, id); {n_anchored} box points "
              f"anchored to them; the raw box point elsewhere")
    # A pair whose two tracks are not one player: the sideline alone draws it
    # (pair_rule; the pairing's median-distance gate catches it upstream now).
    if {"sideline", "endzone"} <= set(df["cam"].unique()):
        bad = mispaired_ids(ground_positions(df[df["cam"] == "sideline"], tracks, ankles=ankles, frame_shift=shift),
                            ground_positions(df[df["cam"] == "endzone"], tracks, ankles=ankles, frame_shift=shift))
        if bad:
            print("mispaired ids drawn from the sideline alone: "
                  + ", ".join(f"{pid} ({d:.1f} m)" for pid, d in sorted(bad.items())))
            # bad holds PLAYER ids, because ground_positions keys by global_player_id; dropping the
            # rows by the tracker's id instead would miss exactly the relabelled tracks
            df = df[~((df["cam"] == "endzone") & df["global_player_id"].isin(list(bad)))]
    ground, views = ground_positions(df, tracks, with_views=True, ankles=ankles, frame_shift=shift)
    # The sideline's point places a body wherever the sideline sees it: the
    # two-camera average carried the endzone's depth error (1.9 m on id 2 of
    # play 1) and refused 9 % of the one-view records' placements against it;
    # the endzone's point stands only where the sideline has none.
    if "sideline" in tracks:
        side_ground = ground_positions(df[df["cam"] == "sideline"], tracks, ankles=ankles, frame_shift=shift)
        # ... and the sideline cannot see along its OWN line of sight, so a body slides on that
        # ray onto the endzone's detection of it where the two agree (render.depth_snap).
        if "endzone" in tracks and not no_depth_snap:
            end_ground = ground_positions(df[df["cam"] == "endzone"], tracks, ankles=ankles, frame_shift=shift)
            teams_of = _teams(P)
            side_ground, n_snap = snap_ground(side_ground, end_ground, tracks["sideline"], teams=teams_of)
            print(f"depth from the endzone on {n_snap} body-frames (the sideline's own line of sight)")
        for f, d in side_ground.items():
            for pid, xy in d.items():
                ground.setdefault(f, {})[pid] = xy
        # a short hole in a sideline track filled from the endzone follows the sideline's own line
        # through it when the endzone's point is a stride off (endzone_only_rule.hold_holes)
        from nfl_gsplat.render.endzone_only_rule import hold_holes

        ground, hole_moves = hold_holes(ground, side_ground, hold_m=hole_hold_m)
        if hole_moves:
            held = [m for m in hole_moves if np.isfinite(m)]
            print(f"endzone-filled hole frames held to the sideline's line: {len(held)} "
                  f"(median {np.median(held) if held else float('nan'):.2f} m off it, max {max(held) if held else float('nan'):.2f}); "
                  f"beyond reach of both ends, left out: {len(hole_moves) - len(held)}")
    # A paired id lives on its sideline span: beyond it the endzone track
    # alone draws a second copy of a player (endzone_only_rule).
    if "sideline" in tracks:
        # side_ground keeps an id beyond its span where the sideline has no body within SAME_BODY_M:
        # the sideline lost him, the endzone did not. This was measured on 2026-09-11 and REJECTED
        # (census 2.93 -> 3.01), and that rejection was wrong twice over (2026-09-13):
        #   - it was scored from the snap to the END OF CLIP, ~200 frames of which are the post-whistle
        #     crowd; on the play itself the same change is +0.00, not +0.08;
        #   - and +0.00 is not "no effect". It restores 751 body-frames, 466 of them Kansas City, which
        #     then die one rule later: dedupe_frames' one-view box kills every one (id 74 by id 1 on
        #     65 of 65 frames, id 38 by id 4, id 22 by id 13; n_duplicates 500 -> 1585). The two rules
        #     delete the same men, so fixing EITHER alone measures nothing. Fixed together, Kansas City
        #     at the snap goes 8.9 -> 10.8 and frames exactly eleven-a-side 0 -> 23.
        # The men are real, on two rulers that do not involve the census: the cross-camera control has
        # paired ids agreeing at p50 0.42 m, while ids 74 and 38 stand 1.24-1.44 m from the nearest
        # sideline body with |dy| ~1.19 m -- across the field, where linemen separate, not along the
        # endzone's blind depth axis -- and the footage shows the merged box (one ground point over two
        # to three players). See ONE_VIEW_ACROSS_M in render.timeline for the other half.
        from nfl_gsplat.render import endzone_only_rule as ezr

        span_report: dict = {}
        snap = play_snap(P)
        ground, n_beyond = beyond_sideline_span(ground, df, tracks["sideline"],
                                                gap=tlm.MAX_GAP_FRAMES if span_gap is None else int(span_gap),
                                                side_ground=side_ground,
                                                hold_m=ezr.HOLD_M if span_hold_m is None else float(span_hold_m),
                                                report=span_report, snap=snap,
                                                presnap=ezr.PRESNAP if span_presnap is None else span_presnap)
        if n_beyond:
            print(f"frames beyond an id's sideline span left out: {n_beyond}")
        if span_report:
            print("beyond-span stretches drawn from the endzone (id: frames, worst m from the sideline's join point, dropped as far): "
                  + ", ".join(f"{p}: {r[0]}, {r[1]:.2f}, {r[2]}" for p, r in sorted(span_report.items())))
        # A body the endzone alone sees stands on the endzone's foot point, blind along the field
        # (render.blind_axis): its x from the id's nearest sideline sightings, sliding the point
        # along the endzone's own ray. Measured on play 1 and NOT adopted (live steps 5 -> 7, census
        # 1.50 -> 1.60; the numbers are in the module); opt-in.
        if blind_axis and "endzone" in tracks:
            ground, moved = hold_blind_axis(ground, views, tracks["endzone"], frame_shift=offset or 0)
            if moved:
                print(f"endzone-only frames held on the sideline's x: {len(moved)} body-frames "
                      f"(median slide {np.median(moved):.2f} m, max {max(moved):.2f})")
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
    lying = tlm.lying_frames(df)
    if lying:
        print(f"on the ground (sideline box wider than {tlm.LYING_ASPECT:.1f} of its height): {len(lying)} body-frames, "
              f"ids {sorted({p for _f, p in lying})[:12]}")
    tl = tlm.build_timeline(frames_all, ground, poses, default_pose=default_pose,
                            default_betas=default_betas, views_by_frame=views,
                            exclude=clipped if not stitch_ids else None, lying=lying,
                            despike_m=tlm.DESPIKE_M if (despike_m is not None and despike_m < 0) else despike_m)
    tl.members = members
    # a short fragment riding another body of its team is that body's second copy (timeline.rider_ids)
    teams_now = _teams(P)
    riders = tlm.rider_ids(tl, teams_now)
    if riders:
        n = tlm.drop_ids(tl, riders)
        print(f"short fragments riding another body left out: {sorted(riders)} ({n} body-frames)")
    # a short fragment with no team is a few detections of a man another id draws (timeline.orphan_ids)
    orphans = tlm.orphan_ids(tl, teams_now)
    if orphans:
        n = tlm.drop_ids(tl, orphans)
        print(f"short teamless fragments left out: {sorted(orphans)} ({n} body-frames)")
    # a short fragment whose kit could not be read and that identity never named wears a guessed
    # team (timeline.unreadable_kit_ids)
    if weak_kit_margin is not None and weak_kit_margin < 0:
        weak_kit_margin = tlm.WEAK_KIT_MARGIN
    if weak_kit_margin is not None:
        weak = tlm.unreadable_kit_ids(tl, df, _named(P), margin=float(weak_kit_margin))
        if weak:
            n = tlm.drop_ids(tl, weak)
            print(f"short fragments with an unreadable kit left out: {sorted(weak)} ({n} body-frames)")
    # two same-team ids within TWIN_M for TWIN_MIN_RUN frames are one man: the shorter-lived loses the
    # stretch (timeline.twin_frames; play 1: census live 1.32 -> 1.20, hops and steps unchanged)
    twins = tlm.twin_frames(tl, teams_now)
    if twins:
        n = tlm.drop_frames(tl, twins)
        by = {}
        for f, pid in twins:
            by.setdefault(pid, []).append(f)
        print(f"twin stretches left out: {n} body-frames -- " + ", ".join(f"{p} {min(v)}-{max(v)}" for p, v in sorted(by.items())))
    # the sideline's own second id on one man (timeline.box_twin_frames): same-team ids whose sideline
    # boxes coincide; measured 2026-09-17 on the live window before shipping
    if box_twin_iou is not None and box_twin_iou < 0:
        box_twin_iou = tlm.BOX_TWIN_IOU
    if box_twin_iou is not None:
        btw = tlm.box_twin_frames(tl, df, teams_now, iou_min=float(box_twin_iou))
        if btw:
            n = tlm.drop_frames(tl, btw)
            by = {}
            for f, pid in btw:
                by.setdefault(pid, []).append(f)
            print(f"box twins left out: {n} body-frames -- " + ", ".join(f"{pid} {min(fs)}-{max(fs)}" for pid, fs in sorted(by.items())))
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
