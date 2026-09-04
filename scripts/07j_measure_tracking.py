#!/usr/bin/env python
"""Measure tracking THROUGH TIME against the Helmet Assignment truth.

Every position number this project has reported is per frame. The tracker
(YOLO + BoT-SORT in tracking.detect_track) has been assumed to link those
frames into players and has never been measured. This does that, the way 07c
measured placement: on plays with real tracking and labelled helmets, with the
cameras fitted from the labelled helmets so that the cameras are not what is
being tested.

    per view, per play:
      purity       for each tracker track, the share of its detections that sit
                   nearest the SAME labelled player (within a gate). 1.0 means
                   the track never changed identity.
      switches     identity changes along a track (consecutive assigned
                   detections that name different players).
      fragments    tracker tracks per labelled player -- how many pieces a
                   player's run was broken into.
      coverage     share of labelled player-frames that any track covers.
      xy error     metres, assigned detections against tracking.

The gate is generous (1.5 m) on purpose: this measures LINKING, and a
detection 1.2 m from its player is still that player's detection. Placement
error is reported separately and is what 07c measures properly.

WHICH POINT IS PROJECTED. The helmet-set cameras put z = 0 at the HELMET
plane (from_helmets), so the point projected onto it must be the helmet --
the top-centre of the person box -- not the feet. Projecting feet onto the
helmet plane put every player ~1.7 m off, most outside the gate: the first
run reported coverage 0.13-0.20 and xy 0.86 m, and that was the harness.

Tracks are also passed through tracking.stitch, which joins pieces across
short gaps, and fragments/purity are reported again after it -- so the stitch
is measured, not assumed, like everything else.

THE GROUND-PLANE LINKER (tracking.link3d) is measured on the SAME detections:
BoT-SORT's track ids are thrown away, every detection is placed on the plane
through the labelled-helmet camera, and link3d links those placements in
metres. Same frames, same boxes, same metrics -- only the linking differs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nfl_gsplat.calibration.from_helmets import cameras_fixed_centre
from nfl_gsplat.calibration.joint_views import ground_points
from nfl_gsplat.data.align_video import helmet_boxes_by_frame
from nfl_gsplat.data.helmet_dataset import load_labels, load_tracking
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.identity.team_color import dominant_jersey_color, split_two_teams_balanced
from nfl_gsplat.tracking import link3d, reid
from nfl_gsplat.tracking.detect_track import TrackingConfig, detect_and_track

VIDEO_FPS = 59.94
WIDTH, HEIGHT = 1280, 720
GATE_M = 1.5
LINK_KW: dict = {}
REID_WEIGHTS = None
FEATURE_WEIGHT = 1.0


def assign_players(df_view, cams, track, offset, *, gate_m=GATE_M):
    """Per detection: nearest labelled player (or -1), its distance, ground xy."""
    player = np.full(len(df_view), -1, int)
    dist = np.full(len(df_view), np.nan)
    ground = np.full((len(df_view), 2), np.nan)
    frames = df_view["frame"].to_numpy()
    # Helmet = top-centre of the box; the cameras' plane is the helmet plane.
    feet = np.column_stack([(df_view["bbox_x1"].to_numpy(float)
                             + df_view["bbox_x2"].to_numpy(float)) / 2.0,
                            df_view["bbox_y1"].to_numpy(float)])
    for f in np.unique(frames):
        secs = offset + f / VIDEO_FPS
        if not track.covers(secs):
            continue
        truth = track.at(secs)                          # [P, 2] metres
        ok = np.isfinite(truth).all(1)
        if not ok.any():
            continue
        cam = cams[min(cams, key=lambda k: abs(k - f))]
        rows = np.flatnonzero(frames == f)
        g = ground_points(cam, feet[rows])
        ground[rows] = g
        for r, xy in zip(rows, g):
            if not np.isfinite(xy).all():
                continue
            d = np.linalg.norm(truth[ok] - xy, axis=1)
            j = int(np.argmin(d))
            if d[j] <= gate_m:
                player[r] = int(np.flatnonzero(ok)[j])
                dist[r] = float(d[j])
    return player, dist, ground


def detection_colours(df_view, video_path, *, max_per_frame=64):
    """``[N, 3]`` HSV torso colour per detection (NaN where unknown)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = df_view["frame"].to_numpy()
    boxes = df_view[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    colours = np.full((len(df_view), 3), np.nan)
    for f in np.unique(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        rows = np.flatnonzero(frames == f)[:max_per_frame]
        h, w = img.shape[:2]
        for r in rows:
            x1, y1, x2, y2 = boxes[r]
            # Torso band: the middle of the box, away from helmet and turf.
            bh = y2 - y1
            ya, yb = int(max(0, y1 + 0.25 * bh)), int(min(h, y1 + 0.6 * bh))
            xa, xb = int(max(0, x1)), int(min(w, x2))
            if yb - ya < 3 or xb - xa < 3:
                continue
            try:
                colours[r] = dominant_jersey_color(img[ya:yb, xa:xb])
            except Exception:                              # noqa: BLE001
                continue
    cap.release()
    return colours


def team_labels(df_view, video_path, *, max_per_frame=64, colours=None):
    """Per-detection team label from jersey colour: 0/1, or -1 when unknown.

    The one cue position-only linking cannot supply. Dominant colour of the
    torso crop per detection, then one balanced two-way split over the whole
    view -- the same split identity uses per track, here per detection so the
    linker can refuse to join a red box to a white track.
    """
    if colours is None:
        colours = detection_colours(df_view, video_path, max_per_frame=max_per_frame)
    ok = np.isfinite(colours).all(1)
    labels = np.full(len(df_view), -1, int)
    team_gap = float("nan")
    if ok.sum() >= 4:
        lab = split_two_teams_balanced(colours[ok]).astype(int)
        # A label only where the colour is CLEARLY one side. Forcing a label
        # on every detection (measured) made linking worse -- 43 -> 73 tracks
        # on one view -- because the colour flips frame to frame near the
        # boundary and the linker then refuses to join a player to himself.
        c = colours[ok]
        centres = np.stack([c[lab == k].mean(0) if (lab == k).any() else np.zeros(3)
                            for k in (0, 1)])
        d = np.linalg.norm(c[:, None] - centres[None], axis=2)      # [N, 2]
        margin = np.abs(d[:, 0] - d[:, 1]) / (d.sum(1) + 1e-9)       # 0..1
        lab[margin < 0.25] = -1
        labels[ok] = lab
        team_gap = float(np.linalg.norm(centres[0] - centres[1]))
    team_labels.last_team_gap = team_gap
    return labels


# A track shorter than this is not scored for purity: one assigned detection
# is trivially 100% pure, and BoT-SORT's 1-2 frame blips were inflating its
# number while the ground linker's own min_frames=3 had already removed the
# same cases from its side (review finding). Same gate for both now.
MIN_SCORED_FRAMES = 3


def track_metrics(df_view, player, track, offset, *, id_col="track_id"):
    """purity, switches, fragments, coverage from per-detection assignments.

    Every detection in ``df_view`` counts toward coverage; tracks with fewer
    than MIN_SCORED_FRAMES assigned detections (and id -1) are excluded from
    purity/switches/fragments only.
    """
    ids = df_view[id_col].to_numpy()
    frames = df_view["frame"].to_numpy()
    purities, switches = [], 0
    pieces = defaultdict(set)
    for tid in np.unique(ids):
        if tid < 0:
            continue
        rows = np.flatnonzero(ids == tid)
        rows = rows[np.argsort(frames[rows])]
        assigned = player[rows]
        assigned = assigned[assigned >= 0]
        if len(assigned) < MIN_SCORED_FRAMES:
            continue
        counts = np.bincount(assigned)
        purities.append(counts.max() / len(assigned))
        switches += int((np.diff(assigned) != 0).sum())
        for p in np.unique(assigned):
            pieces[int(p)].add(int(tid))
    n_players = track.n_players
    covered = defaultdict(set)
    for f, p in zip(frames, player):
        if p >= 0:
            covered[int(p)].add(int(f))
    labelled_frames = sorted(set(frames.tolist()))
    total = 0
    for f in labelled_frames:
        if track.covers(offset + f / VIDEO_FPS):
            total += n_players
    cover = sum(len(v) for v in covered.values()) / max(total, 1)
    return {
        "tracks": int(len(np.unique(ids[ids >= 0]))),
        "purity_median": float(np.median(purities)) if purities else float("nan"),
        "purity_p10": float(np.percentile(purities, 10)) if purities else float("nan"),
        "switches": int(switches),
        "fragments_per_player": (float(np.mean([len(v) for v in pieces.values()]))
                                 if pieces else float("nan")),
        "players_seen": int(len(pieces)),
        "coverage": float(min(cover, 1.0)),
    }


def measure_play(labels, track, game, play, offset, *, cfg, stride, root):
    index = {p: i for i, p in enumerate(track.players)}
    out = {}
    for view in ("Sideline", "Endzone"):
        name = f"{game}_{play:06d}_{view}.mp4"
        sub = labels[labels["video"] == name]
        if sub.empty:
            out[view] = {"failed": "no labels"}
            continue
        byf = helmet_boxes_by_frame(sub, index)
        byf = {f: v for i, (f, v) in enumerate(sorted(byf.items())) if i % stride == 0}
        byf = {f: v for f, v in byf.items() if track.covers(offset + f / VIDEO_FPS)}
        try:
            cams, _centre, mirrored = cameras_fixed_centre(
                byf, lambda f: track.at(offset + f / VIDEO_FPS), WIDTH, HEIGHT)
        except CalibrationError as exc:
            out[view] = {"failed": f"calibration: {str(exc)[:80]}"}
            continue
        if mirrored:
            out[view] = {"failed": "mirrored calibration"}
            continue
        df = detect_and_track(root / "video" / name, view, cfg)
        if df.empty:
            out[view] = {"failed": "tracker returned nothing"}
            continue
        player, dist, ground = assign_players(df, cams, track, offset)
        m = track_metrics(df, player, track, offset)
        m["xy_error_median_m"] = float(np.nanmedian(dist)) if np.isfinite(dist).any() else float("nan")
        m["detections"] = int(len(df))
        m["assigned_frac"] = float((player >= 0).mean())
        # After stitching across gaps.
        try:
            from nfl_gsplat.tracking.stitch import apply_to_tracks, stitch

            # stitch wants {track: [(frame, x, y), ...]} in field metres.
            ids = df["track_id"].to_numpy()
            frs = df["frame"].to_numpy()
            pos = defaultdict(list)
            for i in range(len(df)):
                if np.isfinite(ground[i]).all():
                    pos[int(ids[i])].append((int(frs[i]), float(ground[i, 0]),
                                             float(ground[i, 1])))
            player_of = stitch(dict(pos), fps=VIDEO_FPS)
            df2 = apply_to_tracks(df, player_of)
            m_s = track_metrics(df2, player, track, offset, id_col="player_track")
            m["after_stitch"] = {k: m_s[k] for k in
                                 ("tracks", "purity_median", "switches",
                                  "fragments_per_player")}
        except Exception as exc:                          # noqa: BLE001
            m["after_stitch"] = {"failed": str(exc)[:80]}
        # The ground-plane linker on the same detections, ids discarded --
        # once with position only, once with a team label per detection.
        frs = df["frame"].to_numpy()
        placements = {int(f): ground[frs == f] for f in np.unique(frs)}
        det_colours = detection_colours(df, root / "video" / name)
        det_labels_all = team_labels(df, root / "video" / name, colours=det_colours)
        team_gap = getattr(team_labels, "last_team_gap", float("nan"))
        det_labels = {int(f): det_labels_all[frs == f] for f in np.unique(frs)}
        m["team_label_frac"] = float((det_labels_all >= 0).mean())
        feats_all = reid.embed_detections(root / "video" / name, df, device=cfg.device,
                                          weights=REID_WEIGHTS)
        feats = {int(f): feats_all[frs == f] for f in np.unique(frs)}
        for tag, lab, fe, fw in (("ground_linker", None, None, 0.0),
                                 ("ground_linker_teams", det_labels, None, 0.0),
                                 ("ground_linker_reid", None, feats, FEATURE_WEIGHT)):
            tracks = link3d.link(placements, labels=lab, features=fe, fps=VIDEO_FPS,
                                 feature_weight=fw, **LINK_KW)
            linked = np.full(len(df), -1, int)
            ids_by_frame = link3d.assignments(tracks, placements)
            for f in placements:
                linked[np.flatnonzero(frs == f)] = ids_by_frame[f]
            df3 = df.copy()
            df3["ground_track"] = linked
            # Same population as the BoT-SORT number: every detection stays
            # in the frame set (unlinked ones as their own -1 "track", which
            # the length gate then drops from purity but not from coverage).
            m_g = track_metrics(df3, player, track, offset, id_col="ground_track")
            m[tag] = {k: m_g[k] for k in
                      ("tracks", "purity_median", "purity_p10", "switches",
                       "fragments_per_player", "coverage")}
            if tag == "ground_linker":
                # The position-only stitcher on the ground linker's own pieces,
                # in field metres: does joining fragments buy fewer pieces per
                # player without welding different players (purity)?
                pos = {}
                for f in placements:
                    pts = placements[f]
                    for k, tid in enumerate(ids_by_frame[f]):
                        if tid >= 0:
                            pos.setdefault(int(tid), []).append((int(f), float(pts[k][0]), float(pts[k][1])))
                # Per-fragment median jersey colour (HSV): the cue that per-
                # detection colour lacked -- one frame flips, a track's median
                # does not. Joins are refused across half the team separation.
                track_colour = {}
                for tid in pos:
                    rows_t = np.flatnonzero(linked == tid)
                    c = det_colours[rows_t]
                    c = c[np.isfinite(c).all(1)]
                    track_colour[int(tid)] = np.median(c, axis=0) if len(c) >= 3 else None
                variants = {"ground_linker_stitched": stitch(pos, fps=VIDEO_FPS)}
                if np.isfinite(team_gap):
                    variants["ground_linker_stitched_colour"] = stitch(
                        pos, fps=VIDEO_FPS, colours=track_colour,
                        colour_dist=lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b))),
                        max_colour_dist=0.5 * team_gap)
                for vname, player_of in variants.items():
                    df4 = df3.copy()
                    df4["ground_track"] = [player_of.get(int(t), int(t)) if t >= 0 else -1
                                           for t in linked]
                    m_s = track_metrics(df4, player, track, offset, id_col="ground_track")
                    m[vname] = {k: m_s[k] for k in
                                ("tracks", "purity_median", "purity_p10", "switches",
                                 "fragments_per_player", "coverage")}
        out[view] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/helmet"))
    ap.add_argument("--alignment", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gate-floor", type=float, default=link3d.GATE_FLOOR_M,
                    help="ground linker: metres a detection may sit from a "
                         "track's prediction at zero time gap")
    ap.add_argument("--max-gap", type=float, default=link3d.MAX_GAP_S,
                    help="ground linker: seconds unseen before a track retires")
    ap.add_argument("--reid-weights", type=Path, default=None,
                    help="football-trained embedding (scripts/07k); default ImageNet trunk")
    ap.add_argument("--feature-weight", type=float, default=1.0,
                    help="metres per unit cosine distance for the re-id run")
    args = ap.parse_args()
    global LINK_KW
    LINK_KW = {"gate_floor": args.gate_floor, "max_gap_s": args.max_gap}
    global REID_WEIGHTS, FEATURE_WEIGHT
    REID_WEIGHTS, FEATURE_WEIGHT = args.reid_weights, args.feature_weight

    align_path = args.alignment or (args.root / "alignment.json")
    out_path = args.out or (args.root / "tracking_accuracy.json")
    alignment = json.loads(align_path.read_text(encoding="utf-8"))
    labels = load_labels(args.root / "train_labels.csv")
    plays = load_tracking(args.root / "train_player_tracking.csv")
    usable = [r for r in alignment.values() if r["offset_s"] is not None]
    if args.limit:
        usable = usable[:args.limit]
    cfg = TrackingConfig(yolo_weights=args.weights, imgsz=args.imgsz,
                         device=args.device)
    print(f"{len(usable)} aligned plays; tracker {cfg.tracker} on "
          f"{cfg.yolo_weights} @ {cfg.imgsz}\n")
    results = {}
    for i, rec in enumerate(usable, 1):
        game, play = rec["game_key"], rec["play_id"]
        track = plays.get((game, play))
        if track is None:
            continue
        got = measure_play(labels, track, game, play, rec["offset_s"],
                           cfg=cfg, stride=args.stride, root=args.root)
        results[f"{game}_{play}"] = got
        for view, m in got.items():
            if "failed" in m:
                print(f"[{i}/{len(usable)}] {game}/{play} {view:8s} FAILED {m['failed']}",
                      flush=True)
                continue
            s = m.get("after_stitch", {})
            g = m.get("ground_linker", {})
            print(f"[{i}/{len(usable)}] {game}/{play} {view:8s} botsort: "
                  f"tracks {m['tracks']:3d} purity p10 {m['purity_p10']:.2f} "
                  f"switches {m['switches']:3d} frag/player "
                  f"{m['fragments_per_player']:.1f} coverage {m['coverage']:.2f} "
                  f"xy {m['xy_error_median_m']:.2f} m"
                  + (f" | stitched frag {s['fragments_per_player']:.1f}"
                     if "tracks" in s else ""), flush=True)
            pad = len(f"[{i}/{len(usable)}] {game}/{play} {view:8s}")
            for tag, label in (("ground_linker", "ground: "),
                               ("ground_linker_stitched", "+stitch: "),
                               ("ground_linker_stitched_colour", "+stitch+colour: "),
                               ("ground_linker_teams", "+teams: "),
                               ("ground_linker_reid", "+re-id: ")):
                g = m.get(tag, {})
                if "tracks" in g:
                    print(f"{'':>{pad}} {label} tracks {g['tracks']:3d} purity p10 "
                          f"{g['purity_p10']:.2f} switches {g['switches']:3d} "
                          f"frag/player {g['fragments_per_player']:.1f} coverage "
                          f"{g['coverage']:.2f}"
                          + (f" (labelled {m.get('team_label_frac', 0):.0%})"
                             if tag.endswith("teams") else ""), flush=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
