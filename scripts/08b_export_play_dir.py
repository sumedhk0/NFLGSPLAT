#!/usr/bin/env python
"""Turn an All-22 reconstruction into the play-dir the pose and render stages read.

Bridges two halves of the pipeline that were built apart. scripts/08 produces
per-frame cameras for both views (sampled frames) and per-frame player ground
placements; scripts/05c (pose), 05b (scene) and 05d (render) read a play-dir:

    cameras.npz      every frame, both views   cameras_io.CameraTrack
    tracks.parquet   one row per detection     tracking.detect_track.TRACK_COLUMNS

WHAT HAPPENS HERE, in order:

  1. Cameras for EVERY frame. 08 solves the sideline on ~25 sampled frames and
     the endzone on the frames the players fit; both cameras pan smoothly on
     tripods, so between solved frames the rotation is interpolated (slerp)
     and the lens linearly. Frames outside the solved span get conf = 0, which
     05c treats as "do not pose here".
  2. Detections on every frame of both views (YOLO; the tracker's ids are not
     used -- measured at 5-7 pieces per player, scripts/07j).
  3. Every detection put on the turf through its view's camera; the two views'
     ground clouds reconciled per frame (joint_views.match_count: same player
     within 2.5 m -> one point at the midpoint); unreconciled points kept from
     either view alone.
  4. Linked through time on the turf (tracking.link3d) -> global player ids.
  5. Each detection labelled with the global id of the linked point it came
     from, and written as the parquet the downstream stages expect, with
     track_id == global_player_id so a player is one identity in both views.

Identity (names, numbers) is NOT resolved here; 05b/05d take an identity
JSON separately. Written as one job so the next stage has a real input, not a
fixture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import CameraTrack, write_camera_track
from nfl_gsplat.calibration.from_players import feet_of
from nfl_gsplat.calibration.joint_views import MAX_GAP_M, ground_points
from nfl_gsplat.tracking import link3d
from nfl_gsplat.tracking.detect_track import TRACK_COLUMNS
from nfl_gsplat.tracking.kits import KIT_MARGIN, KIT_PENALTY_M, KitError, kit_margins, labels_from_margins

FIELD_HALF_X_M = 56.0
FIELD_HALF_Y_M = 26.0


def slerp(R0, R1, a):
    """Rotation between two rotations, fraction ``a`` of the way."""
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([R0, R1]))
    return Slerp([0.0, 1.0], rots)(float(a)).as_matrix()


def full_track(cams, n_frames, width, height):
    """Per-frame CameraTrack from cameras solved on a subset of frames."""
    solved = sorted(cams)
    K = np.zeros((n_frames, 3, 3))
    R = np.zeros((n_frames, 3, 3))
    t = np.zeros((n_frames, 3))
    conf = np.zeros(n_frames)
    lo, hi = solved[0], solved[-1]
    for f in range(n_frames):
        if f < lo or f > hi:
            K[f], R[f], t[f] = cams[lo if f < lo else hi]
            conf[f] = 0.0                      # outside the solved span
            continue
        j = int(np.searchsorted(solved, f, side="right")) - 1
        f0 = solved[j]
        f1 = solved[min(j + 1, len(solved) - 1)]
        if f1 == f0:
            K[f], R[f], t[f] = cams[f0]
        else:
            a = (f - f0) / (f1 - f0)
            K0, R0, t0 = cams[f0]
            K1, R1, t1 = cams[f1]
            K[f] = (1 - a) * K0 + a * K1
            R[f] = slerp(R0, R1, a)
            centre0, centre1 = -R0.T @ t0, -R1.T @ t1
            centre = (1 - a) * centre0 + a * centre1
            t[f] = -R[f] @ centre
        conf[f] = 1.0
    return CameraTrack(K=K, R=R, t=t, conf=conf, width=int(width), height=int(height))


def detect_all_frames(model, path, *, conf=0.15, imgsz=1920, device="cuda:0"):
    """``{frame: boxes [N, 4]}`` for every frame, plus ``(width, height, n)``."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    out = {}
    for i, res in enumerate(model.predict(source=str(path), stream=True, classes=[0],
                                          conf=conf, imgsz=imgsz, device=device,
                                          verbose=False)):
        if res.boxes is not None and len(res.boxes):
            out[i] = res.boxes.xyxy.cpu().numpy()
    return out, (w, h, n)


def fuse_frame(cam_s, feet_s, cam_e, feet_e, *, gap_m=MAX_GAP_M, kit_s=None, kit_e=None):
    """One frame: ground points from both views, reconciled where they agree.

    Returns ``(points [M, 2], src_s [M] row index into feet_s or -1,
    src_e [M] row index into feet_e or -1)``.

    ``kit_s`` / ``kit_e`` (``[N]`` ints, -1 unknown; tracking.kits) refuse a
    pair whose two boxes wear different kits: measured 2026-09-05, 24 %, 53 %
    and 30 % of the global ids seen in both cameras on plays 1, 2 and 4
    disagreed on the kit, i.e. the pairing had joined different players.
    """
    from scipy.optimize import linear_sum_assignment

    gs = ground_points(cam_s, feet_s) if len(feet_s) else np.zeros((0, 2))
    ge = ground_points(cam_e, feet_e) if len(feet_e) else np.zeros((0, 2))

    def on_field(g):
        return (np.isfinite(g).all(1) & (np.abs(g[:, 0]) <= FIELD_HALF_X_M)
                & (np.abs(g[:, 1]) <= FIELD_HALF_Y_M))

    ok_s, ok_e = on_field(gs), on_field(ge)
    ia, ib = np.flatnonzero(ok_s), np.flatnonzero(ok_e)
    pts, src_s, src_e = [], [], []
    used_s, used_e = set(), set()
    if len(ia) and len(ib):
        cost = np.linalg.norm(gs[ia][:, None] - ge[ib][None], axis=2)
        if kit_s is not None and kit_e is not None:
            ks, ke = np.asarray(kit_s, int)[ia], np.asarray(kit_e, int)[ib]
            mismatch = (ks[:, None] >= 0) & (ke[None, :] >= 0) & (ks[:, None] != ke[None, :])
            cost = np.where(mismatch, 1e6, cost)
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            if cost[i, j] < gap_m:
                pts.append(0.5 * (gs[ia[i]] + ge[ib[j]]))
                src_s.append(int(ia[i]))
                src_e.append(int(ib[j]))
                used_s.add(int(ia[i]))
                used_e.add(int(ib[j]))
    for i in ia:
        if i not in used_s:
            pts.append(gs[i])
            src_s.append(int(i))
            src_e.append(-1)
    for j in ib:
        if j not in used_e:
            pts.append(ge[j])
            src_s.append(-1)
            src_e.append(int(j))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int)
    return np.asarray(pts), np.asarray(src_s), np.asarray(src_e)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", type=Path,
                    default=Path("C:/Users/sumedh/diag/all22_reconstruction.npz"))
    ap.add_argument("--root", type=Path,
                    default=Path("data/all22/bal_at_kc_2024_wk1"))
    ap.add_argument("--sideline", default="001_Sideline_KC_2-20_BLT_24.mp4")
    ap.add_argument("--endzone", default="002_Endzone_KC_2-20_BLT_24.mp4")
    ap.add_argument("--out", type=Path, required=True, help="play-dir to write")
    ap.add_argument("--cameras", type=Path, default=None,
                    help="a cameras.npz with per-frame tracks for both views (08e refine, 08h endzone from "
                         "the footage) used instead of --recon's sparse solves interpolated; the "
                         "interpolation was measured to invent metres of endzone motion (play 2)")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--fps", type=float, default=59.94)
    ap.add_argument("--device", default="cuda:0", help="YOLO device; 'cpu' when the GPU is off limits")
    ap.add_argument("--pairing", choices=("track", "frame"), default="track",
                    help="track: per-camera ground tracks paired by trajectory over their overlap "
                         "(default); frame: the per-frame pairing by ground distance (measured a coin "
                         "flip on play 2)")
    ap.add_argument("--kit-link", choices=("off", "block", "soft"), default="off",
                    help="the kit in the time linker: off (default; the kit only vetoes cross-camera "
                         "pairs), block (a mismatch never joins), soft (a mismatch costs KIT_PENALTY_M "
                         "metres). Measured on the helmet set 2026-09-05 with 2-5 %% label noise: block "
                         "worse (purity p10 0.45 -> 0.37), soft neutral to slightly worse -- off wins")
    args = ap.parse_args()

    from ultralytics import YOLO

    given = None
    if args.cameras is not None:
        from nfl_gsplat.calibration.cameras_io import load_camera_track
        given = load_camera_track(args.cameras)
        print("cameras: per frame from " + str(args.cameras) + "; " + ", ".join(
            f"{cam} {int((tr.conf > 0).sum())} of {len(tr.conf)} frames" for cam, tr in given.items()))
    else:
        z = np.load(args.recon)
        cams_s = {int(f): (z["side_K"][i], z["side_R"][i], z["side_t"][i])
                  for i, f in enumerate(z["side_frames"])}
        cams_e = {int(f): (z["end_K"][i], z["end_R"][i], z["end_t"][i])
                  for i, f in enumerate(z["end_frames"])}
        print(f"cameras: sideline {len(cams_s)} solved frames, endzone {len(cams_e)}")

    model = YOLO(args.model)
    det_s, (w_s, h_s, n_s) = detect_all_frames(model, args.root / args.sideline, device=args.device)
    det_e, (w_e, h_e, n_e) = detect_all_frames(model, args.root / args.endzone, device=args.device)
    n = min(n_s, n_e)
    print(f"detections: sideline {len(det_s)} frames, endzone {len(det_e)} frames, "
          f"{n} frames in common")

    args.out.mkdir(parents=True, exist_ok=True)
    if given is not None:
        track_s, track_e = given["sideline"], given["endzone"]
        if len(track_s.conf) < n or len(track_e.conf) < n:
            n = min(n, len(track_s.conf), len(track_e.conf))
            print(f"   the given cameras cover {n} frames; linking that many")
        if args.out / "cameras.npz" != args.cameras:
            write_camera_track(args.out / "cameras.npz",
                               {"sideline": track_s, "endzone": track_e}, fps=args.fps)
    else:
        track_s = full_track(cams_s, n, w_s, h_s)
        track_e = full_track(cams_e, n, w_e, h_e)
        write_camera_track(args.out / "cameras.npz",
                           {"sideline": track_s, "endzone": track_e}, fps=args.fps)

    # Kit per detection box (tracking.kits): the pairing across cameras and the
    # linking through time both refuse to join different kits.
    try:
        marg_s, cen_s = kit_margins(det_s, args.root / args.sideline)
        marg_e, cen_e = kit_margins(det_e, args.root / args.endzone)
        kits_s = {f: labels_from_margins(m) for f, m in marg_s.items()}
        kits_e = {f: labels_from_margins(m) for f, m in marg_e.items()}
        n_lab = sum(int((k >= 0).sum()) for k in kits_s.values()) + sum(int((k >= 0).sum()) for k in kits_e.values())
        n_all = sum(len(k) for k in kits_s.values()) + sum(len(k) for k in kits_e.values())
        print(f"kits: saturation centres sideline {cen_s[0]:.0f}/{cen_s[1]:.0f}, endzone "
              f"{cen_e[0]:.0f}/{cen_e[1]:.0f}; {n_lab} of {n_all} boxes labelled at margin {KIT_MARGIN}")
    except KitError as exc:
        # Two kits that saturation cannot tell apart (both white, say): no
        # labels at all rather than random ones -- measured, random labels
        # wreck the linker.
        print(f"kits REFUSED, pairing and linking on position alone: {exc}")
        marg_s, marg_e, kits_s, kits_e = {}, {}, {}, {}
    if args.pairing == "track":
        # Per-camera tracks paired by trajectory (tracking.pair_tracks): the
        # per-frame pairing was measured a coin flip (1.4 m between the two
        # cameras' foot points of one player, players 1-2 m apart).
        from nfl_gsplat.tracking.pair_tracks import camera_placements, global_ids, pair_tracks

        def on_field(g):
            return (np.isfinite(g).all(1) & (np.abs(g[:, 0]) <= FIELD_HALF_X_M)
                    & (np.abs(g[:, 1]) <= FIELD_HALF_Y_M))

        plc_s, idx_s = camera_placements(det_s, track_s, n, on_field=on_field)
        plc_e, idx_e = camera_placements(det_e, track_e, n, on_field=on_field)
        lab_s = {f: kits_s[f][idx_s[f]] for f in plc_s if f in kits_s}
        lab_e = {f: kits_e[f][idx_e[f]] for f in plc_e if f in kits_e}
        pen = KIT_PENALTY_M if args.kit_link == "soft" else None
        tr_s = link3d.link(plc_s, labels=lab_s if args.kit_link != "off" else None, fps=args.fps,
                           label_penalty_m=pen)
        tr_e = link3d.link(plc_e, labels=lab_e if args.kit_link != "off" else None, fps=args.fps,
                           label_penalty_m=pen)
        pairs, lag = pair_tracks(tr_s, tr_e, fps=args.fps)
        gid_s, gid_e = global_ids(len(tr_s), len(tr_e), pairs)
        n_frames = len(set(plc_s) | set(plc_e))
        print(f"per-camera tracks: sideline {len(tr_s)} over {len(plc_s)} frames, endzone {len(tr_e)} over "
              f"{len(plc_e)} frames; {len(pairs)} track pairs (mean offset "
              f"{np.median([q.cost for q in pairs]) if pairs else float('nan'):.2f} m median, overlap "
              f"{np.median([q.overlap for q in pairs]) if pairs else 0:.0f} frames median) at endzone lag {lag:+d} "
              f"frames; {len(set(gid_s) | set(gid_e))} global ids, "
              f"{len(set(gid_s) & set(gid_e))} in both cameras")
        rows, kit_rows = [], []
        for cam, tracks, gids, idx, det, marg in (("sideline", tr_s, gid_s, idx_s, det_s, marg_s),
                                                  ("endzone", tr_e, gid_e, idx_e, det_e, marg_e)):
            for t, gid in zip(tracks, gids):
                for f, r in zip(t.frames, t.rows):
                    src = int(idx[f][r])
                    b = det[f][src]
                    m = marg.get(f)
                    kit_rows.append(float(m[src]) if m is not None else float("nan"))
                    rows.append({
                        "frame": int(f), "cam": cam, "track_id": int(gid),
                        "global_player_id": int(gid),
                        "bbox_x1": float(b[0]), "bbox_y1": float(b[1]),
                        "bbox_x2": float(b[2]), "bbox_y2": float(b[3]),
                        "conf": 1.0, "foot_u": float((b[0] + b[2]) / 2.0),
                        "foot_v": float(b[3]), "jersey_number_ocr": -1,
                    })
        placements = {f: None for f in range(n_frames)}       # for the span print below
        tracks = tr_s + tr_e
        print(f"linked {len(tracks)} per-camera tracks; "
              f"{sum(len(t.frames) >= 0.5 * n_frames for t in tracks)} span half the play")
    else:
        # Fuse and link on the turf.
        placements, sources, labels = {}, {}, {}
        for f in range(n):
            if track_s.conf[f] <= 0 or track_e.conf[f] <= 0:
                continue
            bs = det_s.get(f, np.zeros((0, 4)))
            be = det_e.get(f, np.zeros((0, 4)))
            ks = kits_s.get(f, np.full(len(bs), -1))
            ke = kits_e.get(f, np.full(len(be), -1))
            pts, src_s, src_e = fuse_frame((track_s.K[f], track_s.R[f], track_s.t[f]), feet_of(bs),
                                           (track_e.K[f], track_e.R[f], track_e.t[f]), feet_of(be),
                                           kit_s=ks, kit_e=ke)
            if len(pts):
                placements[f] = pts
                sources[f] = (src_s, src_e)
                lab = np.full(len(pts), -1, int)
                for k in range(len(pts)):
                    a = int(ks[src_s[k]]) if src_s[k] >= 0 else -1
                    b = int(ke[src_e[k]]) if src_e[k] >= 0 else -1
                    lab[k] = a if a >= 0 else b            # a pair never disagrees (gated above)
                labels[f] = lab
        n_two = sum(int(((s_ >= 0) & (e_ >= 0)).sum()) for s_, e_ in sources.values())
        print(f"fused: {sum(len(p) for p in placements.values())} placements, {n_two} seen by both cameras, "
              f"{sum(int((l >= 0).sum()) for l in labels.values())} carry a kit")
        tracks = link3d.link(placements, labels=labels if args.kit_link != "off" else None, fps=args.fps,
                             label_penalty_m=KIT_PENALTY_M if args.kit_link == "soft" else None)
        print(f"linked {len(tracks)} tracks over {len(placements)} frames; "
              f"{sum(len(t.frames) >= 0.5 * len(placements) for t in tracks)} span half the play")

        # Global id per fused point (index-aligned), then per detection.
        gid_by_frame = link3d.assignments(tracks, placements)
        rows, kit_rows = [], []
        for f in placements:
            pts = placements[f]
            src_s, src_e = sources[f]
            bs = det_s.get(f, np.zeros((0, 4)))
            be = det_e.get(f, np.zeros((0, 4)))
            for k, p in enumerate(pts):
                gid = int(gid_by_frame[f][k])
                for cam, src, boxes in (("sideline", src_s[k], bs), ("endzone", src_e[k], be)):
                    if src < 0:
                        continue
                    b = boxes[src]
                    marg = (marg_s if cam == "sideline" else marg_e).get(f)
                    kit_rows.append(float(marg[src]) if marg is not None else float("nan"))
                    rows.append({
                        "frame": int(f), "cam": cam, "track_id": int(gid),
                        "global_player_id": int(gid),
                        "bbox_x1": float(b[0]), "bbox_y1": float(b[1]),
                        "bbox_x2": float(b[2]), "bbox_y2": float(b[3]),
                        "conf": 1.0, "foot_u": float((b[0] + b[2]) / 2.0),
                        "foot_v": float(b[3]), "jersey_number_ocr": -1,
                    })
    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    # Per detection: the kit its own box wears (signed margin; label at
    # KIT_MARGIN) so identity votes on the same evidence without re-reading
    # the videos, and a global id whose two views disagree is visible.
    df["kit_margin"] = kit_rows
    df["kit"] = labels_from_margins(df["kit_margin"].to_numpy())
    per_id = df[df["kit"] >= 0].groupby(["track_id", "cam"])["kit"].agg(lambda s: int(s.mean() > 0.5)).unstack()
    if {"sideline", "endzone"} <= set(per_id.columns):
        both = per_id.dropna()
        n_dis = int((both["sideline"] != both["endzone"]).sum())
        print(f"kits per global id: {len(both)} ids in both cameras, {n_dis} disagree on the kit")
    df.to_parquet(args.out / "tracks.parquet", index=False)
    per_frame = df.groupby(["frame", "cam"]).size()
    print(f"tracks.parquet: {len(df)} rows, {df['global_player_id'].nunique()} players, "
          f"median {per_frame.median():.0f} per frame per view")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
