#!/usr/bin/env python
"""Find the men the tracks table has no box for and add them as new tracks (additive: existing ids untouched).

WHY. On play 1 (2026-09-20) the census read BAL short on 75 play frames: a Raven blocked at the line at 420 dropped
into coverage and stood alone on open turf with NO box in either camera for 173 frames (id 1 after 434), and the
centre's box merged into his neighbour's after 528. The pipeline's detector ran once at 1920 px and its tracks
went through a dozen rules; the raw detections are not kept. A fresh pass at a higher resolution and a lower
confidence floor, kept only where it finds a person the table does not have, recovers such men without touching
the ids the film has been folded on.

WHAT. For each frame of the play window, YOLO (person class) at --imgsz on the camera's footage; a detection is a
candidate when (a) its foot lands on the field through the camera (|x| < 60 m, |y| < 30 m, cameras.npz), (b) its
height is in the tracked players' range, (c) no existing box on that frame overlaps it at IoU >= --dup-iou or
holds its centre, (d) conf >= --conf. Candidates are linked frame to frame (IoU >= --link-iou, gaps <= --gap) and
runs shorter than --min-run are dropped. Each run becomes a new track: track_id and global_player_id from 900 up,
team from the kit classifier's sign (the play's T0/T1 mapping read off the existing rows), and a PlayerIdentity
('P<id>', unnamed) in identity_resolved.pkl. Backups: tracks.parquet.pre03e.N, identity_resolved.pkl.pre03e.N.
A film sheet of every new track is written to --sheet. Downstream, the new ids need keypoints (05m --ids) and
poses (05c --ids) before the loader draws them.

USAGE (nflgsplat env, GPU):
  python scripts/03e_redetect_missing.py --play-dir P --cam sideline --lo 395 --hi 607 [--apply] --sheet diag/x.png
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.pose.place_on_field import ground_point  # noqa: E402
from nfl_gsplat.render.play_timeline import BOX_MARGIN_FRAC  # noqa: E402
from nfl_gsplat.tracking.kits import kit_margins  # noqa: E402


def iou(p, q):
    w = max(0.0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0.0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i) if i > 0 else 0.0


def backup(path: Path) -> Path:
    n = 0
    while (path.with_name(path.name + f".pre03e.{n}")).exists():
        n += 1
    b = path.with_name(path.name + f".pre03e.{n}")
    b.write_bytes(path.read_bytes())
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--dup-iou", type=float, default=0.15)
    ap.add_argument("--link-iou", type=float, default=0.30)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--min-run", type=int, default=12)
    ap.add_argument("--first-id", type=int, default=900)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    P = a.play_dir
    df = pd.read_parquet(P / "tracks.parquet")
    cam_rows = df[df["cam"] == a.cam]
    have: dict = {}
    for r in cam_rows.itertuples():
        have.setdefault(int(r.frame), []).append((float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2)))
    heights = np.array([b[3] - b[1] for bs in have.values() for b in bs])
    h_lo, h_hi = float(np.percentile(heights, 2)) * 0.7, float(np.percentile(heights, 98)) * 1.3
    tracks = load_camera_track(P / "cameras.npz")
    tr = tracks[a.cam]

    from ultralytics import YOLO
    model = YOLO(str(Path(__file__).resolve().parents[1] / a.weights))
    cap = cv2.VideoCapture(str(P / f"{a.cam}.mp4"))
    cands: dict = {}
    n_det = 0
    for f in range(a.lo, a.hi + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok or f >= len(tr.conf) or tr.conf[f] <= 0:
            continue
        intr, pose = tr.at(f)
        K, R, t = intr.K(), pose.R, pose.t
        res = model.predict(img, imgsz=a.imgsz, conf=a.conf, classes=[0], verbose=False, device=0)[0]
        boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
        n_det += len(boxes)
        ex = have.get(f, [])
        for b, c in zip(boxes, confs):
            hgt = float(b[3] - b[1])
            if not (h_lo <= hgt <= h_hi):
                continue
            foot_v = float(b[3]) - BOX_MARGIN_FRAC * hgt
            try:
                g = ground_point((0.5 * float(b[0] + b[2]), foot_v), K, R, t)
            except Exception:  # noqa: BLE001
                continue
            if not (abs(g[0]) < 57.0 and abs(g[1]) < 25.5):
                continue                                  # off the field: the benches and the crowd behind the far sideline
            cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
            dup = any(iou(b, q) >= a.dup_iou or (q[0] <= cx <= q[2] and q[1] <= cy <= q[3]) for q in ex)
            if dup:
                continue
            cands.setdefault(f, []).append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), float(g[0]), float(g[1])))
    n_c = sum(len(v) for v in cands.values())
    print(f"{a.cam} {a.lo}-{a.hi}: {n_det} detections, {n_c} candidates on {len(cands)} frames (no existing box at IoU >= {a.dup_iou} or holding the centre)")

    # link into runs
    runs: list = []                                   # each: list of (frame, box...)
    open_runs: list = []
    for f in sorted(cands):
        used = set()
        for run in open_runs:
            lf, lb = run[-1][0], run[-1][1:5]
            if f - lf > a.gap:
                continue
            best, bi = 0.0, None
            for i, cnd in enumerate(cands[f]):
                if i in used:
                    continue
                v = iou(lb, cnd[:4])
                if v > best:
                    best, bi = v, i
            if bi is not None and best >= a.link_iou:
                run.append((f,) + cands[f][bi]); used.add(bi)
        for i, cnd in enumerate(cands[f]):
            if i not in used:
                open_runs.append([(f,) + cnd])
        keep = []
        for run in open_runs:
            if f - run[-1][0] > a.gap:
                runs.append(run)
            else:
                keep.append(run)
        open_runs = keep
    runs.extend(open_runs)
    runs = [r for r in runs if len(r) >= a.min_run]
    runs.sort(key=lambda r: r[0][0])
    print(f"runs of {a.min_run}+ frames: {len(runs)}")

    # team from the kit classifier: the sign of the margin against the play's own mapping
    lab = cam_rows.dropna(subset=["team"])
    sign_of = {}
    for tlabel, g in lab.groupby("team"):
        sign_of[str(tlabel)] = float(np.sign(np.nanmedian(g["kit_margin"])))
    team_names = {}
    ident = P / "identity_resolved.pkl"
    merged = pickle.load(open(ident, "rb")) if ident.exists() else None
    if merged is not None:
        by_team: dict = {}
        for pid, g in cam_rows.dropna(subset=["team"]).groupby("global_player_id"):
            tl = g["team"].mode()
            p = merged["merged"].get(int(pid))
            if len(tl) and p is not None and getattr(p, "team", None):
                by_team.setdefault(str(tl.iloc[0]), []).append(p.team)
        for tlabel, names in by_team.items():
            team_names[tlabel] = max(set(names), key=names.count)
    det_for_kit: dict = {}
    for ri, run in enumerate(runs):
        for row in run:
            det_for_kit.setdefault(int(row[0]), []).append((ri, row[1:5]))
    # the kit classifier needs both kits in its population: the frame's existing boxes go first, the new ones after
    kit_in = {}
    for f, rows in det_for_kit.items():
        kit_in[f] = np.array(list(have.get(f, [])) + [b for _ri, b in rows], float)
    run_margin: dict = {}
    try:
        margins, _centre = kit_margins(kit_in, str(P / f"{a.cam}.mp4"))
        for f, rows in det_for_kit.items():
            m = margins.get(f)
            if m is None:
                continue
            m = np.asarray(m).ravel()[len(have.get(f, [])):]
            for (ri, _b), mv in zip(rows, m):
                if np.isfinite(mv):
                    run_margin.setdefault(ri, []).append(float(mv))
    except Exception as exc:  # noqa: BLE001
        print(f"kit classifier: {exc} -- new tracks get no team")
    new_rows = []
    summary = []
    next_id = a.first_id
    existing_max = int(max(df["global_player_id"].max(), df["track_id"].max()))
    if next_id <= existing_max:
        next_id = existing_max + 1
    for ri, run in enumerate(runs):
        med = float(np.median(run_margin.get(ri, [np.nan]))) if run_margin.get(ri) else float("nan")
        tlabel = None
        if np.isfinite(med):
            for lab_, sg in sign_of.items():
                if np.sign(med) == sg:
                    tlabel = lab_
        team = team_names.get(tlabel) if tlabel else None
        pid = next_id; next_id += 1
        for row in run:
            f, x1, y1, x2, y2, c, gx, gy = row
            new_rows.append(dict(frame=int(f), cam=a.cam, track_id=pid, global_player_id=pid, bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                                 conf=c, foot_u=0.5 * (x1 + x2), foot_v=y2, jersey_number_ocr=np.nan, kit_margin=med,
                                 kit=(1.0 if tlabel == "T1" else 0.0 if tlabel == "T0" else np.nan), jersey_votes_win=0.0, jersey_votes_total=0.0,
                                 team=tlabel, player_uid=f"P{pid}", entity_type="player"))
        summary.append((pid, run[0][0], run[-1][0], len(run), round(float(np.mean([r[5] for r in run])), 2), round(med, 2) if np.isfinite(med) else None, tlabel, team,
                        (round(float(np.mean([r[6] for r in run])), 1), round(float(np.mean([r[7] for r in run])), 1))))
    print("new tracks (id, first, last, n, mean conf, kit margin, T-label, team, mean ground xy):")
    for s in summary:
        print("  ", s)

    if a.sheet and runs:
        tiles = []
        for pid, f0, f1, n, *_r in summary:
            run = next(r for r in runs if r[0][0] == f0)
            for row in (run[0], run[len(run) // 2], run[-1]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(row[0])); ok, img = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = (int(v) for v in row[1:5])
                for q in have.get(int(row[0]), []):
                    cv2.rectangle(img, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), (0, 200, 255), 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2; R_ = 180
                crop = img[max(0, cy - R_):cy + R_, max(0, cx - R_):cx + R_]
                pad = np.zeros((2 * R_, 2 * R_, 3), np.uint8); pad[:crop.shape[0], :crop.shape[1]] = crop
                cv2.putText(pad, f"new {pid} f{int(row[0])} {row[5]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(pad)
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        a.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.sheet), sheet)
        print(f"sheet: {a.sheet}")

    if not a.apply:
        print("dry run. Re-run with --apply to add the tracks and identities.")
        return
    if not new_rows:
        print("nothing to add")
        return
    b1 = backup(P / "tracks.parquet")
    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_parquet(P / "tracks.parquet", index=False)
    print(f"wrote tracks.parquet (+{len(new_rows)} rows; backup {b1.name})")
    if merged is not None:
        from nfl_gsplat.identity.merge_cameras import PlayerIdentity
        b2 = backup(ident)
        for pid, f0, f1, n, conf, med, tlabel, team, _xy in summary:
            merged["merged"][int(pid)] = PlayerIdentity(jersey=0, player=f"P{pid}", team=team or "", height_m=1.85, weight_lb=0.0,
                                                        tracks={a.cam: int(pid)}, votes={})
        pickle.dump(merged, open(ident, "wb"))
        print(f"wrote identity_resolved.pkl (+{len(summary)} unnamed ids; backup {b2.name})")


if __name__ == "__main__":
    main()
