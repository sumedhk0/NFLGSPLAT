"""05m: joints by two-view triangulation versus the fused monocular lift, measured.

WHY. Each view's SMPL-X comes from a monocular regressor whose depth is a
guess from the box; the calibrated cameras only place the two guesses and
take a compromise (05e). The regressor also reprojects its joints into the
frame (``joints2d`` in the per-view caches, original-frame pixels, in
SMPLest-X's OWN 137-joint layout -- the first 25 are its body set, not the
SMPL-X tree). Triangulating those 2-D joints through the per-frame cameras
gives 3-D joints from geometry instead of from a prior.

RULERS. Reprojection cannot judge this (triangulation minimises it). Three
things neither method optimises:
  * bone constancy -- a bone never changes length: std/mean over the frames
    of each player's bones (hip-knee, knee-ankle, shoulder-elbow, elbow-wrist,
    hip and shoulder widths);
  * ankle height -- ankles ride ~8 cm above the turf: median |z - 0.08| m;
  * stature -- ankle-to-head against the ROSTER height, known to the inch and
    constraining nothing upstream: median |stature - roster| m.
Ray miss (the two rays' closest approach, in pixels of each view) is printed
as the consistency of the 2-D evidence, not as a score.

USAGE (SMPL-X env):
  python scripts/05m_triangulate_compare.py --play-dir data/all22/<game>/play_001
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.utils.geometry import triangulate_two_views  # noqa: E402

# SMPLest-X joints_name[:25], the body block of its 137 layout.
SMPLESTX_BODY = ("Pelvis", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "Neck",
                 "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
                 "L_Big_toe", "L_Small_toe", "L_Heel", "R_Big_toe", "R_Small_toe", "R_Heel",
                 "L_Ear", "R_Ear", "L_Eye", "R_Eye", "Nose")
# SMPL-X body order (22) -> SMPLest-X keypoint(s); None = no 2-D evidence.
SMPLX_FROM_SMPLESTX = {
    0: ("Pelvis",), 1: ("L_Hip",), 2: ("R_Hip",), 4: ("L_Knee",), 5: ("R_Knee",),
    7: ("L_Ankle",), 8: ("R_Ankle",), 10: ("L_Big_toe",), 11: ("R_Big_toe",), 12: ("Neck",),
    15: ("L_Ear", "R_Ear"), 16: ("L_Shoulder",), 17: ("R_Shoulder",), 18: ("L_Elbow",),
    19: ("R_Elbow",), 20: ("L_Wrist",), 21: ("R_Wrist",),
}
BONES = {"hip-knee": [(1, 4), (2, 5)], "knee-ankle": [(4, 7), (5, 8)],
         "shoulder-elbow": [(16, 18), (17, 19)], "elbow-wrist": [(18, 20), (19, 21)],
         "hips": [(1, 2)], "shoulders": [(16, 17)]}
ANKLE_HEIGHT_M = 0.08


def body_uv(joints2d, bbox=None) -> np.ndarray:
    """``[22, 2]`` SMPL-X-ordered pixels from the 137 layout, NaN where no evidence.

    The caches hold the network's projection WITHOUT its Y-up flip: measured
    on play 1, ankles sit a fifth of the way down the box and ears near its
    bottom, and rays through them met 1.4 m above the turf. The projection's
    principal point is the box's vertical centre, so the flip is a mirror
    about it: v -> (y1 + y2) - v. ``bbox`` None keeps the cached values."""
    j = np.array(joints2d, float, copy=True)
    if bbox is not None:
        y1, y2 = float(bbox[1]), float(bbox[3])
        j[:, 1] = (y1 + y2) - j[:, 1]
    out = np.full((22, 2), np.nan)
    for k, names in SMPLX_FROM_SMPLESTX.items():
        out[k] = np.mean([j[SMPLESTX_BODY.index(n)] for n in names], axis=0)
    return out


def projection(track, frame):
    intr, pose = track.at(frame)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
    return K @ np.column_stack([pose.R, pose.t])


def reproject(P, X):
    h = np.column_stack([X, np.ones(len(X))]) @ P.T
    return h[:, :2] / h[:, 2:3]


def rulers(joints_by_frame: dict, roster_height):
    """Bone constancy, ankle height error and stature error for one player."""
    frames = sorted(joints_by_frame)
    J = np.stack([joints_by_frame[f] for f in frames])            # [T, 22, 3], NaN allowed
    cvs = []
    for pairs in BONES.values():
        for a, b in pairs:
            L = np.linalg.norm(J[:, a] - J[:, b], axis=1)
            L = L[np.isfinite(L)]
            if len(L) >= 4 and L.mean() > 0:
                cvs.append(L.std() / L.mean())
    ankles = np.concatenate([J[:, 7, 2], J[:, 8, 2]])
    ankles = ankles[np.isfinite(ankles)]
    stature = np.linalg.norm(J[:, 15] - 0.5 * (J[:, 7] + J[:, 8]), axis=1) + 0.10   # ear to ankle + crown
    stature = stature[np.isfinite(stature)]
    return {
        "bone_cv": float(np.median(cvs)) if cvs else np.nan,
        "ankle_err": float(np.median(np.abs(ankles - ANKLE_HEIGHT_M))) if len(ankles) else np.nan,
        "stature_err": (float(np.median(np.abs(stature - roster_height)))
                        if len(stature) and roster_height else np.nan),
        "frames": len(frames),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--cams", nargs=2, default=["sideline", "endzone"])
    ap.add_argument("--max-miss-px", type=float, default=40.0,
                    help="a joint whose two rays miss by more than this in either view is dropped")
    ap.add_argument("--out", type=Path, default=None, help="write the triangulated joints blob here")
    ap.add_argument("--no-mirror-v", dest="mirror_v", action="store_false",
                    help="take the cached joints2d as they are (see body_uv)")
    args = ap.parse_args()

    P = args.play_dir
    cam_a, cam_b = args.cams
    caches = {c: pickle.load(open(P / f"poses_{c}.json", "rb"))["frames"] for c in (cam_a, cam_b)}
    tracks = load_camera_track(P / "cameras.npz")
    ident = pickle.load(open(P / "identity_resolved.pkl", "rb"))
    merged, stitch = ident["merged"], ident.get("stitch") or {}
    fused = pickle.load(open(P / "poses_fused.json", "rb"))["fused"]

    def frags(cam, pid):
        m = stitch.get(cam, {})
        return {f for f, p in m.items() if p == pid} | {pid}

    def record(cam, frame, fr):
        for fid in fr:
            if frame in caches[cam] and fid in caches[cam][frame]:
                return caches[cam][frame][fid]
        return None

    tri_blob, rows, misses = {}, [], []
    for pid, player in sorted(merged.items()):
        if cam_a not in player.tracks or cam_b not in player.tracks:
            continue
        fa, fb = frags(cam_a, player.tracks[cam_a]), frags(cam_b, player.tracks[cam_b])
        per_frame = {}
        for frame in sorted(set(caches[cam_a]) & set(caches[cam_b])):
            ra, rb = record(cam_a, frame, fa), record(cam_b, frame, fb)
            if ra is None or rb is None:
                continue
            ua = body_uv(ra["joints2d"], ra["bbox"] if args.mirror_v else None)
            ub = body_uv(rb["joints2d"], rb["bbox"] if args.mirror_v else None)
            ok = np.isfinite(ua).all(1) & np.isfinite(ub).all(1)
            Pa, Pb = projection(tracks[cam_a], frame), projection(tracks[cam_b], frame)
            X = np.full((22, 3), np.nan)
            X[ok] = triangulate_two_views(ua[ok], ub[ok], Pa, Pb)
            miss = np.full(22, np.nan)
            miss[ok] = np.maximum(np.linalg.norm(reproject(Pa, X[ok]) - ua[ok], axis=1),
                                  np.linalg.norm(reproject(Pb, X[ok]) - ub[ok], axis=1))
            misses.extend(miss[ok].tolist())
            X[~(miss <= args.max_miss_px)] = np.nan
            per_frame[frame] = X
        if len(per_frame) < 4:
            continue
        tri_blob[pid] = per_frame
        height = getattr(player, "height_m", None)
        # The fused blob is smplx's 127-joint layout: body 0-21, ears 58/59.
        # Its head joint is the skull base; the triangulated head is the
        # ears' midpoint, so the fused stature uses the ears too.
        fused_frames = {}
        for f, j in fused.get(pid, {}).items():
            if f in per_frame:
                a = np.array(np.asarray(j)[:22], float, copy=True)
                if np.asarray(j).shape[0] >= 60:
                    a[15] = 0.5 * (np.asarray(j)[58] + np.asarray(j)[59])
                fused_frames[f] = a
        r_tri = rulers(per_frame, height)
        r_fus = rulers(fused_frames, height) if len(fused_frames) >= 4 else None
        rows.append((pid, getattr(player, "player", ""), r_tri, r_fus))

    if not rows:
        print("no player has both views in both caches", file=sys.stderr)
        sys.exit(2)
    misses = np.asarray(misses)
    print(f"{len(rows)} two-view players; ray miss median {np.median(misses):.1f} px, "
          f"p90 {np.percentile(misses, 90):.1f} px, joints kept {(misses <= args.max_miss_px).mean():.0%}")
    print(f"{'pid':>4} {'player':22s} {'frames':>6} | {'bone cv':>15} | {'ankle err m':>15} | "
          f"{'stature err m':>15}")
    print(f"{'':>4} {'':22s} {'':>6} | {'tri':>7} {'fused':>7} | {'tri':>7} {'fused':>7} | "
          f"{'tri':>7} {'fused':>7}")
    agg = {k: {"tri": [], "fused": []} for k in ("bone_cv", "ankle_err", "stature_err")}
    for pid, name, rt, rf in rows:
        def fmt(r, k):
            return f"{r[k]:7.3f}" if r and np.isfinite(r[k]) else f"{'--':>7}"
        print(f"{pid:>4} {str(name)[:22]:22s} {rt['frames']:>6} | "
              f"{fmt(rt, 'bone_cv')} {fmt(rf, 'bone_cv')} | "
              f"{fmt(rt, 'ankle_err')} {fmt(rf, 'ankle_err')} | "
              f"{fmt(rt, 'stature_err')} {fmt(rf, 'stature_err')}")
        for k in agg:
            if np.isfinite(rt[k]):
                agg[k]["tri"].append(rt[k])
            if rf and np.isfinite(rf[k]):
                agg[k]["fused"].append(rf[k])
    print("\nmedian over players:")
    for k, v in agg.items():
        t = np.median(v["tri"]) if v["tri"] else np.nan
        f = np.median(v["fused"]) if v["fused"] else np.nan
        print(f"  {k:12s} triangulated {t:.3f}   fused {f:.3f}   ({len(v['tri'])}/{len(v['fused'])} players)")
    if args.out:
        pickle.dump({"cam": "triangulated", "world": True, "frames_by_pid": tri_blob,
                     "joint_order": "smplx_body22", "missing": [3, 6, 9, 13, 14]},
                    open(args.out, "wb"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
