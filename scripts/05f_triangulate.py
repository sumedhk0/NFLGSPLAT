"""Triangulate each player's joints from BOTH views, instead of placing a foot point.

Placement currently uses one pixel per player -- the detection box's
bottom-centre -- intersected with the turf, and discards the ~22 joints the pose
network reprojects into each frame. That throws away exactly the information the
error is made of: measured on play_001, 2.2 m of the 2.9 m cross-camera
disagreement lies ALONG the viewing rays, and a foot point carries no depth at
all while two views constrain it directly.

Three checks, none of which need ground truth:

* REPROJECTION -- does the triangulated joint land back on both observations.
* JITTER -- departure from a locally smooth path, the same measure that showed
  the endzone placing at 1.08 m against the sideline's 0.01 m.
* STATURE -- the recovered ankle-to-crown height against the ROSTER, which is
  known to the inch and constrains nothing upstream. A triangulation that is
  quietly wrong in depth gets the scale wrong, and this is what would show it.

Runs in the SMPL-X env (.venv-smplx).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.errors import SetupError
from nfl_gsplat.pose.triangulate import (TriangulationConfig,
                                         triangulate_joints_two_view)
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# SMPL-X body joints, excluding hands and face. Enough for a skeleton and for
# the stature check; the rest are noisier and not needed here.
N_BODY_JOINTS = 22
# Ankles and the head, for stature. SMPL-X joint order: 7/8 ankles, 15 head.
ANKLE_JOINTS = (7, 8)
HEAD_JOINT = 15
# The head joint sits inside the skull, not at the crown, so a standing person
# measures short by roughly this much. Applied only for the stature REPORT, and
# never fed back into the geometry.
HEAD_TO_CROWN_M = 0.13


def smooth_jitter(points, win=9):
    a = np.asarray(points, float)
    if len(a) < win + 2:
        return np.nan
    ker = np.ones(win) / win
    sm = np.stack([np.convolve(a[:, i], ker, "valid") for i in range(a.shape[1])], 1)
    core = a[win // 2: win // 2 + len(sm)]
    return float(np.median(np.linalg.norm(core - sm, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--poses", required=True, nargs=2, type=Path)
    ap.add_argument("--identity", required=True, type=Path)
    ap.add_argument("--reproj-px-max", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    blobs = [pickle.load(open(p, "rb")) for p in args.poses]
    caches = {b["cam"]: b["frames"] for b in blobs}
    if len(caches) != 2:
        raise SetupError("both pose caches are for the same camera")
    cam_a, cam_b = sorted(caches)
    probe = next(iter(next(iter(caches[cam_a].values())).values()))
    if "joints2d" not in probe:
        raise SetupError(
            f"{args.poses[0]} has no 'joints2d'. Re-run 05c_pose_play.py: the "
            "2D joints are what makes triangulation possible, and a cache "
            "written before they were stored cannot be used here.")

    frames = sorted(set(caches[cam_a]) & set(caches[cam_b]))
    cams = load_camera_track(args.play_dir / "cameras.npz")
    ident = pickle.load(open(args.identity, "rb"))
    merged, stitch = ident["merged"], ident.get("stitch") or {}

    def frags(cam, pid):
        return {f for f, p in stitch.get(cam, {}).items() if p == pid} | {pid}

    pairs = {j: (frags(cam_a, p.tracks[cam_a]), frags(cam_b, p.tracks[cam_b]))
             for j, p in merged.items()
             if cam_a in p.tracks and cam_b in p.tracks}
    if not pairs:
        raise SetupError("no player is named by both cameras; identity supplies "
                         "the correspondence and geometry cannot")

    cfg = TriangulationConfig(reproj_px_max=args.reproj_px_max, conf_min=0.3)
    out_rows, fused = [], {}
    for jersey, (fa, fb) in sorted(pairs.items()):
        rows_a, rows_b, used = [], [], []
        for f in frames:
            ra = next((caches[cam_a][f][i] for i in fa if i in caches[cam_a][f]), None)
            rb = next((caches[cam_b][f][i] for i in fb if i in caches[cam_b][f]), None)
            if ra is None or rb is None:
                continue
            rows_a.append(np.asarray(ra["joints2d"])[:N_BODY_JOINTS])
            rows_b.append(np.asarray(rb["joints2d"])[:N_BODY_JOINTS])
            used.append(f)
        if len(used) < 20:
            _LOG.info("#%d: only %d shared frames, skipping", jersey, len(used))
            continue

        uv_a, uv_b = np.stack(rows_a), np.stack(rows_b)
        conf = np.ones(uv_a.shape[:2])
        # The tracks are indexed by POSITION here, so each camera needs a track
        # whose frames line up; slice both camera tracks to the frames used.
        sub = {c: _slice_track(cams[c], used) for c in (cam_a, cam_b)}
        res = triangulate_joints_two_view(
            {cam_a: {"uv": uv_a, "conf": conf},
             cam_b: {"uv": uv_b, "conf": conf}}, sub, cfg)

        ok = res.valid
        if not ok.any():
            _LOG.info("#%d: nothing triangulated within %.0f px", jersey,
                      args.reproj_px_max)
            continue
        root = np.array([np.nanmean(res.joints3d[t][ok[t]], axis=0)
                         if ok[t].any() else [np.nan] * 3
                         for t in range(len(used))])
        good = np.isfinite(root).all(axis=1)
        statures = []
        for t in range(len(used)):
            j = res.joints3d[t]
            if not (np.isfinite(j[list(ANKLE_JOINTS)]).all()
                    and np.isfinite(j[HEAD_JOINT]).all()):
                continue
            ankle_z = float(np.mean(j[list(ANKLE_JOINTS), 2]))
            statures.append(float(j[HEAD_JOINT, 2]) - ankle_z + HEAD_TO_CROWN_M)
        player = merged[jersey]
        out_rows.append({
            "jersey": jersey, "player": player.player,
            "frames": int(good.sum()),
            "valid_frac": float(ok.mean()),
            "reproj_px": float(np.nanmedian(res.reproj[ok])),
            "jitter_m": smooth_jitter(root[good][:, :2]) if good.sum() > 12 else np.nan,
            "stature_m": float(np.median(statures)) if statures else np.nan,
            "roster_m": float(player.height_m),
        })
        fused[jersey] = {"frames": used, "joints3d": res.joints3d,
                         "valid": res.valid}

    print(f"\n{'#':>4} {'player':22s} {'frames':>6} {'valid':>6} {'reproj':>8} "
          f"{'jitter':>8} {'stature':>8} {'roster':>7} {'error':>7}")
    for r in out_rows:
        err = (r["stature_m"] - r["roster_m"]
               if np.isfinite(r["stature_m"]) else np.nan)
        print(f"{('#%d' % r['jersey']):>4} {r['player']:22s} {r['frames']:6d} "
              f"{100 * r['valid_frac']:5.0f}% {r['reproj_px']:6.1f}px "
              f"{r['jitter_m']:6.2f} m {r['stature_m']:6.2f} m "
              f"{r['roster_m']:5.2f} m {err:+6.2f}")
    if out_rows:
        errs = [r["stature_m"] - r["roster_m"] for r in out_rows
                if np.isfinite(r["stature_m"])]
        jit = [r["jitter_m"] for r in out_rows if np.isfinite(r["jitter_m"])]
        print(f"\nmedian stature error {np.median(errs):+.2f} m "
              f"(spread {np.std(errs):.2f} m); median jitter "
              f"{np.median(jit):.2f} m")
        print("for comparison, foot-ray placement jitter measured "
              "0.01 m (sideline) and 1.08 m (endzone)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as fh:
            pickle.dump({"rows": out_rows, "joints": fused,
                         "cameras": (cam_a, cam_b)}, fh)
        _LOG.info("wrote %s", args.out)


def _slice_track(track, frames):
    """A CameraTrack holding only ``frames``, in that order.

    triangulate_joints_two_view indexes cameras by POSITION in the sequence, not
    by frame number, so the two must be lined up explicitly rather than assumed.
    """
    from nfl_gsplat.calibration.cameras_io import CameraTrack
    idx = np.asarray(frames, int)
    return CameraTrack(K=track.K[idx], R=track.R[idx], t=track.t[idx],
                       conf=np.asarray(track.conf)[idx],
                       width=track.width, height=track.height)


if __name__ == "__main__":
    main()
