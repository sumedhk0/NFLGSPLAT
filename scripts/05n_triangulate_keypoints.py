"""05n: joints triangulated from detector keypoints in both calibrated views.

WHY. The 3-D joints have come from a monocular regressor per view, with
the calibrated cameras used only afterwards to place two depth guesses
and take a compromise. With the sideline and endzone cameras nearly
orthogonal and lenses of ~9000 px focal length, a 4 px keypoint error at
100 m is ~4 cm: geometry can do what the prior guesses at.

WHAT. Reads 05m's keypoints (COCO-17 per tracked person per view), maps
them onto the SMPL-X body tree (12 limb joints; pelvis from the hips,
neck from the shoulders, head from the face; NaN elsewhere), solves the
endzone frame offset that makes the two views agree best (the clips are
synchronised only to a frame or two), and triangulates per player with
pose.triangulate.triangulate_joints_two_view. Writes a blob in the
fused-joints format so 05f_refit_fused.py fits SMPL-X parameters to it
unchanged. Runs in the SMPL-X env (numpy 1 pickles).

USAGE:
  python scripts/05n_triangulate_keypoints.py --play-dir data/all22/<game>/play_001
  python scripts/05f_refit_fused.py --play-dir P --fused P/poses_tri.json \\
      --poses P/poses_sideline.json --identity P/identity_resolved.pkl --out P/poses_refit_tri.json
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.pose.forward_kinematics import NUM_BODY_JOINTS  # noqa: E402
from nfl_gsplat.pose.triangulate import TriangulationConfig, triangulate_joints_two_view  # noqa: E402
from nfl_gsplat.utils.logging import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# COCO-17 index -> SMPL-X body joint (pose.forward_kinematics tree)
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
                 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
COCO_L_HIP, COCO_R_HIP, COCO_L_SHO, COCO_R_SHO = 11, 12, 5, 6
COCO_FACE = (0, 1, 2, 3, 4)
SMPLX_PELVIS, SMPLX_NECK, SMPLX_HEAD = 0, 12, 15
OFFSETS = range(-3, 4)


def coco_to_body(xy, conf, *, min_conf: float):
    """``(uv [22, 2], conf [22])`` in SMPL-X body order; 0 / 0 where unknown
    (the triangulator's DLT needs finite pixels; conf 0 fails its gate)."""
    uv = np.zeros((NUM_BODY_JOINTS, 2))
    c = np.zeros(NUM_BODY_JOINTS)
    ok = conf >= min_conf
    for k, s in COCO_TO_SMPLX.items():
        if ok[k]:
            uv[s], c[s] = xy[k], conf[k]
    if ok[COCO_L_HIP] and ok[COCO_R_HIP]:
        uv[SMPLX_PELVIS] = 0.5 * (xy[COCO_L_HIP] + xy[COCO_R_HIP])
        c[SMPLX_PELVIS] = min(conf[COCO_L_HIP], conf[COCO_R_HIP])
    if ok[COCO_L_SHO] and ok[COCO_R_SHO]:
        uv[SMPLX_NECK] = 0.5 * (xy[COCO_L_SHO] + xy[COCO_R_SHO])
        c[SMPLX_NECK] = min(conf[COCO_L_SHO], conf[COCO_R_SHO])
    face = [k for k in COCO_FACE if ok[k]]
    if len(face) >= 2:
        uv[SMPLX_HEAD] = xy[face].mean(0)
        c[SMPLX_HEAD] = float(np.mean(conf[face]))
    return uv, c


def per_player_views(kdf, *, min_conf: float):
    """``{pid: {cam: {frame: (uv [22, 2], conf [22])}}}``."""
    out: dict = {}
    for (pid, cam, f), g in kdf.groupby(["global_player_id", "cam", "frame"]):
        g = g.sort_values("joint")
        xy = g[["x", "y"]].to_numpy(float)
        conf = g["conf"].to_numpy(float)
        if len(xy) != 17:
            continue
        out.setdefault(int(pid), {}).setdefault(cam, {})[int(f)] = coco_to_body(xy, conf, min_conf=min_conf)
    return out


def stack_pair(views, cam_a, cam_b, offset):
    """Frames both views have (endzone shifted by ``offset``) -> arrays for the triangulator."""
    fa = set(views.get(cam_a, {}))
    fb = {f - offset for f in views.get(cam_b, {})}
    frames = sorted(fa & fb)
    if not frames:
        return [], None
    uv_a = np.stack([views[cam_a][f][0] for f in frames])
    c_a = np.stack([views[cam_a][f][1] for f in frames])
    uv_b = np.stack([views[cam_b][f + offset][0] for f in frames])
    c_b = np.stack([views[cam_b][f + offset][1] for f in frames])
    return frames, {cam_a: {"uv": uv_a, "conf": c_a}, cam_b: {"uv": uv_b, "conf": c_b}}


class RowTrack:
    """A CameraTrack indexed by observation ROW: the triangulator reads
    ``cameras[cam].at(row)`` for row 0..T-1, while the rows are an arbitrary
    subset of frames; ``offset`` is the clip's frame shift."""

    def __init__(self, track, frames, offset: int = 0):
        self._t, self._frames, self._o = track, [int(f) for f in frames], int(offset)

    def at(self, row):
        return self._t.at(self._frames[int(row)] + self._o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--keypoints", type=Path, default=None, help="default <play-dir>/keypoints_2d.parquet")
    ap.add_argument("--out", type=Path, default=None, help="default <play-dir>/poses_tri.json")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--reproj-px-max", type=float, default=20.0)
    ap.add_argument("--offset", type=int, default=None, help="endzone frame offset; default: solved")
    args = ap.parse_args()

    P = args.play_dir
    tracks = load_camera_track(P / "cameras.npz")
    kdf = pd.read_parquet(args.keypoints or P / "keypoints_2d.parquet")
    cams = sorted(kdf["cam"].unique())
    if len(cams) != 2:
        raise SystemExit(f"need keypoints from two cameras, got {cams}")
    cam_a, cam_b = ("sideline", "endzone") if set(cams) == {"sideline", "endzone"} else cams
    views = per_player_views(kdf, min_conf=args.min_conf)
    cfg = TriangulationConfig(reproj_px_max=args.reproj_px_max, conf_min=args.min_conf)
    t0 = time.time()

    # Frame offset: the shift of the endzone clip that makes the views agree.
    if args.offset is None:
        scores = {}
        for d in OFFSETS:
            errs = []
            for pid, v in views.items():
                frames, obs = stack_pair(v, cam_a, cam_b, d)
                if not frames:
                    continue
                cams_d = {cam_a: RowTrack(tracks[cam_a], frames), cam_b: RowTrack(tracks[cam_b], frames, d)}
                loose = TriangulationConfig(reproj_px_max=np.inf, conf_min=args.min_conf)
                res = triangulate_joints_two_view(obs, cams_d, loose)
                e = res.reproj[res.valid]
                errs.extend(e.reshape(-1).tolist())
            scores[d] = float(np.median(errs)) if errs else np.nan
        offset = min((d for d in scores if np.isfinite(scores[d])), key=lambda d: scores[d])
        print("frame offset search (median reprojection px): " + ", ".join(
            f"{d:+d}: {scores[d]:.1f}" for d in OFFSETS) + f" -> {cam_b} offset {offset:+d}")
    else:
        offset = args.offset

    fused: dict = {}
    stats: dict = {}
    n_joints = n_valid = 0
    for pid, v in views.items():
        frames, obs = stack_pair(v, cam_a, cam_b, offset)
        if not frames:
            continue
        cams_d = {cam_a: RowTrack(tracks[cam_a], frames), cam_b: RowTrack(tracks[cam_b], frames, offset)}
        res = triangulate_joints_two_view(obs, cams_d, cfg)
        per = {}
        for i, f in enumerate(frames):
            if res.valid[i].sum() >= 6:
                per[int(f)] = res.joints3d[i].astype(np.float32)
        if len(per) >= 8:
            fused[int(pid)] = per
            stats[int(pid)] = {"frames": len(per), "valid_joint_frac": float(res.valid.mean()),
                               "reproj_px_median": float(np.nanmedian(res.reproj[res.valid]))}
        n_joints += res.valid.size
        n_valid += int(res.valid.sum())
    out = args.out or P / "poses_tri.json"
    blob = {"fused": fused, "stats": stats, "cameras": (cam_a, cam_b), "offset": int(offset),
            "source": "05n keypoints triangulation"}
    with open(out, "wb") as fh:
        pickle.dump(blob, fh)
    med = float(np.median([s["reproj_px_median"] for s in stats.values()])) if stats else float("nan")
    print(f"triangulated {len(fused)} players ({len(views)} had keypoints), valid joints "
          f"{100 * n_valid / max(1, n_joints):.0f}%, median reprojection {med:.1f} px, "
          f"{time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
