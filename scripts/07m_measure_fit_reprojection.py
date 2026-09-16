"""A pose cache's OWN per-frame reprojection of one id against its keypoints in one camera, at the
fit's own transl (no timeline): lower joints (hips, knees, ankles) and upper limbs (shoulders, elbows,
wrists) RMS in px, plus the knee flexion angles. Says WHEN a fit leaves confident keypoints -- and,
read beside the same body drawn by the timeline (05q / 05t), WHERE the damage is made: play 1's id 0
sat at 3-5 px here while the timeline drew his legs 30 px off (the half-turn wrap, 2026-09-16).

  07m_measure_fit_reprojection.py --play-dir P --id 0 --lo 280 --hi 340 [--cam sideline] [CACHE ...]

Several caches print side by side (default: <play-dir>/poses_refit.json). Read-only.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

LOWER = [1, 2, 4, 5, 7, 8]
UPPER = [16, 17, 18, 19, 20, 21]


def knee_flexion_deg(body_pose):
    """(left, right) knee axis-angle magnitudes in degrees from a ``[63]`` or ``[21, 3]`` body_pose."""
    bp = np.asarray(body_pose, float).reshape(21, 3)
    return float(np.degrees(np.linalg.norm(bp[3]))), float(np.degrees(np.linalg.norm(bp[4])))


def rms_px(pix, uv, use):
    """RMS pixel distance over the ``use`` mask (nan when fewer than three joints)."""
    use = np.asarray(use, bool)
    if use.sum() < 3:
        return float("nan")
    d = np.asarray(pix, float)[use] - np.asarray(uv, float)[use]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("caches", type=Path, nargs="*")
    args = ap.parse_args()
    import pandas as pd
    import smplx
    import torch

    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.pose.coco import coco_to_body
    from nfl_gsplat.pose.fit_mono2d import project
    from nfl_gsplat.pose.keypoint_filter import reject_outliers

    P = args.play_dir
    caches = args.caches or [P / "poses_refit.json"]
    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral", num_betas=10,
                         use_pca=False, batch_size=1)
    track = load_camera_track(P / "cameras.npz")[args.cam]
    kall, _ = reject_outliers(pd.read_parquet(P / "keypoints_2d.parquet"))
    ks = kall[(kall["cam"] == args.cam) & (kall["global_player_id"] == args.id)]
    rows = {int(f): g.sort_values("joint") for f, g in ks.groupby("frame") if len(g) == 17}

    def joints(rec):
        with torch.no_grad():
            res = model(betas=torch.tensor(np.asarray(rec["betas"])[None, :10].astype(np.float32)),
                        body_pose=torch.tensor(np.asarray(rec["body_pose"]).reshape(1, -1).astype(np.float32)),
                        global_orient=torch.tensor(np.asarray(rec["global_orient"]).reshape(1, 3).astype(np.float32)),
                        transl=torch.tensor(np.asarray(rec["transl"]).reshape(1, 3).astype(np.float32)))
        return res.joints[0, :22].numpy().astype(np.float64)

    def rms(J, f, sel):
        g = rows.get(f)
        if g is None or f < 0 or f >= len(track.conf) or track.conf[f] <= 0:
            return float("nan")
        uv, conf = coco_to_body(g[["x", "y"]].to_numpy(float), g["conf"].to_numpy(float), min_conf=args.min_conf)
        use = np.zeros(22, bool)
        use[sel] = True
        use &= np.asarray(conf, float) >= args.min_conf
        intr, pose = track.at(f)
        pix, _ = project(intr.K(), np.asarray(pose.R, float), np.asarray(pose.t, float), J)
        return rms_px(pix, uv, use)

    tabs = {}
    for c in caches:
        fr = pickle.load(open(c, "rb"))["frames"]
        t = {}
        for f in range(args.lo, args.hi + 1):
            rec = fr.get(f, {}).get(args.id)
            if rec is None:
                continue
            J = joints(rec)
            t[f] = (rms(J, f, LOWER), rms(J, f, UPPER), *knee_flexion_deg(rec["body_pose"]))
        tabs[c.name] = t
    frames = sorted(set().union(*[set(t) for t in tabs.values()]))
    print(f"id {args.id}, {args.cam} px RMS lower | upper, knee L/R deg; caches: " + ", ".join(tabs))
    print(f"{'frame':>5} " + " | ".join(f"{n[:22]:>30}" for n in tabs))
    for f in frames:
        cells = []
        for t in tabs.values():
            if f in t:
                lo, up, kl, kr = t[f]
                cells.append(f"{lo:6.1f} {up:6.1f}  {kl:5.0f}/{kr:<5.0f}      ")
            else:
                cells.append(f"{'-':>30}")
        print(f"{f:>5} " + " | ".join(cells))
    for n, t in tabs.items():
        lo = np.array([v[0] for v in t.values()])
        up = np.array([v[1] for v in t.values()])
        print(f"{n}: lower p50/p90/max {np.nanpercentile(lo, 50):.1f}/{np.nanpercentile(lo, 90):.1f}/{np.nanmax(lo):.1f}"
              f"  upper p50/p90 {np.nanpercentile(up, 50):.1f}/{np.nanpercentile(up, 90):.1f}  n {len(t)}")


if __name__ == "__main__":
    main()
