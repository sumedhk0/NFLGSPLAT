#!/usr/bin/env python
"""Keypoint support per body-frame and per body_pose row, for the VPoser shift (pose.pose_prior).

WHY. The shift moves implausible poses toward the mocap manifold, but "implausible" is not "wrong": on play 1 a
back's low pass-protection crouch scored 11 and sat on its keypoints while a sprinter's flung arms scored 22 and did
not. So the shift is gated by how well each joint's fitted position matches the detector in the sideline camera --
per body_pose row, through the joint at the end of the row's bone (pose_prior.row_support). That needs the timeline
as the renderer draws it (gait and foot lock applied), one forward pass per body-frame, and the camera; this script
does that once and writes the result for 05k / 07l to read.

WHAT. Writes ``<play-dir>/pose_support.json``:
  {"window": [lo, hi], "rows": {"<pid>": {"<frame>": [21 residual px or null]}}, "score": {"<pid>": {"<frame>": s}}}
USAGE (smplx312):
  python scripts/09j_pose_support.py --play-dir P [--lo 395 --hi 607] [--body-models data/body_models]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", type=Path, required=True)
    ap.add_argument("--lo", type=int, default=None)
    ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import smplx
    import torch

    import nfl_gsplat.render.play_timeline as pt
    from nfl_gsplat.calibration.cameras_io import load_camera_track
    from nfl_gsplat.pose import pose_prior as pp
    from nfl_gsplat.render import foot_lock as fl
    from nfl_gsplat.render.gait import gait_timeline

    P = args.play_dir
    pe = json.loads((P / "play_end.json").read_text()) if (P / "play_end.json").exists() else {}
    lo = args.lo if args.lo is not None else int(pe.get("snap", 300))
    hi = args.hi if args.hi is not None else int(pe.get("end", 460))
    model = smplx.create(str(args.body_models), model_type="smplx", gender="neutral", num_betas=10, use_pca=False,
                         batch_size=1)
    team_of = {int(p): v.team for p, v in pickle.load(open(P / "identity_resolved.pkl", "rb"))["merged"].items()}
    tl, _tracks, df, *_ = pt.load_play_timeline(P, model)
    gait_timeline(tl)
    if fl.MODE != "off":
        fl.foot_lock_timeline(tl, str(args.body_models), team_of=team_of, boxes_df=df)
    tr = load_camera_track(str(P / "cameras.npz"))["sideline"]
    k = pd.read_parquet(P / "keypoints_2d.parquet")
    k = k[k["cam"] == "sideline"]
    piv = k.pivot_table(index=["global_player_id", "frame"], columns="joint", values=["x", "y", "conf"])
    vp = pp.load()

    def joints(s):
        with torch.no_grad():
            r = model(betas=torch.tensor(s.betas[None, :10].astype(np.float32)),
                      body_pose=torch.tensor(s.body_pose.reshape(1, -1).astype(np.float32)),
                      global_orient=torch.tensor(s.global_orient.reshape(1, 3).astype(np.float32)))
        j = r.joints[0, :22].numpy().astype(np.float64)
        return j - j[0]

    def project(f, X):
        K, R, t = tr.K[f], tr.R[f], np.asarray(tr.t[f]).ravel()
        p = K @ (R @ X + t)
        return p[:2] / p[2]

    rows_out: dict = {}
    score_out: dict = {}
    n_frames = n_keyed = 0
    for f, states in tl.states.items():
        if f < lo or f > hi:
            continue
        for s in states:
            pid = int(s.pid)
            n_frames += 1
            sc = float(pp.scores(vp, s.body_pose.reshape(1, 21, 3))[0])
            score_out.setdefault(str(pid), {})[str(int(f))] = round(sc, 3)
            if (pid, int(f)) not in piv.index or int(f) >= tr.num_frames or tr.conf[int(f)] <= 0:
                continue
            row = piv.loc[(pid, int(f))]
            J = joints(s)
            pz = -J[:, 2].min()
            per = {}
            for kk, jj in COCO_TO_SMPLX.items():
                if row["conf"][kk] < args.min_conf:
                    continue
                u = project(int(f), np.array([s.xy[0] + J[jj, 0], s.xy[1] + J[jj, 1], pz + J[jj, 2]]))
                per[kk] = float(np.hypot(u[0] - row["x"][kk], u[1] - row["y"][kk]))
            if not per:
                continue
            med = float(np.median(list(per.values())))
            rs = pp.row_support(per, med)
            rows_out.setdefault(str(pid), {})[str(int(f))] = [None if not np.isfinite(v) else round(float(v), 2) for v in rs]
            n_keyed += 1
    out = args.out or (P / "pose_support.json")
    out.write_text(json.dumps({"window": [lo, hi], "rows": rows_out, "score": score_out}))
    print(f"pose support: {n_keyed} of {n_frames} body-frames on {lo}-{hi} with sideline keypoints -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
