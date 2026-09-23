#!/usr/bin/env python
"""Refine the endzone camera per frame against the bodies: the fused fit's joints that the SIDELINE anchors (its
detector confident and within AGREE_PX of the fit's projection there) are 3D landmarks; the confident endzone
detections of the same joints are their images; a five-parameter correction per frame (a small rotation, a shift
along the optical axis, a focal scale) is fitted, smoothed over frames, and written as a new cameras file.

WHY. 2026-09-23: on play 1 the fit's projection sits a median 8 px (p90 14) above or below every confident endzone
detection AT ONCE per frame, 4 px at the far end of the image and 30 px at the near end, smooth over time -- the
endzone camera's per-frame depth / zoom error seen from the fit's side (the sideline shows 0.3 px common mode). The
fused fit is fitted against that camera, so its poses carry the error, and any labels taken from it in the endzone
would too. The sideline camera (2.5 px total residual) is the ruler; the bodies are the landmarks.

USAGE (smplx312 venv; read-only on the play unless --out is inside it):
  python scripts/09i_refine_endzone.py --play-dir P --out P/cameras_refined.npz [--lo 214 --hi 616] [--smooth 5]
Prints the endzone residual per frame before and after (median px over the confident detections), the parameter
ranges, and writes <out> (the sideline camera unchanged) plus <out>.report.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.calibration.cameras_io import CameraTrack, load_camera_track, write_camera_track  # noqa: E402
from nfl_gsplat.pose import pseudo_labels as pl  # noqa: E402
from nfl_gsplat.pose.coco import COCO_TO_SMPLX  # noqa: E402
from nfl_gsplat.render.play_timeline import clip_offset  # noqa: E402

BODY = "data/body_models"


def project(K, R, t, X):
    p = (K @ (R @ X.T + t[:, None])).T
    return p[:, :2] / p[:, 2:3]


# The parameter set: a small rotation about the camera's own axes (rx, ry, rz), a shift of the camera in its own
# frame (dy: up/down, dz: along the optical axis) and a focal scale s. dz and s are nearly the same thing for a camera
# 88 m out (both scale the image about the principal point) and an unbounded solve of both wandered to thousands of
# metres; every model is bounded and the models are compared on the residual they leave.
MODELS = {"rot_focal": ("rx", "ry", "rz", "s"), "rot_depth": ("rx", "ry", "rz", "dz"),
          "rot_pos": ("rx", "ry", "rz", "dy", "dz"), "rot_pos_focal": ("rx", "ry", "rz", "dy", "dz", "s")}
BOUNDS = {"rx": 0.05, "ry": 0.05, "rz": 0.05, "dy": 3.0, "dz": 15.0, "s": 0.15}
SCALE = {"rx": 1e-3, "ry": 1e-3, "rz": 1e-3, "dy": 0.5, "dz": 1.0, "s": 0.01}


def apply_params(K, R, t, p, names):
    q = dict(zip(names, p))
    dR = Rotation.from_rotvec([q.get("rx", 0.0), q.get("ry", 0.0), q.get("rz", 0.0)]).as_matrix()
    R2 = dR @ R
    t2 = dR @ t + np.array([0.0, q.get("dy", 0.0), q.get("dz", 0.0)])
    K2 = K.copy()
    K2[0, 0] *= 1.0 + q.get("s", 0.0)
    K2[1, 1] *= 1.0 + q.get("s", 0.0)
    return K2, R2, t2


def solve_frame(K, R, t, X, x, names, *, f_scale: float = 6.0):
    """Fit one frame's parameters (``names`` from MODELS): landmarks ``X [N, 3]`` and their endzone images ``x [N, 2]``."""
    def resid(p):
        K2, R2, t2 = apply_params(K, R, t, p, names)
        return (project(K2, R2, t2, X) - x).ravel()
    lo = [-BOUNDS[n] for n in names]; hi = [BOUNDS[n] for n in names]
    res = least_squares(resid, np.zeros(len(names)), loss="soft_l1", f_scale=f_scale, max_nfev=300,
                        bounds=(lo, hi), x_scale=[SCALE[n] for n in names])
    return res.x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=int, default=214)
    ap.add_argument("--hi", type=int, default=616)
    ap.add_argument("--smooth", type=int, default=5, help="median over +-this many solved frames")
    ap.add_argument("--min-landmarks", type=int, default=12)
    ap.add_argument("--agree-px", type=float, default=pl.AGREE_PX)
    ap.add_argument("--anchor-conf", type=float, default=pl.ANCHOR_CONF)
    ap.add_argument("--model", default="rot_focal", choices=sorted(MODELS))
    a = ap.parse_args()
    names = MODELS[a.model]
    import smplx
    import torch

    P = Path(a.play_dir)
    model = smplx.create(BODY, model_type="smplx", gender="neutral", num_betas=10, use_pca=False, batch_size=1)
    tracks = load_camera_track(P / "cameras.npz")
    side, endz = tracks["sideline"], tracks["endzone"]
    refit = pickle.load(open(P / "poses_refit.json", "rb"))["frames"]
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    kg = {k: g.sort_values("joint") for k, g in kdf.groupby(["cam", "frame", "global_player_id"])}
    offset = clip_offset(P)
    inv = {s: k for k, s in COCO_TO_SMPLX.items()}                   # body joint -> COCO index

    def joints(rec):
        with torch.no_grad():
            res = model(betas=torch.tensor(np.asarray(rec["betas"])[None, :10].astype(np.float32)),
                        body_pose=torch.tensor(np.asarray(rec["body_pose"]).reshape(1, -1).astype(np.float32)),
                        global_orient=torch.tensor(np.asarray(rec["global_orient"]).reshape(1, 3).astype(np.float32)),
                        transl=torch.tensor(np.asarray(rec["transl"]).reshape(1, 3).astype(np.float32)))
        return res.joints[0, :22].numpy().astype(np.float64)

    frames = sorted(int(f) for f in refit if a.lo <= int(f) <= a.hi)
    solved: dict[int, np.ndarray] = {}
    before: dict[int, float] = {}
    counts: dict[int, int] = {}
    land: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f in frames:
        cf = f + offset                                            # the endzone clip frame beside sideline f
        if cf < 0 or cf >= endz.num_frames or endz.conf[cf] <= 0 or f >= side.num_frames or side.conf[f] <= 0:
            continue
        Ks, Ps = side.at(f)
        Ke, Pe = endz.at(cf)
        X, x = [], []
        for pid, rec in refit[f].items():
            pid = int(pid)
            gs = kg.get(("sideline", f, pid))
            ge = kg.get(("endzone", cf, pid))
            if gs is None or ge is None or len(gs) != 17 or len(ge) != 17:
                continue
            J = joints(rec)
            ps = project(Ks.K(), np.asarray(Ps.R, float), np.asarray(Ps.t, float), J)
            sx, sc = gs[["x", "y"]].to_numpy(float), gs["conf"].to_numpy(float)
            ex, ec = ge[["x", "y"]].to_numpy(float), ge["conf"].to_numpy(float)
            for s_idx, k in inv.items():
                if sc[k] >= a.anchor_conf and np.linalg.norm(ps[s_idx] - sx[k]) <= a.agree_px and ec[k] >= a.anchor_conf:
                    X.append(J[s_idx])
                    x.append(ex[k])
        if len(X) < a.min_landmarks:
            continue
        X, x = np.asarray(X), np.asarray(x)
        Ke_, Re, te = Ke.K(), np.asarray(Pe.R, float), np.asarray(Pe.t, float)
        before[cf] = float(np.median(np.linalg.norm(project(Ke_, Re, te, X) - x, axis=1)))
        solved[cf] = solve_frame(Ke_, Re, te, X, x, names)
        counts[cf] = len(X)
        land[cf] = (X, x)
    if not solved:
        raise SystemExit("no frames with enough landmarks")
    cfs = np.array(sorted(solved))
    Pm = np.stack([solved[c] for c in cfs])
    # smooth the parameters over the solved frames (the error is smooth in time; the per-frame solve is not)
    Ps_ = Pm.copy()
    for i in range(len(cfs)):
        lo, hi = max(0, i - a.smooth), min(len(cfs), i + a.smooth + 1)
        Ps_[i] = np.median(Pm[lo:hi], axis=0)
    after: dict[int, float] = {}
    for i, c in enumerate(cfs):
        Ke, Pe = endz.at(int(c))
        K2, R2, t2 = apply_params(Ke.K(), np.asarray(Pe.R, float), np.asarray(Pe.t, float), Ps_[i], names)
        X, x = land[int(c)]
        after[int(c)] = float(np.median(np.linalg.norm(project(K2, R2, t2, X) - x, axis=1)))
    b = np.array([before[int(c)] for c in cfs]); af = np.array([after[int(c)] for c in cfs])
    print(f"endzone frames solved: {len(cfs)} (landmarks per frame p50 {np.median(list(counts.values())):.0f})")
    print(f"residual to the confident endzone detections, median px per frame: before p50 {np.median(b):.1f} p90 {np.percentile(b, 90):.1f}"
          f"  ->  after p50 {np.median(af):.1f} p90 {np.percentile(af, 90):.1f}")
    print(f"model {a.model}; parameters (smoothed): " + "; ".join(
        f"{n} p50 {np.median(Ps_[:, i]):+.4f} [{Ps_[:, i].min():+.4f}, {Ps_[:, i].max():+.4f}]" for i, n in enumerate(names)))
    print(f"  rotation deg p50 {np.median(np.degrees(np.linalg.norm(Ps_[:, :3], axis=1))):.3f} max {np.degrees(np.linalg.norm(Ps_[:, :3], axis=1)).max():.3f}")
    # write: every endzone frame takes the nearest solved frame's parameters
    K_new, R_new, t_new = endz.K.copy(), endz.R.copy(), endz.t.copy()
    for c in range(endz.num_frames):
        i = int(np.argmin(np.abs(cfs - c)))
        K2, R2, t2 = apply_params(endz.K[c], endz.R[c], endz.t[c], Ps_[i], names)
        K_new[c], R_new[c], t_new[c] = K2, R2, t2
    new_endz = CameraTrack(K=K_new, R=R_new, t=t_new, conf=endz.conf.copy(), width=endz.width, height=endz.height)
    fps = float(np.load(P / "cameras.npz")["fps"])
    out = Path(a.out)
    write_camera_track(out, {"sideline": side, "endzone": new_endz}, fps=fps)
    report = {"model": a.model, "names": list(names), "frames": [int(c) for c in cfs], "before_px": b.tolist(),
              "after_px": af.tolist(), "params": Ps_.tolist(), "landmarks": [counts[int(c)] for c in cfs]}
    Path(str(out) + ".report.json").write_text(json.dumps(report))
    print(f"wrote {out} (endzone refined, sideline unchanged) and its report")


if __name__ == "__main__":
    main()
