#!/usr/bin/env python
"""How far a fitted body stands from the point BOTH cameras put its feet, split along and across the ray.

    python scripts/05u_along_ray_gap.py --play-dir P --refit a.json [b.json]

WHY. 05t scores placement in pixels of the camera a body was NOT fitted to, which is the right ruler but
an ambiguous unit: pixels are not comparable between these two cameras (their px per metre differ), and a
change can trade pixels in one view for pixels in the other. This is the same question in metres, and it
decided the two adoptions this project has made:

  the two-view bound (05p TWO_VIEW_PX_MAX)  |along| p90 0.21 -> 0.05 m, p99 2.51 -> 0.34, 0 of 23 worse
  id 19 alone                               1.04 m -> 0.05 m while every other player moved under 1 cm

THE SPLIT IS THE DIAGNOSIS, not decoration. The two cameras' ankle rays cross at one point; a body's
distance from it decomposes into the component ALONG the fitted camera's ray and the component ACROSS it:

  along only, across ~0   a depth slide -- the body sits on its own camera's ray at the wrong range,
                          which is the one axis that camera cannot see. A placement fault.
  both large              the two boxes are different men. An identity fault, and no amount of
                          re-placing will fix it (play 1's id 19 stood 1.04 m along and 0.02 m across).

Give two caches to compare them on exactly the frames both cover, which is how a change is judged: the
pooled percentiles move and NO player may get materially worse.

WHAT THIS RULER CANNOT SEE, and it matters when judging a pairing change. A player the other camera does
not hold has no crossing point, so he is absent from the table entirely -- not scored as bad, simply
missing. A change that UNPAIRS someone therefore makes him vanish rather than show as worse, and the
pooled numbers improve for free. Play 1's id 19 did exactly that: he was the worst body in v35 (31.4 px
in the camera he was fitted to, 16.5/86.6 in the other) because his endzone partner was a different man
(the two cameras' own ground points for "him" sat 1.31 m apart), and relabelling that endzone track onto
its real owner left him one-view and unmeasurable here. Removing a wrong pairing is still right, but
count the players in each column before reading the percentiles.

Reads only. numpy 1 (smplx312).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.calibration.endzone_paint import _closest_points, _rays
from nfl_gsplat.pose.forward_kinematics import fk_forward, load_smplx_skeleton
from nfl_gsplat.pose.fuse_smplx import _pack_params
from nfl_gsplat.pose.keypoint_filter import reject_outliers
from nfl_gsplat.render.play_timeline import clip_offset

ANKLE_JOINTS = (15, 16)
ANKLE_CONF = 0.5
FEET_JOINTS = (7, 8)          # the SMPL-X ankles, whose midpoint is where the body stands
WORSE_M = 0.10                # a player this much worse is a regression, not noise


def ankle_pixels(kdf, cam: str) -> dict:
    """``{(frame, id): [n, 2]}`` confident ankle pixels."""
    out = {}
    for (f, p), g in kdf[kdf["cam"] == cam].groupby(["frame", "global_player_id"]):
        pts = []
        for j in ANKLE_JOINTS:
            r = g[g["joint"] == j]
            if len(r) == 1 and float(r["conf"].iloc[0]) >= ANKLE_CONF:
                pts.append(r[["x", "y"]].to_numpy(float)[0])
        if pts:
            out[(int(f), int(p))] = np.asarray(pts)
    return out


def records_by_id(path: Path) -> dict:
    blob = pickle.load(open(path, "rb"))
    per: dict = {}
    for f, recs in blob["frames"].items():
        for pid, rec in recs.items():
            per.setdefault(int(pid), {})[int(f)] = rec
    return per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--refit", required=True, type=Path, nargs="+",
                    help="one cache to measure, or two to compare (the second is the change)")
    ap.add_argument("--body-models", type=Path, default=Path("data/body_models"))
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--min-frames", type=int, default=15)
    args = ap.parse_args()
    P = args.play_dir
    other = "endzone" if args.cam == "sideline" else "sideline"
    if len(args.refit) > 2:
        raise SystemExit("give one cache, or two to compare")

    tracks = load_camera_track(P / "cameras.npz")
    ta, tb = tracks[args.cam], tracks[other]
    offset = clip_offset(P)
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    kdf, _ = reject_outliers(kdf)
    A, B = ankle_pixels(kdf, args.cam), ankle_pixels(kdf, other)
    caches = {p.stem: records_by_id(p) for p in args.refit}
    names = list(caches)

    ids = set.intersection(*[set(c) for c in caches.values()])
    pooled = {n: [] for n in names}
    rows = []
    for pid in sorted(ids):
        fw = {}
        for n in names:
            betas = np.asarray(next(iter(caches[n][pid].values()))["betas"], float)
            rest, _ = load_smplx_skeleton(args.body_models, betas=betas)
            fw[n] = fk_forward(rest)
        common = sorted(set.intersection(*[set(caches[n][pid]) for n in names]))
        out = {n: {"al": [], "ac": []} for n in names}
        for f in common:
            if (f, pid) not in A or (f + offset, pid) not in B:
                continue
            if f >= len(ta.conf) or ta.conf[f] <= 0:
                continue
            ef = f + offset
            if ef < 0 or ef >= len(tb.conf) or tb.conf[ef] <= 0:
                continue
            i1, p1 = ta.at(f)
            i2, p2 = tb.at(ef)
            pa, pb = A[(f, pid)], B[(ef, pid)]
            k = min(len(pa), len(pb))
            C1, d1 = _rays(i1.K(), np.asarray(p1.R, float), np.asarray(p1.t, float), pa[:k])
            C2, d2 = _rays(i2.K(), np.asarray(p2.R, float), np.asarray(p2.t, float), pb[:k])
            X, _m = _closest_points(C1, d1, C2, d2)
            tri = X[:, :2].mean(axis=0)
            centre = (-np.asarray(p1.R, float).T @ np.asarray(p1.t, float))[:2]
            u = tri - centre
            u = u / np.linalg.norm(u)
            for n in names:
                rec = caches[n][pid][f]
                J = fw[n](_pack_params(np.asarray(rec["body_pose"], float).reshape(-1),
                                       np.asarray(rec["global_orient"], float).reshape(3),
                                       np.asarray(rec["transl"], float).reshape(3)))
                d = J[list(FEET_JOINTS), :2].mean(axis=0) - tri
                a = float(d @ u)
                out[n]["al"].append(abs(a))
                out[n]["ac"].append(float(np.linalg.norm(d - a * u)))
        n_f = len(out[names[0]]["al"])
        if n_f >= args.min_frames:
            rows.append((pid, n_f, {n: (float(np.median(out[n]["al"])), float(np.median(out[n]["ac"]))) for n in names}))
            for n in names:
                pooled[n].extend(out[n]["al"])

    if not rows:
        raise SystemExit("no player has enough frames both cameras see")
    head = "  id  frames | " + " | ".join(f"{n[:22]:>22s}" for n in names)
    print(f"gap from the point both cameras put the feet, metres ({len(rows)} players)")
    print(head)
    print("  " + " " * 12 + "   " + "   ".join("along  across        " for _ in names))
    worse = []
    for pid, n_f, per in sorted(rows, key=lambda r: -r[2][names[-1]][0]):
        cells = " | ".join(f"{per[n][0]:9.2f} {per[n][1]:8.2f}" for n in names)
        flag = ""
        if len(names) == 2:
            ch = per[names[1]][0] - per[names[0]][0]
            if ch > WORSE_M:
                flag = f"  WORSE by {ch:+.2f} m"
                worse.append((pid, round(ch, 2)))
            elif ch < -WORSE_M:
                flag = f"  better by {ch:+.2f} m"
        print(f"  {pid:3d} {n_f:6d} | {cells}{flag}")
    for n in names:
        a = np.array(pooled[n])
        print(f"  pooled {n}: |along| p50 {np.median(a):.2f} m, p90 {np.percentile(a, 90):.2f}, "
              f"p99 {np.percentile(a, 99):.2f}, max {a.max():.2f}  ({len(a)} body-frames)")
    if len(names) == 2:
        print(f"  players worse by more than {WORSE_M:.2f} m: {len(worse)} of {len(rows)} -> {worse}")


if __name__ == "__main__":
    main()
