#!/usr/bin/env python
"""Fit the per-play joint in-filler (pose.infill) on a play's own pose caches and score it on held-out sure joints
against today's SLERP fill; with --write, fill the unsure rows and save them for the timeline.

USAGE (smplx312 venv: the pose caches are numpy-1 pickles):
  python scripts/09f_infill.py --play-dir P [--lo 395 --hi 607] [--every 5] [--epochs 60] [--write]

The sure rows: the row's driving keypoint (timeline.CONF_GATE_KEYPOINT: the elbow vouches for the shoulder's rotation,
the wrist for the elbow's, ...) with confidence >= SURE_CONF in the best camera. Records the loader would drop (no box
for that id on that frame; a merged box) are dropped here too, so the model never sees them as context.
Writes: <play-dir>/infill_poses.pkl ({pid: {frame: body_pose[21, 3]}}, the unsure rows replaced) and infill_net.pt.
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
from nfl_gsplat.pose import infill as inf  # noqa: E402
from nfl_gsplat.render import timeline as tlm  # noqa: E402
from nfl_gsplat.render.play_timeline import clip_offset, keypoint_confidence  # noqa: E402


def load_sequences(P: Path, *, lo: int | None, hi: int | None) -> list[inf.Sequence]:
    refit_path, side_path = P / "poses_refit.json", P / "poses_sideline.json"
    refit = pickle.load(open(refit_path, "rb"))["frames"] if refit_path.exists() else {}
    side = pickle.load(open(side_path, "rb")) if side_path.exists() else None
    poses: dict = {}
    for f, recs in refit.items():
        for pid, r in recs.items():
            poses.setdefault(int(pid), {})[int(f)] = np.asarray(r["body_pose"], float).reshape(21, 3)
    if side is not None:
        for f, recs in side["frames"].items():
            for pid, r in recs.items():
                poses.setdefault(int(pid), {}).setdefault(int(f), np.asarray(r["body_pose"], float).reshape(21, 3))
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0].copy()
    offset = clip_offset(P)
    if offset and "endzone" in set(df["cam"].unique()):
        df.loc[df["cam"] == "endzone", "frame"] = df.loc[df["cam"] == "endzone", "frame"].astype(int) - offset
    boxed = set(zip(df["frame"].astype(int).tolist(), df["global_player_id"].astype(int).tolist()))
    poses, n_unboxed = tlm.drop_unboxed_poses(poses, boxed)
    merged = tlm.merged_box_frames(df, cam="sideline")
    keep = {(int(f), int(p)) for p, recs in poses.items() for f in recs} - merged
    poses, n_merged = tlm.drop_unboxed_poses(poses, keep)
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    if offset:
        kdf = kdf.copy()
        kdf.loc[kdf["cam"] == "endzone", "frame"] = kdf.loc[kdf["cam"] == "endzone", "frame"].astype(int) - offset
    conf_by = keypoint_confidence(kdf, cam=None)
    seqs = []
    for pid, recs in sorted(poses.items()):
        frames = sorted(f for f in recs if (lo is None or f >= lo) and (hi is None or f <= hi))
        if len(frames) < 2 * inf.HALF + 1:
            continue
        pose = np.stack([recs[f] for f in frames])
        conf = np.stack([conf_by.get(pid, {}).get(f, np.full(21, np.nan)) for f in frames])
        seqs.append(inf.Sequence(pid, np.asarray(frames), pose, conf))
    print(f"sequences: {len(seqs)} players, {sum(len(s.frames) for s in seqs)} keyframes "
          f"({n_unboxed} unboxed and {n_merged} merged-box records dropped first)")
    sure = sum(int(inf.sure_mask(s.conf).sum()) for s in seqs)
    unsure = sum(int(inf.unsure_mask(s.conf).sum()) for s in seqs)
    total = sum(s.conf.size for s in seqs)
    print(f"rows: {total} total, {sure} sure (conf >= {inf.SURE_CONF}), {unsure} unsure (< {inf.UNSURE_CONF}), the rest in between or unknown")
    runs = []
    for s in seqs:
        u = inf.unsure_mask(s.conf)
        for j in range(inf.J):
            k = 0
            while k < len(s.frames):
                if u[k, j]:
                    a = k
                    while k < len(s.frames) and u[k, j]:
                        k += 1
                    runs.append(k - a)
                else:
                    k += 1
    if runs:
        r = np.asarray(runs)
        print(f"unsure runs: {len(r)}, keyframes per run p50 {np.percentile(r, 50):.0f} p90 {np.percentile(r, 90):.0f} max {r.max()}; "
              f"rows by limb: " + ", ".join(f"{name} {int(sum(inf.unsure_mask(s.conf)[:, rows].sum() for s in seqs))}" for name, rows in inf.LIMBS.items()))
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--lo", type=int, default=None)
    ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--every", type=int, default=5, help="hold out every n-th keyframe's sure rows")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true", help="fill the unsure rows with a model trained on all sure rows and save")
    a = ap.parse_args()
    P = Path(a.play_dir)
    seqs = load_sequences(P, lo=a.lo, hi=a.hi)
    res = inf.holdout_score(seqs, every=a.every, seed=a.seed, epochs=a.epochs, width=a.width)
    print(f"held-out sure rows: {res['n']}  (best epoch {res['best_epoch']}; 0 = the zero correction, i.e. SLERP)")
    print(f"  model  mean {res['model_mean']:.2f} deg  p90 {res['model_p90']:.2f}")
    print(f"  slerp  mean {res['slerp_mean']:.2f} deg  p90 {res['slerp_p90']:.2f}   (today's fill)")
    for name, g in res["by_group"].items():
        print(f"  {name:6s} model {g['model_mean']:.2f}  slerp {g['slerp_mean']:.2f}  (n {g['n']})")
    report = {k: v for k, v in res.items() if k != "net"}
    (P / "infill_report.json").write_text(json.dumps(report, indent=1))
    if not a.write:
        return
    import torch

    net = inf.train(inf.build_windows(seqs, inf.run_masks(seqs, seed=a.seed)), epochs=a.epochs, width=a.width, seed=a.seed)
    filled = inf.fill(seqs, net)
    n_rows = sum(int(inf.unsure_mask(s.conf).sum()) for s in seqs)
    with open(P / "infill_poses.pkl", "wb") as fh:
        pickle.dump({int(p): {int(f): np.asarray(v, np.float32) for f, v in recs.items()} for p, recs in filled.items()}, fh)
    torch.save(net.state_dict(), P / "infill_net.pt")
    print(f"wrote infill_poses.pkl ({n_rows} unsure rows filled over {sum(len(v) for v in filled.values())} keyframes) and infill_net.pt")


if __name__ == "__main__":
    main()
