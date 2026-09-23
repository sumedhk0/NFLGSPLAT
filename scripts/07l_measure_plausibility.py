"""Score a play's drawn timeline on the plausibility rulers (render.motion_rulers) and write the report.

    PYS scripts/07l_measure_plausibility.py --play-dir P --tag v40 [--lo 300 --hi 460] [--joints]

The timeline is built exactly as 05k renders it (play_timeline.load_play_timeline with the same model),
so every number is about what the user sees, not about the tracks. ``--joints`` adds the limb rulers:
one forward pass per drawn body-frame in the live window (~40 s on CPU for a play).

Why this exists: the user's complaints -- "players teleport", "movement is jittery", "a man appears in
the other team's line" -- were each invisible to the census, and the first two were dismissed as noise
by a probe that printed 48 m/s and moved on. This script prints the same table for every version, so
v38 vs v39 vs v40 is one diff: steps over 0.25 / 0.6 m per frame, root jitter percentiles, census on the
live window and the whole clip, worst ids by each ruler, and (with --joints) limb jitter and speed.

Report goes to ``--out`` (default DIAG/<play>_<tag>_plausibility.json); nothing under data/ is written.
Runs under the smplx venv (PYS): it needs the body model, and the pose caches are numpy-1 pickles.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from nfl_gsplat.render import motion_rulers as mr
from nfl_gsplat.render import play_timeline as pt

BODY = "C:/Users/sumedh/NFLGSPLAT/data/body_models"


def team_of(P: Path) -> dict:
    p = P / "identity_resolved.pkl"
    if not p.exists():
        return {}
    ident = pickle.load(open(p, "rb")).get("merged", {})
    return {int(k): getattr(v, "team", None) for k, v in ident.items()}


def joints_for(tl, model, lo, hi):
    import torch

    out: dict = {}
    for f in tl.frames:
        if not (lo <= f <= hi):
            continue
        for s in tl.states.get(f, ()):
            with torch.no_grad():
                res = model(betas=torch.tensor(s.betas[None, :10].astype(np.float32)),
                            body_pose=torch.tensor(s.body_pose.reshape(1, -1).astype(np.float32)),
                            global_orient=torch.tensor(s.global_orient.reshape(1, 3).astype(np.float32)))
            j = res.joints[0, :22].numpy().astype(np.float64)
            out.setdefault(int(s.pid), {})[int(f)] = j - j[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True)
    ap.add_argument("--tag", required=True, help="version label for the report file, e.g. v40")
    ap.add_argument("--lo", type=int, default=None, help="live window start (default: play_end.json's snap, else 300)")
    ap.add_argument("--hi", type=int, default=None, help="live window end (default: play_end.json's end, else 460)")
    ap.add_argument("--joints", action="store_true", help="also score the limbs (one forward pass per body-frame)")
    ap.add_argument("--gait", action="store_true", help="score the timeline with the running gait applied (as 05k --gait draws it)")
    ap.add_argument("--body-models", default=BODY)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.lo is None or args.hi is None:
        import json

        pe = Path(args.play_dir) / "play_end.json"
        d = json.loads(pe.read_text()) if pe.exists() else {}
        if args.lo is None:
            args.lo = int(d["snap"]) if d.get("snap") is not None else 300
        if args.hi is None:
            args.hi = int(d["end"]) if d.get("end") is not None else 460
        print(f"live window {args.lo}-{args.hi}" + (" from play_end.json" if d else " (defaults; no play_end.json)"))

    import smplx

    P = Path(args.play_dir)
    model = smplx.create(args.body_models, model_type="smplx", gender="neutral", num_betas=10,
                         use_pca=False, batch_size=1)
    tl, *_ = pt.load_play_timeline(P, model)
    if args.gait:
        from nfl_gsplat.render.gait import gait_timeline

        reps = gait_timeline(tl)
        print(f"gait applied: legs synthesised on {sum(r['on'] for r in reps.values())} body-frames")
        from nfl_gsplat.render import foot_lock as _fl

        if _fl.MODE != "off":                                  # as 05k draws it (read at call time)
            lreps = _fl.foot_lock_timeline(tl, str(args.body_models), team_of=team_of(P))
            print(f"foot lock ({_fl.MODE}): {sum(r['segments'] for r in lreps.values())} stances on "
                  f"{sum(r['frames'] for r in lreps.values())} body-frames")
    pos = mr.positions_by_id(tl.states)
    joints = joints_for(tl, model, args.lo, args.hi) if args.joints else None
    rep = mr.summarize(pos, tl.states, team_of(P), lo=args.lo, hi=args.hi, joints_by_id=joints)
    live_bp = [np.asarray(s.body_pose, float).reshape(21, 3) for f in tl.frames if args.lo <= f <= args.hi
               for s in tl.states.get(f, ())]
    rep["joint_limits"] = mr.hinge_violations(np.stack(live_bp) if live_bp else np.zeros((0, 21, 3)))
    rep["tag"] = args.tag
    rep["play"] = P.name

    st, rj, ce = rep["steps"], rep["root_jitter"], rep["census"]
    print(f"\n{P.name} {args.tag}: {rep['ids_drawn']} ids, {rep['body_frames']} body-frames")
    print(f"steps  > {mr.STEP_M} m/frame: full {st['full_over_step']} (handovers {st['full_handover']})  "
          f"live {st['live_over_step']} (handovers {st['live_handover']})   "
          f"> {mr.STEP_HARD_M}: full {st['full_over_hard']}  live {st['live_over_hard']}")
    print("       worst: " + ", ".join(f"id {w['pid']} {w['m']:.2f} m @{w['frame']}{' (handover)' if w['handover'] else ''}"
                                       for w in st["worst"][:6]))
    print("       worst live: " + ", ".join(f"id {w['pid']} {w['m']:.2f} m @{w['frame']}{' (handover)' if w['handover'] else ''}"
                                            for w in st["worst_live"][:6]))
    hp = rep["hops"]
    print(f"hops   (a step > {mr.JERK_EXCESS_M} m beyond its neighbours' median): full {hp['full']}  live {hp['live']}")
    if hp["worst_live"]:
        print("       worst live: " + ", ".join(f"id {w['pid']} {w['step_m']:.2f} m (+{w['excess_m']:.2f}) @{w['frame']}"
                                                for w in hp["worst_live"][:6]))
    if "skating" in rep:
        sk = rep["skating"]
        print(f"skating (moving bodies, slower ankle / pelvis speed): p50 {sk['ratio_p50']:.2f}, planted (< {mr.SKATE_PLANTED}) "
              f"{100 * sk['planted']:.0f} % of {sk['n']} frames   (a runner plants ~50 %)")
    print(f"root jitter m/frame^2  full p50 {rj['full']['p50']:.4f} p90 {rj['full']['p90']:.4f} p99 {rj['full']['p99']:.4f}"
          f"   live p50 {rj['live']['p50']:.4f} p90 {rj['live']['p90']:.4f} p99 {rj['live']['p99']:.4f}")
    print("       worst live: " + ", ".join(f"id {w['pid']} {w['p90']:.3f}" for w in rj["worst_live"][:6]))
    print(f"census |KC-11|+|BAL-11|  live {ce['live']:.2f} ({ce['live_teams']})   full {ce['full']:.2f}")
    jl = rep["joint_limits"]
    print(f"hinges (live, {jl['n']} body-frames): hyperextended {100 * jl['hyperextended']:.1f}%  "
          f"off-axis {100 * jl['off_axis']:.1f}%   (share of hinge-frames; a knee does neither)")
    if "joints" in rep:
        jo = rep["joints"]
        print(f"joints ({jo['ids']} ids, live): jitter p50 {jo['jitter']['p50']:.4f} p90 {jo['jitter']['p90']:.4f} "
              f"p99 {jo['jitter']['p99']:.4f}   speed p50 {jo['speed']['p50']:.4f} p90 {jo['speed']['p90']:.4f}")
        print("       worst: " + ", ".join(f"id {w['pid']} {w['jitter_p90']:.3f}" for w in jo["worst"][:6]))

    out = args.out
    if out is None:
        diag = os.environ.get("DIAG", "C:/Users/sumedh/diag")
        out = Path(diag) / f"{P.name}_{args.tag}_plausibility.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, default=float))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
