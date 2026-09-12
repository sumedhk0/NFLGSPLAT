#!/usr/bin/env python
"""Propose cross-camera id joins from the two cameras' ankle rays. Proposes only; merges nothing.

    python scripts/08r_pair_by_rays.py --play-dir P [--out pair_proposal.json]

WHY. The body census is short of eleven a side at the snap not because players are missed but because
the two cameras hold the same man under two global ids: at frame 300 the sideline has 21 ids and the
endzone 24 and only 12 are shared, and 5-7 of the endzone-only ids a frame stand 0.14-0.70 m from a
sideline body. Pooled, the two cameras see KC in 11-12 distinct places -- the right number -- so the
repair is to JOIN ids, not to admit more bodies.

WHAT THE MEASUREMENTS SAY THIS MUST LOOK LIKE (play 1, 2026-09-12):

  - A per-frame ray test cannot do it. Of 137 (frame, endzone id) tests only 10 gave a best partner
    within 0.30 m that was also 0.30 m clear of the runner-up; 113 were ambiguous, the runner-up
    typically 0.03-0.05 m behind. At ~100 m the rays to a neighbour a metre away miss by almost as
    little as the rays to the right man. The 0.20 m against 1.11 m separation that twins uses came from
    a grossly mis-paired track and does NOT generalise to picking one man out of a formation.

  - A one-to-one assignment accumulated over the play does generate the right candidates -- 14 of 28
    ids get a partner winning 60 % of the frames they share, seven of them joins the pipeline does not
    make -- but it CANNOT be trusted as a decision. Verified against the pairing each would replace,
    two of those seven were flatly wrong (endzone 5 -> sideline 19 puts two bodies 6.44 m apart under
    one id) and two more had the ray test and the ground gap disagreeing. When the true partner is
    undetected on a frame the assignment must still give the id to somebody, and a distant body
    absorbs the vote.

So the vote is a candidate GENERATOR and the ground gap is the GATE, and a candidate must beat the
pairing it would replace on BOTH rulers before it is proposed. Survivors on play 1 were 45->4, 89->7
(0.08 m over 179 frames, a man the two cameras have never held under one id) and 101->5.

This writes a proposal and changes nothing: a wrong merge costs more than a merge not made (the 0.35 m
twin threshold folded a passer into the lineman beside him), so applying is a separate, measured step.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.calibration.endzone_paint import _closest_points, _rays
from nfl_gsplat.pose.keypoint_filter import reject_outliers
from nfl_gsplat.render.play_timeline import ankle_ground, clip_offset, ground_positions

ANKLE_JOINTS = (15, 16)
ANKLE_CONF = 0.5
COST_CAP_M = 5.0          # the assignment needs a finite cost for every pair
ACCEPT_M = 0.5            # a frame's assignment only votes if the rays come this close


def ankle_points(kdf, cam: str) -> dict:
    """``{frame: {id: [n, 2] ankle pixels}}`` for the confident ankles of one camera."""
    out: dict = {}
    for (f, p), g in kdf[kdf["cam"] == cam].groupby(["frame", "global_player_id"]):
        pts = []
        for j in ANKLE_JOINTS:
            r = g[g["joint"] == j]
            if len(r) == 1 and float(r["conf"].iloc[0]) >= ANKLE_CONF:
                pts.append(r[["x", "y"]].to_numpy(float)[0])
        if pts:
            out.setdefault(int(f), {})[int(p)] = np.asarray(pts)
    return out


def cam_of(track, f):
    """``(K, R, t)`` for a frame the camera actually solved, else None."""
    if f < 0 or f >= len(track.conf) or track.conf[f] <= 0:
        return None
    intr, pose = track.at(int(f))
    return intr.K(), np.asarray(pose.R, float), np.asarray(pose.t, float)


def ray_miss(cam_a, cam_b, pa, pb) -> float | None:
    """How far the two cameras' ankle rays miss each other, metres."""
    n = min(len(pa), len(pb))
    if n == 0:
        return None
    C1, d1 = _rays(*cam_a, pa[:n])
    C2, d2 = _rays(*cam_b, pb[:n])
    _X, m = _closest_points(C1, d1, C2, d2)
    return float(np.median(m))


def vote(A, B, tracks, cam: str, other: str, offset: int, teams: dict, frames, *, stride: int = 2):
    """``(votes, co_present)``: how often a one-to-one assignment by ray miss joins each (own, other)
    pair, and how often both were on the field to be joined. Cross-team pairs are never assigned."""
    from scipy.optimize import linear_sum_assignment

    votes: dict = {}
    co: dict = {}
    used = 0
    for f in frames[::stride]:
        ca = cam_of(tracks[cam], f)
        cb = cam_of(tracks[other], f + offset)
        if ca is None or cb is None:
            continue
        S, E = A.get(int(f), {}), B.get(int(f) + offset, {})
        if len(S) < 2 or len(E) < 2:
            continue
        sl, el = sorted(S), sorted(E)
        C = np.full((len(sl), len(el)), COST_CAP_M)
        for i, s in enumerate(sl):
            for j, e in enumerate(el):
                co[(s, e)] = co.get((s, e), 0) + 1
                ts, te = teams.get(s), teams.get(e)
                if ts is not None and te is not None and ts != te:
                    continue                      # one man cannot be on both teams
                m = ray_miss(ca, cb, S[s], E[e])
                if m is not None:
                    C[i, j] = min(m, COST_CAP_M)
        r, c = linear_sum_assignment(C)
        used += 1
        for i, j in zip(r, c):
            if C[i, j] <= ACCEPT_M:
                votes[(sl[i], el[j])] = votes.get((sl[i], el[j]), 0) + 1
    return votes, co, used


def pair_stats(A, B, gs, ge, tracks, cam, other, offset, s: int, e: int, frames):
    """``(ray miss p50, ground gap p50, n)`` for one (own id, other id) pair over the frames both appear."""
    miss, gap = [], []
    for f in frames:
        ca, cb = cam_of(tracks[cam], f), cam_of(tracks[other], f + offset)
        if ca is None or cb is None:
            continue
        pa = A.get(int(f), {}).get(s)
        pb = B.get(int(f) + offset, {}).get(e)
        if pa is not None and pb is not None:
            m = ray_miss(ca, cb, pa, pb)
            if m is not None:
                miss.append(m)
        a = gs.get(int(f), {}).get(s)
        b = ge.get(int(f) + offset, {}).get(e)
        if a is not None and b is not None:
            gap.append(float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))))
    return (float(np.median(miss)) if miss else None, float(np.median(gap)) if gap else None, len(miss))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--out", type=Path, default=None, help="default <play-dir>/pair_proposal.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-share", type=float, default=0.6, help="a candidate must win this fraction of the "
                                                                "frames the two ids share")
    ap.add_argument("--min-frames", type=int, default=20, help="and they must share at least this many")
    ap.add_argument("--max-ground-gap", type=float, default=1.0,
                    help="the GATE: two ids this far apart on the turf are not one man, whatever the vote "
                         "says (the vote proposed a pair 6.44 m apart on play 1)")
    ap.add_argument("--max-ray-miss", type=float, default=0.30)
    ap.add_argument("--allow-repairing", action="store_true",
                    help="also propose joins that collide with an existing cross-camera pairing. Applying one "
                         "of those means UNPAIRING the target's current other-camera track first, not adding a "
                         "view to it, because a union may not put two tracks of one camera under one id at the "
                         "same time (pair_by_appearance.global_ids_checked refuses it). Four of play 1's six "
                         "gated candidates are re-pairings of this kind")
    args = ap.parse_args()
    P = args.play_dir
    other = "endzone" if args.cam == "sideline" else "sideline"

    tracks = load_camera_track(P / "cameras.npz")
    if other not in tracks:
        raise SystemExit(f"{P} has no {other} camera: nothing to pair against")
    offset = clip_offset(P)
    kdf = pd.read_parquet(P / "keypoints_2d.parquet")
    kdf, _ = reject_outliers(kdf)
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    ident = pickle.load(open(P / "identity_resolved.pkl", "rb")).get("merged", {})
    teams = {int(k): getattr(v, "team", None) for k, v in ident.items()}
    ank = ankle_ground(kdf, tracks)
    gs = ground_positions(df[df["cam"] == args.cam], tracks, ankles=ank)
    ge = ground_positions(df[df["cam"] == other], tracks, ankles=ank)
    A, B = ankle_points(kdf, args.cam), ankle_points(kdf, other)
    frames = sorted(A)

    votes, co, used = vote(A, B, tracks, args.cam, other, offset, teams, frames, stride=args.stride)
    print(f"one-to-one assignment by ankle rays over {used} frames of {P.name}")

    by_other: dict = {}
    for (s, e), v in votes.items():
        n = co.get((s, e), 0)
        if n >= args.min_frames:
            by_other.setdefault(e, []).append((v / n, v, n, s))
    # every frame a track exists on, to tell an ADDITION from a RE-PAIRING: a union may not put two
    # tracks of one camera under one id at the same time, so a candidate whose target already holds an
    # other-camera track on the same frames can only be applied by unpairing that track first
    spans = {(c, int(p)): set(int(x) for x in g["frame"].unique())
             for (c, p), g in df.groupby(["cam", "global_player_id"])}

    proposals = []
    for e in sorted(by_other):
        share, v, n, s = sorted(by_other[e], reverse=True)[0]
        if share < args.min_share or s == e:
            continue
        pm, pg, pn = pair_stats(A, B, gs, ge, tracks, args.cam, other, offset, s, e, frames)
        cm, cg, cn = pair_stats(A, B, gs, ge, tracks, args.cam, other, offset, e, e, frames)
        ov_other = len(spans.get((other, e), set()) & spans.get((other, s), set()))
        ov_own = len(spans.get((args.cam, s), set()) & spans.get((args.cam, e), set()))
        kind = "addition" if not (ov_other or ov_own) else "re-pairing"
        why = None
        if kind == "re-pairing" and not args.allow_repairing:
            why = (f"a re-pairing, not an addition: {ov_other} {other} and {ov_own} {args.cam} frames collide, "
                   f"so applying it means unpairing {other} {s} first")
        if pg is None or pm is None:
            why = "the proposed pair never co-occurs"
        elif pg > args.max_ground_gap:
            why = f"the two stand {pg:.2f} m apart on the turf"
        elif pm > args.max_ray_miss:
            why = f"the rays miss by {pm:.2f} m"
        elif cn and cm is not None and cg is not None and not (pm < cm and pg < cg):
            why = (f"it does not beat the pairing it would replace on both rulers "
                   f"(rays {pm:.2f} vs {cm:.2f} m, turf {pg:.2f} vs {cg:.2f} m)")
        row = {"other_id": int(e), "own_id": int(s), "kind": kind, "collide_other": int(ov_other),
               "collide_own": int(ov_own), "share": round(float(share), 3), "frames": int(n),
               "ray_miss_m": None if pm is None else round(pm, 3),
               "ground_gap_m": None if pg is None else round(pg, 3),
               "current_ray_miss_m": None if cm is None else round(cm, 3),
               "current_ground_gap_m": None if cg is None else round(cg, 3),
               "current_frames": int(cn), "rejected": why}
        proposals.append(row)
        mark = "PROPOSE" if why is None else "reject "
        print(f"  {mark} {other} {e:3d} -> {args.cam} {s:3d}  share {share:4.0%} of {n:3d} frames, "
              f"rays {pm if pm is None else round(pm, 2)} m, turf {pg if pg is None else round(pg, 2)} m"
              + ("" if why is None else f"   [{why}]"))
    keep = [p for p in proposals if p["rejected"] is None]
    out = args.out or P / "pair_proposal.json"
    out.write_text(json.dumps({"cam": args.cam, "other": other, "offset": int(offset),
                               "min_share": args.min_share, "max_ground_gap_m": args.max_ground_gap,
                               "proposals": proposals}, indent=2))
    print(f"{len(keep)} of {len(proposals)} candidates survive the gate; wrote {out} (nothing merged)")


if __name__ == "__main__":
    main()
