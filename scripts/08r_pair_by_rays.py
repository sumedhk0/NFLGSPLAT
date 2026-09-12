#!/usr/bin/env python
"""Propose cross-camera id joins from the two cameras' ankle rays. Proposes only; merges nothing.

    python scripts/08r_pair_by_rays.py --play-dir P [--out pair_proposal.json]

WHY. The body census is short of eleven a side at the snap not because players are missed but because
the two cameras hold the same man under two global ids: at frame 300 the sideline has 21 ids and the
endzone 24 and only 12 are shared, while the two pooled see Kansas City in 11-12 distinct places, which
is the right number. So the repair is to JOIN ids, not to admit more bodies.

HOW, AND WHY IT IS SHAPED THIS WAY (play 1, 2026-09-12, every step measured):

  - A per-frame ray test cannot pick the partner. Of 137 (frame, endzone id) tests only 10 gave a best
    partner within 0.30 m that was also 0.30 m clear of the runner-up; 113 were ambiguous with the
    runner-up 0.03-0.05 m behind. At ~100 m the rays to a neighbour standing a metre away miss by almost
    as little as the rays to the right man. The 0.20 m against 1.11 m separation that twins relies on
    came from a grossly mis-paired track and does NOT generalise to a formation.

  - Accumulating per-frame assignments as VOTES was tried and is worse than it looks. It gave 14 of 28
    ids a partner winning 60 % of their shared frames, but when the true partner is undetected on a
    frame the assignment must still give the id to somebody, so a distant body absorbs the vote: it
    proposed welding a referee onto a player 14.24 m away at 91 % confidence, and two bodies 6.44 m
    apart at 74 %.

  - What works is ONE global assignment on per-pair MEDIANS -- the median ray miss over every frame the
    two ids share, which is stable where a single frame is not. Validated by what it leaves alone: it
    reproduces all 14 pairings already known good (30->30 at 0.04 m, 2->2 at 0.05, 12->12, 5->5, 3->3,
    15->15, 82->82 ...) and independently corroborates candidates that two other methods found
    (7->89 at 0.08 m over 184 frames, 4->45, 25->33, 0->102, 80->22, 28->21).

  - But a one-to-one matching must assign EVERYONE, so a good re-pairing displaces a chain of others
    into worse slots (22->38 at 0.45 m against its current 0.22, 38->54 at 0.16 against 0.10). The
    assignment is therefore a candidate GENERATOR and the gate below is what decides: a candidate must
    beat the pairing it would replace on BOTH rulers, and the turf gap caps it outright.

A proposal is classified twice over, and the two must not be confused.

  kind      ADDITION when the target holds no track in the other camera on those frames, RE-PAIRING
            when it does and one of the two has to be given up. Re-pairings need --allow-repairing.
  apply_as  RELABEL when the two ids also both exist in OUR camera (they are different men here), so
            the join must rename the other camera's track onto the target. UNION otherwise. An id
            spans both cameras, so unioning where a relabel is needed would sweep this camera's other
            man along with it -- and a union that puts two tracks of one camera under one id at the
            same time is what pair_by_appearance.global_ids_checked refuses outright.

Getting that distinction wrong costs real candidates: classifying on the own-camera overlap rejected
`sideline 11 <- endzone 19` (0.14 m over 150 frames) for "370 sideline frames collide" when its
endzone collision was zero, along with `28 <- 21` (0.08 m over 138 frames).

EVERY PROPOSAL CARRIES THE INTERVAL IT IS EVIDENCED OVER (`frame_from`, `frame_to`). A join is chosen on
the frames where both ids have confident ankles and would otherwise be applied to the track's whole span,
where nothing supports it. Play 1's two v37 re-pairings hold for one long run each and then break
cleanly -- id 7 at 0.08 m over frames 140-494 jumping to 1.40 m from 498, id 25 at 0.13 m over 400-486
jumping to 1.81 m from 518 -- and they break at the SAME moment, around frame 495-500, which is when the
play ends and the field fills with players converging on the ball. Applied past its interval, id 25's
join left an endzone p90 of 483 px while its median improved sixfold. Only 2 of 16 pairings vary over
time at all, and both are freshly re-paired ones, so this is a trim on new joins rather than a property
of the tracker.

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
UNPAIRED = 9.0            # the assignment needs a finite cost for every pair; this is "not a pair"


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


def ray_miss(cam_a, cam_b, pa, pb):
    """How far the two cameras' ankle rays miss each other, metres."""
    n = min(len(pa), len(pb))
    if n == 0:
        return None
    C1, d1 = _rays(*cam_a, pa[:n])
    C2, d2 = _rays(*cam_b, pb[:n])
    _X, m = _closest_points(C1, d1, C2, d2)
    return float(np.median(m))


def pair_medians(A, B, gs, ge, tracks, cam: str, other: str, offset: int, teams: dict, frames,
                 *, stride: int = 2) -> dict:
    """``{(own id, other id): (ray miss p50, turf gap p50, frames shared)}`` over every frame the two
    appear together. Medians, not per-frame decisions: a single frame cannot tell neighbours apart."""
    miss: dict = {}          # {(own, other): [(frame, ray miss)]} -- the frames matter, see agreeing_interval
    turf: dict = {}
    for f in frames[::stride]:
        ca, cb = cam_of(tracks[cam], f), cam_of(tracks[other], f + offset)
        if ca is None or cb is None:
            continue
        S, E = A.get(int(f), {}), B.get(int(f) + offset, {})
        for s, pa in S.items():
            ts = teams.get(s)
            for e, pb in E.items():
                te = teams.get(e)
                if ts is not None and te is not None and ts != te:
                    continue                       # one man cannot be on both teams
                m = ray_miss(ca, cb, pa, pb)
                if m is not None:
                    miss.setdefault((s, e), []).append((int(f), m))
        # The turf gap is measured over EVERY frame both ids stand somewhere, not only the
        # ankle-confident frames the rays need -- that subset is exactly where a pairing looks its
        # best. Measured on it, `37 <- 139` reported 0.58 m and passed the gate while the two
        # actually stand 2.69 m apart across their shared span, which is two different men.
        GS, GE = gs.get(int(f), {}), ge.get(int(f) + offset, {})
        for s, a in GS.items():
            ts = teams.get(s)
            for e, b in GE.items():
                te = teams.get(e)
                if ts is not None and te is not None and ts != te:
                    continue
                turf.setdefault((s, e), []).append(
                    float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))))
    out = {}
    series = {}
    for key, v in miss.items():
        g = turf.get(key, [])
        vals = [m for _f, m in v]
        out[key] = (float(np.median(vals)), float(np.median(g)) if g else None, len(vals))
        series[key] = sorted(v)
    return out, series


def agreeing_interval(samples, max_miss: float):
    """``(first frame, last frame, samples)`` of the longest unbroken run whose ray miss is within
    ``max_miss``, or None. ``samples`` is ``[(frame, miss)]``, sorted.

    WHY. A join is chosen on the frames where both ids carry confident ankles and then applied to the
    track's WHOLE span, so its edges rest on no evidence at all. Play 1: both tracks re-paired in v37
    hold for one long run and then break cleanly -- id 7 sits at 0.08 m over frames 140-494 and jumps to
    1.40 m from 498, id 25 at 0.13 m over 400-486 and jumps to 1.81 m from 518. They break at the same
    moment, around frame 495-500, which is when the play ends and the field fills: the endzone track
    wanders onto one of the players converging on the ball. Applying a join beyond its interval is what
    left id 25 with an endzone p90 of 483 px while its median improved sixfold."""
    best = None
    run: list = []
    for f, m in samples:
        if m <= max_miss:
            run.append(f)
            continue
        if run and (best is None or len(run) > best[2]):
            best = (run[0], run[-1], len(run))
        run = []
    if run and (best is None or len(run) > best[2]):
        best = (run[0], run[-1], len(run))
    return best


def assign(stats: dict, *, min_frames: int, max_turf: float) -> list:
    """One global one-to-one assignment minimising the total ray miss. Returns the finite pairs only.
    It must match everyone it can, so its output is candidates, never decisions -- see the gate."""
    from scipy.optimize import linear_sum_assignment

    own = sorted({s for s, _e in stats})
    oth = sorted({e for _s, e in stats})
    C = np.full((len(own), len(oth)), UNPAIRED)
    for i, s in enumerate(own):
        for j, e in enumerate(oth):
            v = stats.get((s, e))
            if v and v[2] >= min_frames and v[1] is not None and v[1] <= max_turf:
                C[i, j] = v[0]
    r, c = linear_sum_assignment(C)
    return [(own[i], oth[j]) for i, j in zip(r, c) if C[i, j] < UNPAIRED]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--cam", default="sideline")
    ap.add_argument("--out", type=Path, default=None, help="default <play-dir>/pair_proposal.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=20, help="two ids must share at least this many frames")
    ap.add_argument("--max-ground-gap", type=float, default=1.0,
                    help="the GATE: two ids this far apart on the turf are not one man, whatever the "
                         "assignment says (a vote once proposed a pair 14.24 m apart)")
    ap.add_argument("--max-ray-miss", type=float, default=0.30)
    ap.add_argument("--allow-repairing", action="store_true",
                    help="also propose joins that collide with an existing cross-camera pairing. Applying one "
                         "of those means UNPAIRING the target's current other-camera track first, not adding a "
                         "view to it, because a union may not put two tracks of one camera under one id at the "
                         "same time (pair_by_appearance.global_ids_checked refuses it)")
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

    stats, series = pair_medians(A, B, gs, ge, tracks, args.cam, other, offset, teams, frames,
                                 stride=args.stride)
    pairs = assign(stats, min_frames=args.min_frames, max_turf=args.max_ground_gap)
    kept = sum(1 for s, e in pairs if s == e)
    print(f"{P.name}: one global assignment on per-pair medians over {len(frames[::args.stride])} frames; "
          f"{len(pairs)} pairs, {kept} of them the id the pipeline already uses")

    # every frame a track exists on, to tell an ADDITION from a RE-PAIRING
    spans = {(c, int(p)): set(int(x) for x in g["frame"].unique())
             for (c, p), g in df.groupby(["cam", "global_player_id"])}

    proposals = []
    for s, e in pairs:
        if s == e:
            continue
        pm, pg, pn = stats[(s, e)]
        cm, cg, cn = stats.get((s, s), (None, None, 0))
        # where the join is actually evidenced, rather than the whole span it could be applied to
        iv = agreeing_interval(series.get((s, e), []), args.max_ray_miss)
        ov_other = len(spans.get((other, e), set()) & spans.get((other, s), set()))
        ov_own = len(spans.get((args.cam, s), set()) & spans.get((args.cam, e), set()))
        # Only a collision in the OTHER camera makes this a re-pairing: id s already holds a track there
        # on the same frames, so one of the two has to be given up. A collision in our OWN camera means
        # something different -- that s and e are two distinct men in this view -- and it does not block
        # the join, it only dictates HOW: relabel the other camera's track to s, never union the global
        # ids, because an id spans both cameras and a union would sweep this camera's id e along with it.
        apply_as = "relabel" if ov_own else "union"
        kind = "addition" if not ov_other else "re-pairing"
        why = None
        if kind == "re-pairing" and not args.allow_repairing:
            why = (f"a re-pairing, not an addition: {other} {s} already holds a track on {ov_other} of these "
                   f"frames, so applying it means giving that one up first")
        elif pg is None or pm is None:
            why = "the proposed pair never co-occurs"
        elif pg > args.max_ground_gap:
            why = f"the two stand {pg:.2f} m apart on the turf"
        elif pm > args.max_ray_miss:
            why = f"the rays miss by {pm:.2f} m"
        elif cn and cm is not None and cg is not None and not (pm < cm and pg < cg):
            why = (f"it does not beat the pairing it would replace on both rulers "
                   f"(rays {pm:.2f} vs {cm:.2f} m, turf {pg:.2f} vs {cg:.2f} m)")
        row = {"own_id": int(s), "other_id": int(e), "kind": kind, "apply_as": apply_as,
               "frame_from": None if iv is None else int(iv[0]), "frame_to": None if iv is None else int(iv[1]),
               "interval_samples": 0 if iv is None else int(iv[2]),
               "collide_other": int(ov_other), "collide_own": int(ov_own), "frames": int(pn),
               "ray_miss_m": None if pm is None else round(pm, 3),
               "ground_gap_m": None if pg is None else round(pg, 3),
               "current_ray_miss_m": None if cm is None else round(cm, 3),
               "current_ground_gap_m": None if cg is None else round(cg, 3),
               "current_frames": int(cn), "rejected": why}
        proposals.append(row)
        mark = "PROPOSE" if why is None else "reject "
        span = "" if iv is None else f", agrees {iv[0]}-{iv[1]} ({iv[2]}/{pn} samples)"
        print(f"  {mark} {args.cam} {s:3d} <- {other} {e:3d}  rays {pm:.2f} m, turf "
              f"{'--' if pg is None else format(pg, '.2f')} m over {pn:3d} frames{span}"
              + (f", apply by {apply_as}" if why is None else f"   [{why}]"))
    keep = [p for p in proposals if p["rejected"] is None]
    out = args.out or P / "pair_proposal.json"
    out.write_text(json.dumps({"cam": args.cam, "other": other, "offset": int(offset),
                               "max_ground_gap_m": args.max_ground_gap, "max_ray_miss_m": args.max_ray_miss,
                               "unchanged_pairs": kept, "proposals": proposals}, indent=2))
    print(f"{len(keep)} of {len(proposals)} candidates survive the gate; wrote {out} (nothing merged)")


if __name__ == "__main__":
    main()
