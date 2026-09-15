"""Unpair the intervals where one camera's track under a global id is a DIFFERENT MAN from the other's.

    python scripts/08u_unpair_bad_runs.py --play-dir P            # dry run: print the plan
    python scripts/08u_unpair_bad_runs.py --play-dir P --apply    # split the tables, backups first

WHY. The teleports in play 1's v38 -- players snapping metres in a frame -- were cross-camera pairings that
are right for most of a track and wrong for a stretch of it: id 17 carried four sideline frames (457-460) of
a man 6.2 m from where its endzone track stood; id 82's endzone rows from frame 315 began 5.9 m from where
its sideline track ended; id 5's endzone sat 6.2 m from its context over 389-403. ``mispaired_ids`` sees none
of it because it gates on the whole-track MEDIAN, which a short excursion never moves. Applied to play 1
(2026-09-15): 13 intervals, 187 rows, contiguous impossible steps 22 -> 14.

RULE, in two parts, in :func:`plan_unpairings`:
  1. A BAD RUN is a stretch of overlap frames where the two cameras disagree by more than ``ctrl_m`` for at
     least ``min_run`` frames. ``ctrl_m`` is the cross-camera control's p90: correctly paired ids on play 1
     agree at p50 0.42 m, p90 0.95 m. Shorter runs are jitter.
  2. The INTRUDER camera is whichever one's positions inside the run are discontinuous with the id's own
     positions just OUTSIDE it. It must be displaced by at least ``min_intruder_m`` AND at least ``ratio``
     times further than the other camera -- without that margin 50 intervals fired on play 1, most of them
     coin flips at 0.6 m against 0.6 m, which would fragment good tracks. With it, every one of the 13 was
     unambiguous. The intruder's rows move to a FRESH id in that camera. Never deleted: it is a real man,
     just not this one.
  A handover with no overlap (one span ends, the other begins within ``handover_gap`` frames, more than
  ``ctrl_m`` apart) unpairs the later camera's span.

TEAM. Every fragment takes its team from its OWN ``kit_margin`` majority (play 1: BAL 0.09 / KC 0.88 share
positive, 96 % of rows finite) and never inherits the head's -- an intruder is by definition a different
man, possibly of the other team. Written into identity_resolved.pkl as a copy of the source identity with
the team replaced (``PlayerIdentity`` is frozen; ``dataclasses.replace``).

Both tracks.parquet and keypoints_2d.parquet are split identically, with the clip offset applied to endzone
rows (the file stores the endzone clip's own frame numbers). Fails loud if either table is missing.
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.calibration.cameras_io import load_camera_track  # noqa: E402
from nfl_gsplat.errors import SetupError  # noqa: E402
from nfl_gsplat.render.play_timeline import ankle_ground, clip_offset, ground_positions  # noqa: E402

CTRL_M: float = 0.95
MIN_RUN: int = 4
HANDOVER_GAP: int = 12
CONTEXT: int = 15
MIN_INTRUDER_M: float = 1.5
RATIO: float = 2.0


def per_id(ground) -> dict:
    """frame -> {pid: xy}  to  pid -> {frame: xy}"""
    out: dict = {}
    for f, d in ground.items():
        for pid, xy in d.items():
            out.setdefault(int(pid), {})[int(f)] = np.asarray(xy, float)
    return out


def runs_of(flags, frames, min_run: int):
    """(first, last) of every run of True at least ``min_run`` long."""
    out, cur = [], []
    for f, b in zip(frames, flags):
        if b:
            cur.append(f)
        else:
            if len(cur) >= min_run:
                out.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= min_run:
        out.append((cur[0], cur[-1]))
    return out


def plan_unpairings(S: dict, E: dict, *, ctrl_m: float = CTRL_M, min_run: int = MIN_RUN,
                    handover_gap: int = HANDOVER_GAP, context: int = CONTEXT,
                    min_intruder_m: float = MIN_INTRUDER_M, ratio: float = RATIO) -> list:
    """``[(pid, intruder_cam, lo, hi, why)]`` over ``S``/``E`` = pid -> {frame: xy} per camera, frames
    already on one clock. Pure: no I/O, so it is the thing the tests exercise."""
    plan = []
    for pid in sorted(set(S) | set(E)):
        s, e = S.get(pid, {}), E.get(pid, {})
        both = sorted(set(s) & set(e))
        if both:
            flags = [float(np.linalg.norm(e[f] - s[f])) > ctrl_m for f in both]
            for lo, hi in runs_of(flags, both, min_run):
                inside = set(range(lo, hi + 1))
                ctx = [s[f] for f in s if lo - context <= f <= hi + context and f not in inside]
                ctx += [e[f] for f in e if lo - context <= f <= hi + context and f not in inside]
                if not ctx:
                    continue
                ref = np.median(np.stack(ctx), axis=0)
                in_s = np.median(np.stack([s[f] for f in both if lo <= f <= hi]), axis=0)
                in_e = np.median(np.stack([e[f] for f in both if lo <= f <= hi]), axis=0)
                ds, de = float(np.linalg.norm(in_s - ref)), float(np.linalg.norm(in_e - ref))
                hi_d, lo_d = max(ds, de), min(ds, de)
                if hi_d < min_intruder_m or hi_d < ratio * max(lo_d, 0.05):
                    continue                      # a coin flip is jitter, not an intruder
                cam = "sideline" if ds > de else "endzone"
                plan.append((pid, cam, lo, hi, f"overlap run {lo}-{hi}: sideline {ds:.1f} m / "
                                                f"endzone {de:.1f} m from context"))
        if s and e and not both:
            s_lo, s_hi, e_lo, e_hi = min(s), max(s), min(e), max(e)
            if 0 < e_lo - s_hi <= handover_gap:
                d = float(np.linalg.norm(e[e_lo] - s[s_hi]))
                if d > ctrl_m:
                    plan.append((pid, "endzone", e_lo, e_hi, f"handover {s_hi}->{e_lo}: {d:.1f} m apart"))
            elif 0 < s_lo - e_hi <= handover_gap:
                d = float(np.linalg.norm(s[s_lo] - e[e_hi]))
                if d > ctrl_m:
                    plan.append((pid, "sideline", s_lo, s_hi, f"handover {e_hi}->{s_lo}: {d:.1f} m apart"))
    return plan


def team_from_kit(rows: pd.DataFrame, min_rows: int = 4):
    if "kit_margin" not in rows.columns:
        return None
    mg = rows["kit_margin"].to_numpy(float)
    mg = mg[np.isfinite(mg)]
    if len(mg) < min_rows:
        return None
    return "KC" if float(np.mean(mg > 0)) > 0.5 else "BAL"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--apply", action="store_true", help="write the tables (default: dry run)")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="restrict to these global ids")
    args = ap.parse_args()
    P = args.play_dir
    tp, kp, ip = P / "tracks.parquet", P / "keypoints_2d.parquet", P / "identity_resolved.pkl"
    for need in (tp, kp, P / "cameras.npz"):
        if not need.exists():
            raise SetupError(f"08u: missing {need}")

    tracks = load_camera_track(P / "cameras.npz")
    raw = pd.read_parquet(tp)
    keys = pd.read_parquet(kp)
    offset = clip_offset(P)
    shift = {"endzone": offset} if offset else None
    df = raw[raw["track_id"] >= 0].copy()
    df.loc[df["cam"] == "endzone", "frame"] = df.loc[df["cam"] == "endzone", "frame"].astype(int) - offset
    kdf = keys.copy()
    kdf.loc[kdf["cam"] == "endzone", "frame"] = kdf.loc[kdf["cam"] == "endzone", "frame"].astype(int) - offset
    ank = ankle_ground(kdf, tracks, frame_shift=shift)
    S = per_id(ground_positions(df[df["cam"] == "sideline"], tracks, ankles=ank, frame_shift=shift))
    E = per_id(ground_positions(df[df["cam"] == "endzone"], tracks, ankles=ank, frame_shift=shift))
    if args.only:
        S = {p: v for p, v in S.items() if p in args.only}
        E = {p: v for p, v in E.items() if p in args.only}

    plan = plan_unpairings(S, E)
    nxt = max(int(raw["global_player_id"].max()), int(keys["global_player_id"].max())) + 1
    print(f"08u: control {CTRL_M} m, min run {MIN_RUN}, margin {MIN_INTRUDER_M} m x{RATIO}; "
          f"clip offset {offset:+d}; {len(plan)} intervals")
    moved_total = 0
    assigned = []
    for i, (pid, cam, lo, hi, why) in enumerate(plan):
        new = nxt + i
        cam_lo, cam_hi = (lo + offset, hi + offset) if cam == "endzone" else (lo, hi)
        m = ((raw["cam"] == cam) & (raw["global_player_id"] == pid)
             & (raw["frame"] >= cam_lo) & (raw["frame"] <= cam_hi))
        mk = ((keys["cam"] == cam) & (keys["global_player_id"] == pid)
              & (keys["frame"] >= cam_lo) & (keys["frame"] <= cam_hi))
        moved_total += int(m.sum())
        print(f"  id {pid:>3} {cam:>8} frames {lo}-{hi} -> id {new}: {int(m.sum())} track rows, "
              f"{int(mk.sum())} keypoint rows   [{why}]")
        assigned.append((pid, new, m, mk))
    if not args.apply:
        print(f"dry run: {moved_total} track rows would move. Re-run with --apply.")
        return

    for pid, new, m, mk in assigned:
        raw.loc[m, "global_player_id"] = new
        keys.loc[mk, "global_player_id"] = new
    shutil.copy2(tp, tp.with_suffix(".parquet.pre08u"))
    shutil.copy2(kp, kp.with_suffix(".parquet.pre08u"))
    raw.to_parquet(tp, index=False)
    keys.to_parquet(kp, index=False)
    print(f"wrote {tp} and {kp} (backups .pre08u); {moved_total} track rows moved")

    if ip.exists():
        blob = pickle.load(open(ip, "rb"))
        merged = blob.get("merged", {})
        n_ident = 0
        for pid, new, _m, _mk in assigned:
            src = merged.get(pid)
            team = team_from_kit(raw[raw["global_player_id"] == new])
            if src is None or team is None:
                print(f"  id {new}: no identity written (source {src is not None}, team {team})")
                continue
            merged[new] = dataclasses.replace(src, team=team)
            n_ident += 1
        shutil.copy2(ip, ip.with_suffix(".pkl.pre08u"))
        blob["merged"] = merged
        pickle.dump(blob, open(ip, "wb"))
        print(f"wrote {ip} (backup .pre08u); {n_ident} fragment identities, team from kit colour")


if __name__ == "__main__":
    main()
