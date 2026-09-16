#!/usr/bin/env python
"""A weak kit vote is overruled by where the man lines up before the snap.

    python scripts/08w_team_from_formation.py --play-dir P            # dry run: lists the candidates
    python scripts/08w_team_from_formation.py --play-dir P --apply    # rewrites identity_resolved.pkl (backup first)

WHY. The team of a fragment comes from its kit colour (kit_margin, signed saturation margin per
detection: positive is the saturated kit). A crouched lineman shows the camera his white pants and
little jersey, and the vote goes to the white team: play 1's id 82 (kit_margin median -0.12, 55 rows)
was drawn as a Baltimore player standing 1.5 m inside the Kansas City line for the whole pre-snap --
exactly "a Chiefs offensive lineman appearing as a Baltimore player" as the user saw it (2026-09-16).

WHAT. On the pre-snap frames (LOS snap - PRE_FRAMES .. snap - 10) each id's sideline ground point is
measured against the line of scrimmage from identity_resolved.pkl (08n's block: x, the offence's side
sign, the snap frame). An id that stands DEEP on the offence's side (more than --depth m) on at least
--min-frames of those frames, and whose kit vote is weak (positive share within --weak of 0.5), takes
the offence's team. A strong kit vote is never overruled: a defensive lineman crouched at the line is
on his own side of it, and a wide receiver split out is deep on the offence's side but wears his kit
plainly. Every candidate is printed with its numbers so the rule is checked on the footage.

Runs under nflgsplat (pandas, no pose caches touched); the pickle holds dataclasses only.
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
from nfl_gsplat.render.play_timeline import ground_positions  # noqa: E402

PRE_FRAMES = 120


def positive_share(km: pd.Series) -> float:
    v = km.to_numpy(float)
    v = v[np.isfinite(v)]
    return float((v > 0).mean()) if len(v) else float("nan")


def plan(ground: dict, kit_share: dict, team_of: dict, los: dict, *, depth_m: float, min_frames: int, weak: float,
         offence: str, defence: str) -> list:
    """[(pid, team_now, frames deep, depth p50, positive share)] of the ids the rule would move.
    ``ground``: pre-snap frame -> {pid: xy}; ``kit_share``: pid -> share of positive kit margins;
    ``team_of``: pid -> team; ``los``: 08n's block (x, sign, snap)."""
    x0, sign = float(los["x"]), float(los["sign"])
    deep: dict = {}
    for f, d in ground.items():
        for pid, xy in d.items():
            dep = sign * (float(xy[0]) - x0)                 # positive = the offence's side
            deep.setdefault(int(pid), []).append(dep)
    out = []
    for pid, deps in deep.items():
        team = team_of.get(pid)
        if team != defence:
            continue
        deps = np.asarray(deps)
        n_deep = int((deps > depth_m).sum())
        if n_deep < min_frames:
            continue
        share = kit_share.get(pid, float("nan"))
        if not np.isfinite(share) or abs(share - 0.5) > weak:
            continue
        out.append((pid, team, n_deep, float(np.median(deps)), share))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--offence", default="KC", help="the team with the ball (its side of the LOS is the offence's)")
    ap.add_argument("--defence", default="BAL")
    ap.add_argument("--depth", type=float, default=0.5, help="metres on the offence's side of the LOS (no defender may stand there before the snap)")
    ap.add_argument("--min-frames", type=int, default=15)
    ap.add_argument("--weak", type=float, default=0.3, help="a positive share within this of 0.5 is a weak vote")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    ip = P / "identity_resolved.pkl"
    if not ip.exists():
        raise SetupError(f"{ip} missing: 08c/08n have not run")
    blob = pickle.load(open(ip, "rb"))
    merged = blob.get("merged", {})
    los = blob.get("line_of_scrimmage")
    if not los:
        raise SetupError("identity_resolved.pkl has no line_of_scrimmage block: run 08n first")
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    tracks = load_camera_track(P / "cameras.npz")
    snap = int(los["snap"])
    side = df[(df["cam"] == "sideline") & (df["frame"] >= snap - PRE_FRAMES) & (df["frame"] <= snap - 10)]
    ground = ground_positions(side, tracks)
    kit_share = ({int(p): positive_share(g["kit_margin"]) for p, g in df[df["cam"] == "sideline"].groupby("global_player_id")}
                 if "kit_margin" in df else {})
    team_of = {int(k): getattr(v, "team", None) for k, v in merged.items()}
    cands = plan(ground, kit_share, team_of, los, depth_m=args.depth, min_frames=args.min_frames, weak=args.weak,
                 offence=args.offence, defence=args.defence)
    print(f"08w: LOS x {float(los['x']):.2f}, offence's side sign {float(los['sign']):+.0f}, snap {int(los['snap'])}; "
          f"{len(cands)} {args.defence}-labelled ids stand > {args.depth} m on the {args.offence} side with a weak kit vote")
    for pid, team, n_deep, dep, share in cands:
        print(f"  id {pid:>4}: {team} -> {args.offence}   deep on {n_deep} pre-snap frames (p50 {dep:.2f} m), "
              f"kit positive share {share:.2f}")
    if not args.apply:
        print("dry run. Re-run with --apply.")
        return
    if cands:
        shutil.copy2(ip, ip.with_name(ip.name + ".pre08w"))
        for pid, *_ in cands:
            merged[pid] = dataclasses.replace(merged[pid], team=args.offence)
        blob["merged"] = merged
        pickle.dump(blob, open(ip, "wb"))
    print(f"wrote {ip} (backup .pre08w); {len(cands)} ids re-teamed")


if __name__ == "__main__":
    main()
