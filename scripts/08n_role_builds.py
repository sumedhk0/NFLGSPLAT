#!/usr/bin/env python
"""Builds for the unnamed ids from their pre-snap role (identity.roles).

    python scripts/08n_role_builds.py --play-dir <P> --offence KC [--dry-run]

Reads the sideline ground positions (ankle keypoints where present), the boxes, the
identity; finds the snap, the pre-snap window and the line of scrimmage; assigns a
role per id (OL DL LB DB WR TE RB QB) and gives every UNNAMED id its role's median
roster height and weight (KC/BAL 2024). Rewrites identity_resolved.pkl (original kept as
identity_resolved_prerole.pkl); the fits (05p) and the timeline read the build from it.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_gsplat.calibration.cameras_io import load_camera_track
from nfl_gsplat.identity.roles import (POSITION_BUILDS, apply_role_builds, assign_roles, formation_frame,
                                       inherit_roles, line_of_scrimmage, position_builds, presnap_summary,
                                       roles_from_roster, snap_frame, track_ends, WINDOW_BEFORE, WINDOW_MARGIN)
from nfl_gsplat.render.play_timeline import ankle_ground, ground_positions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--offence", required=True, help="the team with the ball (KC, BAL, ...)")
    ap.add_argument("--rosters", type=Path, default=Path("data/rosters/2024/rosters.parquet"))
    ap.add_argument("--snap", type=int, default=None, help="snap frame (default: found from the motion)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    tracks = load_camera_track(P / "cameras.npz")
    fps = float(np.load(P / "cameras.npz", allow_pickle=True)["fps"])
    df = pd.read_parquet(P / "tracks.parquet")
    df = df[df["track_id"] >= 0]
    ankles = ankle_ground(pd.read_parquet(P / "keypoints_2d.parquet"), tracks) if (P / "keypoints_2d.parquet").exists() else None
    ground = ground_positions(df[df["cam"] == "sideline"], tracks, ankles=ankles)
    ident_path = P / "identity_resolved.pkl"
    blob = pickle.load(open(ident_path, "rb"))
    merged = blob.get("merged", {})
    teams = {int(k): getattr(v, "team", None) for k, v in merged.items()}
    snap = args.snap if args.snap is not None else snap_frame(ground, fps=fps)
    if snap is None:
        raise SystemExit("no snap found: the bodies never start moving together")
    settled = formation_frame(ground, fps=fps)
    lo = max(min(ground), snap - WINDOW_BEFORE) if settled is None else settled
    window = (lo, snap - WINDOW_MARGIN)
    summary = presnap_summary(ground, df, window)
    los, sign, yc = line_of_scrimmage(summary, teams, args.offence)
    roles = assign_roles(summary, teams, args.offence, los, sign, yc)
    roster = pd.read_parquet(args.rosters) if args.rosters.exists() else None
    builds = position_builds(roster, {args.offence} | {t for t in teams.values() if t}) if roster is not None else POSITION_BUILDS
    n_formation = len(roles)
    # the roster's position beats the formation for a named id: it is ground truth, and the
    # formation reads a back who lines up behind the centre as the passer (play 1's Pacheco)
    named = roles_from_roster(merged, roster) if roster is not None else {}
    roles.update(named)
    print(f"snap at frame {snap} (formation set at {settled}); pre-snap window {window}; line of scrimmage x = "
          f"{los:.1f} m, offence on the {'+' if sign > 0 else '-'}x side, formation centre y = {yc:.1f}")
    print("  id team role   dx    dy   aspect  name -> build")
    on_field = [p for p in roles if p in summary]          # a named id may never appear before the snap
    for pid in sorted(on_field, key=lambda p: ((summary[p][0] - los) * sign, summary[p][1])):
        x, y, a, n = summary[pid]
        p = merged.get(pid) or merged.get(str(pid))
        name = getattr(p, "player", "?")
        h, kg = builds.get(roles[pid], POSITION_BUILDS["DB"])
        tag = f"{h:.2f} m {kg:.0f} kg" if str(name).startswith("P") else f"named ({getattr(p, 'height_m', 0):.2f} m)"
        print(f"  {pid:3d} {teams[pid]:3s} {roles[pid]:3s} {(x - los) * sign:+5.1f} {y - yc:+5.1f}  {a:5.2f}   {name} -> {tag}")
    # every roled track passes its role to the track that continues it
    spans, ends = track_ends(ground, set(teams))
    inherited = inherit_roles(roles, teams, spans, ends)
    for pid, (role, q) in sorted(inherited.items(), key=lambda kv: spans[kv[0]][0]):
        roles[pid] = role
        p = merged.get(pid) or merged.get(str(pid))
        print(f"  {pid:3d} {teams[pid]:3s} {role:3s} inherited from {q:3d} (ended {spans[q][1]}, starts {spans[pid][0]}, "
              f"{np.linalg.norm(np.asarray(ends[q][1]) - np.asarray(ends[pid][0])):.1f} m)  {getattr(p, 'player', '?')}")
    n = apply_role_builds(merged, roles, builds)
    left = sorted(p for p in spans if p not in roles)
    print(f"{n_formation} roles from the formation, {len(named)} from the roster, {len(inherited)} inherited; "
          f"{n} unnamed ids given their role's build; {len(left)} tracks without a role: {left}")
    if args.dry_run:
        print("dry run; identity_resolved.pkl untouched")
        return
    backup = P / "identity_resolved_prerole.pkl"
    if not backup.exists():
        shutil.copy(ident_path, backup)
    blob["roles"] = {int(k): v for k, v in roles.items()}
    with open(ident_path, "wb") as fh:
        pickle.dump(blob, fh)
    print(f"wrote {ident_path} (original in {backup.name})")


if __name__ == "__main__":
    main()
