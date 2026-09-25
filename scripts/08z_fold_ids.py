#!/usr/bin/env python
"""Fold ids that the FOOTAGE shows to be one man into one id (an explicit twin merge).

    C:/venvs/smplx312/Scripts/python scripts/08z_fold_ids.py --play-dir P --keep 74 --drop 48 75 [--frames LO HI] [--apply]

WHY. 08o finds twins by ankle rays, boxes and team; it cannot fold a fragment whose team label is
wrong. Play 1 (2026-09-18): the receiver, Noah Gray (id 74, jersey 83 read by OCR, sideline track
421-520), continues as id 48 (508-588, unnamed, given an OL build by 08n) and then as id 75 (528-602,
labelled BAL on an unreadable kit) through the tackle. Three ids, one man, and the ball chain (08y)
lost him at 588 because his continuation wore the other team's colour. The footage names the ids;
this folds them (tracking.fold).

WHAT IT DOES. tracks.parquet: every row of a dropped id takes the kept GLOBAL id (track ids stay);
where two of the folded ids have a box on one frame of one camera one row survives (confidence, then
the kept id's own, then the taller box). Every keypoint table (keypoints_2d.parquet and the fine-tuned
detector's keypoints_2d_ft2.parquet, tracking.relabel.KEYPOINT_TABLES) follows the rows. The pose caches are
carried by 08v (before/after tables joined on (cam, frame, track_id)), which this runs.
identity_resolved.pkl loses the dropped ids from ``merged`` and ``roles``. Backups: *.pre_fold (never
overwritten: tracking.relabel.backup_path). Runs under the venv that WRITES the caches (numpy-1
pickles: smplx312).
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nfl_gsplat.errors import SetupError  # noqa: E402
from nfl_gsplat.tracking.fold import carry_roles, fold_ids, keypoint_map  # noqa: E402
from nfl_gsplat.tracking.relabel import assert_numpy1_for_pickles, backup_path, relabel_keypoint_tables  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--keep", type=int, required=True, help="the id that survives (its identity names the man)")
    ap.add_argument("--drop", type=int, nargs="+", required=True, help="ids the footage shows to be the same man")
    ap.add_argument("--frames", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="fold only the dropped ids' rows on these frames (inclusive); the rest keep their id")
    ap.add_argument("--cam", default=None, help="fold only this camera's rows of the dropped ids")
    ap.add_argument("--track-id", type=int, default=None, help="... and only this camera track's rows (with --cam)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    assert_numpy1_for_pickles()
    df = pd.read_parquet(P / "tracks.parquet")
    ids = set(df["global_player_id"].astype(int))
    for pid in [args.keep, *args.drop]:
        if pid not in ids:
            raise SetupError(f"08z: id {pid} has no rows in tracks.parquet")
    out, n_rows = fold_ids(df, args.keep, args.drop, frames=tuple(args.frames) if args.frames else None,
                           cam=args.cam, track_id=args.track_id)
    for pid in [args.keep, *args.drop]:
        for cam, g in df[df["global_player_id"] == pid].groupby("cam"):
            print(f"before: id {pid} {cam} frames {int(g.frame.min())}-{int(g.frame.max())} ({len(g)} boxes)")
    for cam, g in out[out["global_player_id"] == args.keep].groupby("cam"):
        print(f"after:  id {args.keep} {cam} frames {int(g.frame.min())}-{int(g.frame.max())} ({len(g)} boxes)")
    print(f"{n_rows} boxes dropped where two of the folded ids shared a frame")
    if not args.apply:
        print("dry run. Re-run with --apply.")
        return
    tb = backup_path(P / "tracks.parquet", ".pre_fold")
    shutil.copy2(P / "tracks.parquet", tb)
    out.to_parquet(P / "tracks.parquet", index=False)
    print(f"wrote tracks.parquet (backup {tb.name})")
    for name, n_rows_k, kdrop, n_changed, kb in relabel_keypoint_tables(P, keypoint_map(df, out), ".pre_fold"):
        print(f"{name}: {n_rows_k} rows written, {kdrop} dropped, {n_changed} relabelled (backup {kb})")
    subprocess.run([sys.executable, str(Path(__file__).with_name("08v_remap_poses_after_relabel.py")),
                    "--play-dir", str(P), "--before", tb.name, "--apply"], check=True)
    ip = P / "identity_resolved.pkl"
    blob = pickle.load(open(ip, "rb"))
    ib = backup_path(ip, ".pre_fold")
    shutil.copy2(ip, ib)
    gone = [p for p in args.drop if not (out["global_player_id"] == p).any()]     # an id folded in part keeps its identity
    roles_before = blob.get("roles", {}) or {}
    roles_after = carry_roles(roles_before, keep=args.keep, gone=gone)
    if roles_after.get(int(args.keep)) and not (roles_before.get(args.keep) or roles_before.get(str(args.keep))):
        print(f"identity: roles[{args.keep}] = {roles_after[int(args.keep)]} carried from the folded id")
    blob["roles"] = roles_after
    for key in ("merged", "roles"):
        d = blob.get(key, {})
        for pid in gone:
            for k in (pid, str(pid)):
                if k in d:
                    print(f"identity: {key}[{pid}] = {d.pop(k)} removed")
    pickle.dump(blob, open(ip, "wb"))
    print(f"wrote identity_resolved.pkl (backup {ib.name}); id {args.keep} = {blob['merged'].get(args.keep)}")


if __name__ == "__main__":
    main()
