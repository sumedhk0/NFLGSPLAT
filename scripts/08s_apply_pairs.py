#!/usr/bin/env python
"""Apply 08r's cross-camera pairing proposals by RELABELLING the other camera's tracks. Backs up first.

    python scripts/08s_apply_pairs.py --play-dir P [--proposal pair_proposal.json] [--dry-run]

WHAT IT DOES, AND THE ONE OPERATION IT WILL PERFORM. For a proposal "sideline s <- endzone e" it rewrites
`global_player_id` from e to s on the ENDZONE rows only, in tracks.parquet and keypoints_2d.parquet. It
never unions the two ids. A global id spans both cameras, so unioning 11 and 19 to give sideline 11 the
endzone track 19 would also sweep in sideline 19 -- a different man who happens to carry that number. The
relabel is the minimal statement of what was measured: "this endzone track belongs to that sideline id".

THE INVARIANT IT ENFORCES. After relabelling, one id may not hold two tracks of one camera at the same
time (pair_by_appearance.global_ids_checked exists because a re-pairing once merged Kansas City's centre
and quarterback into one sideline id). If the target already has endzone rows on any frame the incoming
track covers, the relabel is refused and the run fails loud -- disjoint spans are fine, overlapping ones
are not.

WHAT TO DO AFTERWARDS. The two-view pass pairs by global id, so 05p has to run again for the new pairing
to reach the fits, and the result is scored with 05t (the camera the bodies were NOT fitted to) plus the
along-ray gap from the point both cameras put the feet. The sideline's own reprojection is not evidence
about placement and will not judge this.

Only proposals 08r left unrejected are applied; everything else in the file is ignored.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


class ApplyError(RuntimeError):
    """A relabel that would break the one-id-one-place invariant, or a proposal file that does not fit."""


def frames_of(df: pd.DataFrame, cam: str, pid: int) -> set:
    g = df[(df["cam"] == cam) & (df["global_player_id"] == pid)]
    return set(int(x) for x in g["frame"].unique())


def check(df: pd.DataFrame, cam: str, own: int, other: int) -> None:
    """Refuse a relabel that would put two tracks of one camera under one id on the same frame."""
    clash = frames_of(df, cam, own) & frames_of(df, cam, other)
    if clash:
        lo, hi = min(clash), max(clash)
        raise ApplyError(
            f"{cam} id {own} already holds a track on {len(clash)} of the frames {cam} id {other} covers "
            f"({lo}-{hi}); relabelling would put two {cam} tracks under id {own} at once. Give one up "
            f"first, or drop this proposal.")


def relabel(df: pd.DataFrame, cam: str, own: int, other: int) -> int:
    """Move every row of ``other`` in ``cam`` onto ``own``; returns the number of rows moved."""
    m = (df["cam"] == cam) & (df["global_player_id"] == other)
    n = int(m.sum())
    df.loc[m, "global_player_id"] = own
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--proposal", type=Path, default=None, help="default <play-dir>/pair_proposal.json")
    ap.add_argument("--dry-run", action="store_true", help="report and write nothing")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="apply only the proposals whose own_id is in this list")
    args = ap.parse_args()
    P = args.play_dir
    prop_path = args.proposal or P / "pair_proposal.json"
    if not prop_path.exists():
        raise SystemExit(f"no proposal at {prop_path}; run scripts/08r_pair_by_rays.py first")
    blob = json.loads(prop_path.read_text())
    cam, other = blob["cam"], blob["other"]
    rows = [r for r in blob["proposals"] if r.get("rejected") is None]
    if args.only is not None:
        rows = [r for r in rows if int(r["own_id"]) in args.only]
    if not rows:
        raise SystemExit(f"{prop_path} holds no accepted proposals to apply")

    tracks_path, kp_path = P / "tracks.parquet", P / "keypoints_2d.parquet"
    tdf = pd.read_parquet(tracks_path)
    kdf = pd.read_parquet(kp_path)

    print(f"{len(rows)} proposals from {prop_path.name}; relabelling {other} tracks onto {cam} ids")
    for r in rows:
        own, oid = int(r["own_id"]), int(r["other_id"])
        check(tdf, other, own, oid)
        print(f"  {cam} {own:3d} <- {other} {oid:3d}: rays {r['ray_miss_m']} m, turf {r['ground_gap_m']} m "
              f"over {r['frames']} frames ({r['kind']}, would apply by {r.get('apply_as', '?')})")
    if args.dry_run:
        print("dry run; nothing written")
        return

    for name, path in (("tracks", tracks_path), ("keypoints", kp_path)):
        backup = path.with_name(path.stem + "_prepair" + path.suffix)
        if not backup.exists():
            shutil.copy(path, backup)
            print(f"kept {name} as {backup.name}")

    moved_t = moved_k = 0
    for r in rows:
        own, oid = int(r["own_id"]), int(r["other_id"])
        moved_t += relabel(tdf, other, own, oid)
        moved_k += relabel(kdf, other, own, oid)
    tdf.to_parquet(tracks_path, index=False)
    kdf.to_parquet(kp_path, index=False)
    print(f"moved {moved_t} track rows and {moved_k} keypoint rows onto {len(rows)} {cam} ids")
    for r in rows:
        own = int(r["own_id"])
        print(f"  {cam} {own:3d} now has {len(frames_of(tdf, cam, own))} {cam} frames and "
              f"{len(frames_of(tdf, other, own))} {other} frames")
    print("re-run 05p for the new pairing to reach the fits, then score with 05t and the along-ray gap")


if __name__ == "__main__":
    main()
