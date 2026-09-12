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

GIVING UP AN INCUMBENT (--give-up-incumbent). A re-pairing can only be applied by moving the sitting
track out of the way first, so with this flag the incumbent is relabelled to a FRESH unused id rather
than deleted: it survives as an unpaired track of its own camera and the existing drawing rules decide
whether it appears. Use it only where the evidence is lopsided. Play 1's case is the example --
sideline 7's endzone partner is id 7 on 13 ankle frames at 0.17 m with a 0.80 m turf gap, against id 89
on 184 frames at 0.08 m with a 0.37 m gap; id 89 spans 125-634, the whole play, while id 7 exists only
at 539-634, which is precisely the collision, and id 7's best alternative partner is 7.61 m away, so it
has no other home. Evicting it is the honest reading of the geometry, not a convenience.

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


def check(df: pd.DataFrame, cam: str, own: int, other: int, *, give_up: bool = False) -> int:
    """The number of frames the incumbent track collides on. Refuses the relabel unless ``give_up``,
    because one id may not hold two tracks of one camera at the same time."""
    clash = frames_of(df, cam, own) & frames_of(df, cam, other)
    if clash and not give_up:
        lo, hi = min(clash), max(clash)
        raise ApplyError(
            f"{cam} id {own} already holds a track on {len(clash)} of the frames {cam} id {other} covers "
            f"({lo}-{hi}); relabelling would put two {cam} tracks under id {own} at once. Give one up "
            f"first (--give-up-incumbent), or drop this proposal.")
    return len(clash)


def fresh_id(*frames: pd.DataFrame) -> int:
    """An id no table uses, for a track being moved out of the way rather than deleted."""
    return max(int(d["global_player_id"].max()) for d in frames) + 1


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
    ap.add_argument("--give-up-incumbent", action="store_true",
                    help="a re-pairing needs the sitting other-camera track moved out of the way; with this "
                         "it is relabelled to a FRESH unused id (it survives unpaired, it is not deleted) "
                         "instead of the run refusing. Only where the evidence is lopsided -- see the "
                         "docstring for play 1's 184 frames at 0.08 m against 13 at 0.17 m")
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
        clash = check(tdf, other, own, oid, give_up=args.give_up_incumbent)
        note = "" if not clash else f", evicting the incumbent {other} {own} from {clash} frames"
        print(f"  {cam} {own:3d} <- {other} {oid:3d}: rays {r['ray_miss_m']} m, turf {r['ground_gap_m']} m "
              f"over {r['frames']} frames ({r['kind']}, would apply by {r.get('apply_as', '?')}){note}")
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
        if check(tdf, other, own, oid, give_up=args.give_up_incumbent):
            spare = fresh_id(tdf, kdf)
            n_t = relabel(tdf, other, spare, own)      # the incumbent steps aside, it is not deleted
            n_k = relabel(kdf, other, spare, own)
            print(f"  {other} {own} gives up the id: {n_t} track rows and {n_k} keypoint rows moved to "
                  f"the unused id {spare}, where they stay as an unpaired {other} track")
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
