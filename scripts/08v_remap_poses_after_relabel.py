"""Carry posed frames across a relabel: the pose caches are keyed by global id and 08s/08t/08u move ids.

    python scripts/08v_remap_poses_after_relabel.py --play-dir P --before tracks.parquet.bak
    python scripts/08v_remap_poses_after_relabel.py --play-dir P --before tracks.parquet.bak --apply

WHY. poses_refit.json and poses_sideline.json are ``frames[f][global_player_id] -> record``. When a stage
moves rows of an id to a fresh id -- 08t cutting a track that switches men, 08u unpairing an intruding
camera, 08s relabelling a pair -- the tables change and the caches do not, so every fragment renders
DEFAULT-POSED: a stiff stance where a man was fitted. Play 1 (2026-09-15): 29 fragments across 08t and 08u.

HOW. ``track_id`` is stable per detection row, so joining the table BEFORE the relabels to the table after
on ``(cam, frame, track_id)`` yields every ``(frame, old_gid -> new_gid)`` exactly, with no guessing at
which stage did what. Endzone rows are converted from the endzone clip's frame numbers to the timeline's
by the clip offset, because the caches sit on the timeline clock.

RULE. A pose at frame f follows the SIDELINE row's relabel. The timeline draws a body where the sideline
sees it, so when 08u moves only an endzone intrusion the pose stays with the sideline man; when 08t cuts
both cameras the pose moves. An endzone-only frame (no sideline row) follows the endzone row.

Runs under the venv that WRITES the caches (numpy-1 pickles: smplx312). Backs up before writing; fails loud
on a missing table.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_gsplat.errors import SetupError  # noqa: E402
from nfl_gsplat.render.play_timeline import clip_offset  # noqa: E402
from nfl_gsplat.tracking.relabel import backup_path  # noqa: E402

CACHES = ("poses_refit.json", "poses_sideline.json")


def relabel_map(before: pd.DataFrame, after: pd.DataFrame, offset: int) -> dict:
    """``{(timeline_frame, old_gid): new_gid}`` preferring the sideline row's relabel at each frame."""
    key = ["cam", "frame", "track_id"]
    b = before[before["track_id"] >= 0][key + ["global_player_id"]].rename(columns={"global_player_id": "old"})
    a = after[after["track_id"] >= 0][key + ["global_player_id"]].rename(columns={"global_player_id": "new"})
    j = b.merge(a, on=key, how="inner")
    j = j[j["old"] != j["new"]].copy()
    j["tl"] = j["frame"].astype(int)
    j.loc[j["cam"] == "endzone", "tl"] = j.loc[j["cam"] == "endzone", "frame"].astype(int) - offset
    out: dict = {}
    # endzone first, then sideline overwrites: the sideline's relabel wins where both moved
    for cam in ("endzone", "sideline"):
        for r in j[j["cam"] == cam].itertuples():
            out[(int(r.tl), int(r.old))] = int(r.new)
    return out


def dropped_keys(before: pd.DataFrame, after: pd.DataFrame, offset: int) -> set:
    """``{(timeline_frame, old_gid)}`` whose DECIDING row was removed (08za): the sideline row when the id had one on
    that frame, else the endzone row. A pose fitted to a dropped box is that box's man's, not the id's (play 1
    2026-09-25: the centre's refit record at 474 fitted to 84's box). An endzone row dropped under a sideline row that
    stays leaves the pose, as the relabel rule does."""
    key = ["cam", "frame", "track_id"]

    def tl_rows(df):
        d = df[df["track_id"] >= 0][key + ["global_player_id"]].copy()
        d["tl"] = d["frame"].astype(int)
        d.loc[d["cam"] == "endzone", "tl"] = d.loc[d["cam"] == "endzone", "frame"].astype(int) - offset
        return d

    b, a = tl_rows(before), tl_rows(after)
    j = b.merge(a[key].assign(kept=True), on=key, how="left")
    gone = j[j["kept"].isna()]
    side_before = set(zip(b.loc[b["cam"] == "sideline", "tl"].astype(int), b.loc[b["cam"] == "sideline", "global_player_id"].astype(int)))
    left = {(c, int(t), int(g)) for c, t, g in zip(a["cam"], a["tl"], a["global_player_id"])}
    out = set()
    for r in gone.itertuples():
        k = (int(r.tl), int(r.global_player_id))
        if r.cam == "sideline" and ("sideline", *k) not in left:
            out.add(k)
        elif r.cam == "endzone" and k not in side_before and ("endzone", *k) not in left:
            out.add(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--play-dir", required=True, type=Path)
    ap.add_argument("--before", required=True, help="the tracks table from BEFORE the relabels, relative to "
                                                     "the play dir (e.g. tracks.parquet.bak)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    P = args.play_dir
    bp, ap_ = P / args.before, P / "tracks.parquet"
    for need in (bp, ap_):
        if not need.exists():
            raise SetupError(f"08v: missing {need}")
    offset = clip_offset(P)
    tb_df, ta_df = pd.read_parquet(bp), pd.read_parquet(ap_)
    m = relabel_map(tb_df, ta_df, offset)
    drops = dropped_keys(tb_df, ta_df, offset)
    new_ids = sorted({v for v in m.values()})
    print(f"08v: {len(m)} (frame, old id) relabels across {len(new_ids)} new ids; {len(drops)} (frame, id) whose "
          f"deciding row was dropped; clip offset {offset:+d}")

    for name in CACHES:
        fp = P / name
        if not fp.exists():
            print(f"  {name}: absent")
            continue
        d = pickle.load(open(fp, "rb"))
        frames = d.get("frames", {})
        moved, by_new, removed = 0, {}, 0
        rebuilt: dict = {}
        for f, per in frames.items():
            if not isinstance(per, dict):
                continue
            # per frame: records that stay under their id first, then the moving ones; a moving record whose target
            # id is posed here already (it stays, or an earlier move took it) is that man's box under the old id --
            # play 1 2026-09-25: the quarterback's record on the centre's box, the centre posed there by the endzone
            # -- and goes; two ids that trade rows trade records
            stay, moves = {}, []
            for old, rec in per.items():
                new = m.get((int(f), int(old)))
                if new is None:
                    if (int(f), int(old)) in drops:
                        removed += 1
                    else:
                        stay[old] = rec
                else:
                    moves.append((new, rec))
            out_per = dict(stay)
            for new, rec in moves:
                if new in out_per:
                    removed += 1
                    continue
                out_per[new] = rec
                moved += 1
                by_new[new] = by_new.get(new, 0) + 1
            rebuilt[f] = out_per
        if args.apply:
            for f, out_per in rebuilt.items():
                frames[f] = out_per
        print(f"  {name} (cam {d.get('cam')}): {moved} posed frames "
              f"{'moved' if args.apply else 'would move'} across {len(by_new)} new ids; {removed} "
              f"{'removed' if args.apply else 'would go'} (their row dropped, or its man posed there already)")
        if args.apply and (moved or removed):
            b = backup_path(fp, ".pre08v")             # never overwrites an earlier pass's backup
            shutil.copy2(fp, b)
            pickle.dump(d, open(fp, "wb"))
            print(f"    wrote {fp} (backup {b.name})")
    if not args.apply:
        print("dry run. Re-run with --apply.")


if __name__ == "__main__":
    main()
