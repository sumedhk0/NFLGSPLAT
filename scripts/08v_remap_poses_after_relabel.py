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
    m = relabel_map(pd.read_parquet(bp), pd.read_parquet(ap_), offset)
    new_ids = sorted({v for v in m.values()})
    print(f"08v: {len(m)} (frame, old id) relabels across {len(new_ids)} new ids; clip offset {offset:+d}")

    for name in CACHES:
        fp = P / name
        if not fp.exists():
            print(f"  {name}: absent")
            continue
        d = pickle.load(open(fp, "rb"))
        frames = d.get("frames", {})
        moved, by_new = 0, {}
        for f, per in frames.items():
            if not isinstance(per, dict):
                continue
            for old in list(per.keys()):
                new = m.get((int(f), int(old)))
                if new is None or new in per:
                    continue
                if args.apply:
                    per[new] = per.pop(old)
                moved += 1
                by_new[new] = by_new.get(new, 0) + 1
        print(f"  {name} (cam {d.get('cam')}): {moved} posed frames "
              f"{'moved' if args.apply else 'would move'} across {len(by_new)} new ids")
        if args.apply and moved:
            b = backup_path(fp, ".pre08v")             # never overwrites an earlier pass's backup
            shutil.copy2(fp, b)
            pickle.dump(d, open(fp, "wb"))
            print(f"    wrote {fp} (backup {b.name})")
    if not args.apply:
        print("dry run. Re-run with --apply.")


if __name__ == "__main__":
    main()
