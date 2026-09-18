"""Fold ids that the FOOTAGE shows to be one man into one global id (an explicit twin merge).

WHY NOT twins.merge_map. 08o's merge works in the tracker-id space and rewrites every row's global id
from its track id; since the pairing stages (08s/08t/08u) the two differ on a fifth of play 1's rows,
so that path would silently relabel unrelated men. This folds in the GLOBAL id space and leaves
``track_id`` untouched -- 08v carries the pose caches across by joining the before/after tables on
(cam, frame, track_id).
"""
from __future__ import annotations

import pandas as pd


def fold_ids(df: pd.DataFrame, keep: int, drop: list[int]) -> tuple[pd.DataFrame, int]:
    """``(tracks with the dropped global ids folded into keep, rows dropped)``. Where two of the folded
    ids have a box on one frame of one camera one row survives: the more confident, then the kept id's
    own, then the taller box (the tracker's confidences tie at 1.0 on play 1); ``track_id`` stays."""
    drop = [int(d) for d in drop]
    out = df.copy()
    gid = out["global_player_id"].astype(int)
    folded = gid.isin([int(keep), *drop])
    sub = out[folded].copy()
    sub["_conf"] = out["conf"][folded] if "conf" in out else 1.0
    sub["_own"] = (gid[folded] == int(keep)).astype(int)
    sub["_h"] = (out["bbox_y2"] - out["bbox_y1"])[folded] if "bbox_y2" in out else 0.0
    sub = sub.sort_values(["_conf", "_own", "_h"], ascending=False)
    dropped = sub.index[sub.duplicated(subset=["cam", "frame"], keep="first")]
    out.loc[folded, "global_player_id"] = int(keep)
    out = out.drop(index=dropped)
    return out, int(len(dropped))


def keypoint_map(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """``{(cam, frame, old global id): new global id}`` for every row of ``before`` (-1 where the row
    was dropped), joined on the detection's (cam, frame, track_id)."""
    key = ["cam", "frame", "track_id"]
    b = before[before["track_id"] >= 0][key + ["global_player_id"]].rename(columns={"global_player_id": "old"})
    a = after[after["track_id"] >= 0][key + ["global_player_id"]].rename(columns={"global_player_id": "new"})
    j = b.merge(a, on=key, how="left")
    j["new"] = j["new"].fillna(-1).astype(int)
    return {(str(r.cam), int(r.frame), int(r.old)): int(r.new) for r in j.itertuples()}
