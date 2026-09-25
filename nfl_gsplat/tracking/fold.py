"""Fold ids that the FOOTAGE shows to be one man into one global id (an explicit twin merge).

WHY NOT twins.merge_map. 08o's merge works in the tracker-id space and rewrites every row's global id
from its track id; since the pairing stages (08s/08t/08u) the two differ on a fifth of play 1's rows,
so that path would silently relabel unrelated men. This folds in the GLOBAL id space and leaves
``track_id`` untouched -- 08v carries the pose caches across by joining the before/after tables on
(cam, frame, track_id).
"""
from __future__ import annotations

import pandas as pd


def fold_ids(df: pd.DataFrame, keep: int, drop: list[int], *, frames: tuple[int, int] | None = None,
             cam: str | None = None, track_id: int | None = None) -> tuple[pd.DataFrame, int]:
    """``(tracks with the dropped global ids folded into keep, rows dropped)``. Where two of the folded
    ids have a box on one frame of one camera one row survives: the more confident, then the kept id's
    own, then the taller box (the tracker's confidences tie at 1.0 on play 1); ``track_id`` stays.
    ``frames`` (lo, hi) folds only the dropped ids' rows on those frames (inclusive); the rest keep
    their id -- an endzone track that is one man for part of the clip and another after (play 1's
    164: the quarterback's endzone rows after his sideline id changed at 511). ``cam`` / ``track_id`` fold only
    that camera's rows / that camera track's rows of the dropped ids: a global id's sideline track can swap men
    while its endzone track stays on the first man (play 1's 211 at 523), and a whole-id fold would drag the
    endzone track onto the wrong man; the kept id's other rows are left alone."""
    drop = [int(d) for d in drop]
    out = df.copy()
    gid = out["global_player_id"].astype(int)
    is_drop = gid.isin(drop)
    if frames is not None:
        lo, hi = int(frames[0]), int(frames[1])
        is_drop = is_drop & out["frame"].astype(int).between(lo, hi)
    if cam is not None:
        is_drop = is_drop & (out["cam"].astype(str) == str(cam))
    if track_id is not None:
        is_drop = is_drop & (out["track_id"].astype(int) == int(track_id))
    folded = is_drop | (gid == int(keep))
    if cam is not None:                                   # the dedupe population stays within that camera
        folded = folded & (out["cam"].astype(str) == str(cam))
    sub = out[folded].copy()
    sub["_conf"] = out["conf"][folded] if "conf" in out else 1.0
    sub["_own"] = (gid[folded] == int(keep)).astype(int)
    sub["_h"] = (out["bbox_y2"] - out["bbox_y1"])[folded] if "bbox_y2" in out else 0.0
    sub = sub.sort_values(["_conf", "_own", "_h"], ascending=False)
    dropped = sub.index[sub.duplicated(subset=["cam", "frame"], keep="first")]
    out.loc[folded, "global_player_id"] = int(keep)
    out = out.drop(index=dropped)
    return out, int(len(dropped))


def carry_roles(roles: dict, *, keep: int, gone: list[int]) -> dict:
    """``roles`` (pid -> role, int or str keys) with the ``gone`` ids' entries removed and, when the kept id has no
    role, the first gone id's role given to it. Play 1 (2026-09-25): folding the centre's early id 17 (role OL) into
    204 (Humphrey, no role) dropped the role, and the ball path and the quarterback-under-centre hold -- both take the
    centre as the role-OL id nearest the line's middle -- chose the guard beside him."""
    out = {int(k): v for k, v in roles.items()}
    carried = None
    for pid in gone:
        r = out.pop(int(pid), None)
        if carried is None and r:
            carried = r
    if carried and not out.get(int(keep)):
        out[int(keep)] = carried
    return {k: v for k, v in out.items() if v}


def drop_rows(df: pd.DataFrame, pid: int, *, cam: str, frames: tuple[int, int],
              track_id: int | None = None) -> tuple[pd.DataFrame, int]:
    """``(df without id ``pid``'s rows of camera ``cam`` on frames lo..hi, rows removed)``; with ``track_id``
    only that camera track's rows (play 1 2026-09-25: the left tackle's endzone id alternating between his own
    box and a track that drifted onto the Raven he blocks). For a stray box the
    tracker re-associated to a track after a loss (play 1 id 4: one sideline box at 465, 22 frames after his
    track ended hidden behind KC 65, on another man): that box is the span's edge the beyond-span rule joins
    the endzone's stretch to, and a wrong edge drops the whole stretch. Keypoints follow through keypoint_map."""
    lo, hi = int(frames[0]), int(frames[1])
    m = ((df["global_player_id"].astype(int) == int(pid)) & (df["cam"].astype(str) == str(cam))
         & df["frame"].astype(int).between(lo, hi))
    if track_id is not None:
        m &= df["track_id"].astype(int) == int(track_id)
    return df.loc[~m].copy(), int(m.sum())


def keypoint_map(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """``{(cam, frame, old global id): new global id}`` for every row of ``before`` (-1 where the row
    was dropped), joined on the detection's (cam, frame, track_id)."""
    key = ["cam", "frame", "track_id"]
    b = before[before["track_id"] >= 0][key + ["global_player_id"]].rename(columns={"global_player_id": "old"})
    a = after[after["track_id"] >= 0][key + ["global_player_id"]].rename(columns={"global_player_id": "new"})
    j = b.merge(a, on=key, how="left")
    j["new"] = j["new"].fillna(-1).astype(int)
    return {(str(r.cam), int(r.frame), int(r.old)): int(r.new) for r in j.itertuples()}
