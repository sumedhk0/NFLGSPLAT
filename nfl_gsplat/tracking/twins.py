"""Two tracker ids on ONE body, and how to fold them into one.

WHY. The detector fires twice on a lineman in a pile and the linker keeps both
boxes as separate tracks, so the render stands two avatars on one man: play 1's
sideline held the left tackle as ids 18 and 19 and the quarterback as 16 and 22.
It also shows as a structural impossibility -- six interior linemen on the
offensive line, two quarterbacks under centre.

WHAT the test is. NOT box geometry: from the sideline the line of scrimmage is
seen end-on, so two men standing a metre apart in depth overlap in the image as
much as a duplicate does (play 1: 14 & 27, two Baltimore players one behind the
other, boxes 0.19 box-heights apart with IoU 0.66). The ankle keypoints' rays
carried to the turf DO separate them, because that resolves depth: over play 1's
sideline pairs the two duplicates sit 0.03 and 0.11 m apart and the next
same-team pair 0.80 m. Each id's ankle track is filled across short gaps first --
the pose detector suppresses one of two overlapping boxes, so the two ids rarely
carry ankles on the same frame.

Gates, all of them: the same team (an offensive and a defensive lineman locked
together stand close too), the ankle gap, box overlap, and enough frames to
judge. A merge keeps the longer track's id; where both boxes exist on a frame the
more confident one survives and the other is dropped, so the merged id holds ONE
box per frame -- which is what the fits, the pairing and the render all assume.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# 0.35 was too loose: play 1's endzone had the quarterback and the lineman beside him 0.29 m apart
# at the feet, plainly two men in the footage. The duplicates sit at 0.03 and 0.11 m.
MAX_ANKLE_GAP_M: float = 0.20    # two ids' feet this close on the turf are one body ...
MIN_ANKLE_FRAMES: int = 20       # ... judged over at least this many frames ...
MIN_IOU: float = 0.3             # ... whose boxes also overlap this much at the median
FILL_FRAMES: int = 4             # an ankle point is carried this far across a gap
BOX = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def filled_ankles(ankles: dict, df: pd.DataFrame, cam: str, *, fill: int = FILL_FRAMES) -> dict:
    """``{pid: {frame: xy}}`` per id: its ankle ground points, linearly filled across gaps of up
    to ``fill`` frames (beyond that the id has no point on that frame)."""
    raw: dict = defaultdict(dict)
    for (c, f, pid), xy in ankles.items():
        if str(c) == cam:
            raw[int(pid)][int(f)] = np.asarray(xy, float)[:2]
    span = df[df["cam"] == cam].groupby("track_id")["frame"].agg(["min", "max"])
    out: dict = {}
    for pid, d in raw.items():
        fs = np.array(sorted(d))
        if len(fs) < 2 or pid not in span.index:
            continue
        xy = np.stack([d[f] for f in fs])
        grid = np.arange(int(span.loc[pid, "min"]), int(span.loc[pid, "max"]) + 1)
        vals = np.column_stack([np.interp(grid, fs, xy[:, k]) for k in range(2)])
        near = np.abs(grid[:, None] - fs[None, :]).min(axis=1) <= fill
        out[int(pid)] = {int(f): vals[i] for i, f in enumerate(grid) if near[i]}
    return out


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9))


def numbers_of(df: pd.DataFrame, *, min_votes: int = 2) -> dict:
    """``{pid: jersey number}`` from the OCR column, by majority over the id's rows."""
    if "jersey_number_ocr" not in df:
        return {}
    g = df[(df["track_id"] >= 0) & (df["jersey_number_ocr"] >= 0)]
    out = {}
    for pid, rows in g.groupby("track_id")["jersey_number_ocr"]:
        vals, counts = np.unique(rows.to_numpy(int), return_counts=True)
        if counts.max() >= min_votes:
            out[int(pid)] = int(vals[int(np.argmax(counts))])
    return out


def twin_pairs(df: pd.DataFrame, ankles: dict, teams: dict, *, cam: str, max_gap_m: float = MAX_ANKLE_GAP_M,
               min_frames: int = MIN_ANKLE_FRAMES, min_iou: float = MIN_IOU, numbers: dict | None = None) -> list:
    """``[(keep, drop, gap m, iou, frames)]``: pairs of ids of ``cam`` that are one body. The id
    with more boxes is kept. A pid whose team is unknown is never merged, and two ids whose jersey
    numbers were read and disagree are two players however close their feet."""
    numbers = numbers_of(df) if numbers is None else numbers
    g = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    filled = filled_ankles(ankles, df, cam)
    boxes = {(int(r.frame), int(r.track_id)): np.array([r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2], float)
             for r in g.itertuples()}
    by_frame: dict = defaultdict(list)
    for (f, pid) in boxes:
        by_frame[f].append(pid)
    gaps: dict = defaultdict(list)
    ious: dict = defaultdict(list)
    for f, ids in by_frame.items():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if teams.get(a) is None or teams.get(a) != teams.get(b):
                    continue
                na, nb = numbers.get(a), numbers.get(b)
                if na is not None and nb is not None and na != nb:
                    continue
                ious[(a, b)].append(_iou(boxes[(f, a)], boxes[(f, b)]))
                pa, pb = filled.get(a, {}).get(f), filled.get(b, {}).get(f)
                if pa is not None and pb is not None:
                    gaps[(a, b)].append(float(np.linalg.norm(pa - pb)))
    n_box = g.groupby("track_id").size()
    out = []
    for key, d in gaps.items():
        if len(d) < min_frames:
            continue
        gap, iou = float(np.median(d)), float(np.median(ious[key]))
        if gap > max_gap_m or iou < min_iou:
            continue
        a, b = key
        keep, drop = (a, b) if n_box.get(a, 0) >= n_box.get(b, 0) else (b, a)
        out.append((keep, drop, gap, iou, len(d)))
    # one drop per id, and never keep an id that is itself dropped (chains collapse to the longest)
    out.sort(key=lambda r: r[2])
    taken: set = set()
    final = []
    for keep, drop, gap, iou, n in out:
        if keep in taken or drop in taken:
            continue
        taken.update({keep, drop})
        final.append((keep, drop, gap, iou, n))
    return final


def id_map(df: pd.DataFrame, pairs: list) -> dict:
    """``{old id: new id}`` over ALL cameras from ``[(keep, drop, ...)]``. Ids are global -- one
    person carries the same id in both cameras -- so a merge found in one camera must move the
    other camera's rows too, or the cross-camera pairing breaks. Chains collapse to the id with
    the most boxes."""
    parent: dict = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for keep, drop, *_ in pairs:
        ra, rb = find(int(keep)), find(int(drop))
        if ra != rb:
            parent[rb] = ra
    n_box = df[df["track_id"] >= 0].groupby("track_id").size()
    groups: dict = defaultdict(list)
    for pid in list(parent):
        groups[find(pid)].append(pid)
    out = {}
    for members in groups.values():
        rep = max(members, key=lambda p: int(n_box.get(p, 0)))
        for p in members:
            if p != rep:
                out[int(p)] = int(rep)
    return out


def merge_map(df: pd.DataFrame, pairs: list) -> tuple[dict, int]:
    """``({(cam, frame, old id): new id}, boxes dropped)``. Every row is mapped; where a merge
    puts two boxes of one camera on one frame under one id, the less confident is dropped (-1)."""
    drop_to_keep = id_map(df, pairs)
    g = df[df["track_id"] >= 0]
    conf = {(str(r.cam), int(r.frame), int(r.track_id)): float(getattr(r, "conf", 1.0)) for r in g.itertuples()}
    new_of = {k: drop_to_keep.get(k[2], k[2]) for k in conf}
    best: dict = {}
    for k, new in new_of.items():
        cell = (k[0], k[1], new)
        if cell not in best or conf[k] > conf[best[cell]]:
            best[cell] = k
    mapping = {}
    dropped = 0
    for k, new in new_of.items():
        if best[(k[0], k[1], new)] == k:
            mapping[k] = new
        else:
            mapping[k] = -1
            dropped += 1
    return mapping, dropped


def apply_merge(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, int]:
    """``(tracks with the merged ids, rows dropped)``; a row mapped to -1 is dropped."""
    keys = list(zip(df["cam"].astype(str), df["frame"].astype(int), df["track_id"].astype(int)))
    nid = np.array([mapping.get(k, t) for k, t in zip(keys, df["track_id"].astype(int))], int)
    keep = nid >= 0
    out = df.loc[keep].copy()
    out["track_id"] = nid[keep]
    if "global_player_id" in out:
        out["global_player_id"] = nid[keep]
    return out, int((~keep).sum())
