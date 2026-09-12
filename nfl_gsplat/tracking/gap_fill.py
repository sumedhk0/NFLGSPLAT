"""Boxes for the frames a track loses, from detections the threshold rejected.

WHY. Counting bodies per team is the one check that needs no labels: eleven a
side. At play 1's snap the sideline camera holds 8 of Kansas City's 11, and the
render can only draw what the sideline tracks. The men it loses are the linemen
in the pile, and the detector does see them -- at frame 300 the two boxes that
appear only below its 0.35 threshold are a lineman half hidden behind another, at
confidence 0.17 and 0.13.

Lowering the threshold everywhere is not the answer: late in the play the camera
pans and the sideline crowd enters frame, where 0.35 already finds 42 people and
0.10 finds 76. So the low-confidence detections are used ONLY to fill a gap in a
track that already exists: a frame inside a track's own life where it has no box,
with a detection near where the track must be. No new identities are created, and
every filled box inherits the track it belongs to.

WHAT. ``fill_gaps`` takes the tracks table and a per-frame pool of low-confidence
detections, finds each track's interior gaps up to ``MAX_GAP`` frames, predicts
the box across the gap (linear between the frames either side) and accepts the
detection with the best overlap above ``MIN_IOU`` that no other track claims more
strongly. Filled rows carry ``filled = True``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP: int = 30          # frames; a longer hole is a track that ended, not a missed detection
MIN_IOU: float = 0.35      # the detection must sit where the track should be
BOX = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9))


def track_gaps(frames, *, max_gap: int = MAX_GAP):
    """``[(before, after)]`` frame pairs bounding each interior hole of a track."""
    fr = np.asarray(sorted(int(f) for f in frames), int)
    out = []
    for a, b in zip(fr[:-1], fr[1:]):
        if 1 < b - a <= max_gap + 1:
            out.append((int(a), int(b)))
    return out


def predict_box(box_a, box_b, f, fa, fb) -> np.ndarray:
    """The track's box at ``f``, linear between its boxes at ``fa`` and ``fb``."""
    w = 0.0 if fb == fa else (f - fa) / (fb - fa)
    return np.asarray(box_a, float) * (1 - w) + np.asarray(box_b, float) * w


def fill_gaps(df: pd.DataFrame, pool: dict, *, cam: str, max_gap: int = MAX_GAP, min_iou: float = MIN_IOU):
    """``(rows to add, per-track counts)``. ``df`` is tracks.parquet; ``pool`` ``{frame: boxes [N, 4]}``
    the low-confidence detections of that camera. A detection is taken by the track whose predicted
    box overlaps it most, and only once."""
    g = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    by_track = {int(pid): rows.sort_values("frame") for pid, rows in g.groupby("track_id")}
    taken: dict = {}
    add = []
    counts: dict = {}
    wanted: dict = {}                                    # frame -> [(iou, pid, box, predicted)]
    for pid, rows in by_track.items():
        fr = rows["frame"].to_numpy(int)
        boxes = rows[BOX].to_numpy(float)
        at = {int(f): boxes[i] for i, f in enumerate(fr)}
        for fa, fb in track_gaps(fr, max_gap=max_gap):
            for f in range(fa + 1, fb):
                cands = pool.get(int(f))
                if cands is None or not len(cands):
                    continue
                pred = predict_box(at[fa], at[fb], f, fa, fb)
                sc = [(iou(pred, b), k) for k, b in enumerate(np.asarray(cands, float))]
                sc.sort(reverse=True)
                if sc and sc[0][0] >= min_iou:
                    wanted.setdefault(int(f), []).append((sc[0][0], int(pid), int(sc[0][1]), pred))
    for f, want in wanted.items():
        want.sort(reverse=True)
        used: set = set()
        for score, pid, k, pred in want:
            if k in used or (f, k) in taken:
                continue
            used.add(k)
            taken[(f, k)] = pid
            b = np.asarray(pool[f], float)[k]
            add.append({"cam": cam, "frame": int(f), "track_id": int(pid), "bbox_x1": float(b[0]),
                        "bbox_y1": float(b[1]), "bbox_x2": float(b[2]), "bbox_y2": float(b[3]),
                        "conf": float("nan"), "filled": True})
            counts[int(pid)] = counts.get(int(pid), 0) + 1
    return add, counts
