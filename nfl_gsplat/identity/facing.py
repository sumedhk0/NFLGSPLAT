"""Which detections show a jersey number to the camera, from the pose.

WHY. Jersey OCR reads maybe one crop in ten, and the crops it is given are
chosen by box HEIGHT (jersey_ocr.vote_jersey_numbers, top-k largest). A tall
box of a player side-on to the camera carries no number; the numbers are on
the chest and the back. The pose stage already knows where each body faces:
SMPL-X ``global_orient`` in the camera frame. This turns that into a score
per posed detection so the OCR budget goes to crops that can read.

HOW. SMPL-X's canonical body faces +Z. ``forward = R(global_orient) @ z``;
the ray from the camera to the body is the body's camera-frame position,
normalised. ``score = |forward . ray|``: 1 facing the camera or facing away
(chest or back to the lens), 0 side-on. Poses exist on a stride of frames;
``nearest`` looks up the closest posed frame within a gap.
"""
from __future__ import annotations

import numpy as np

PELVIS = 0


def facing_score(global_orient, position_cam) -> float:
    """``|forward . ray|`` for one body: 1 = chest or back square to the lens."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(global_orient, float).reshape(3, 1))
    forward = R @ np.array([0.0, 0.0, 1.0])
    ray = np.asarray(position_cam, float).reshape(3)
    n = np.linalg.norm(ray)
    if not np.isfinite(n) or n < 1e-6:
        return float("nan")
    return float(abs(forward @ (ray / n)))


def facing_table(cache: dict) -> dict[tuple[int, int], float]:
    """``{(frame, track_id): score}`` from a 05c pose cache's ``frames``
    (``rec['global_orient']``, ``rec['joints3d_cam']`` or ``rec['transl']``)."""
    out = {}
    for f, recs in cache.items():
        for tid, rec in recs.items():
            pos = rec.get("joints3d_cam")
            pos = (np.asarray(pos, float)[PELVIS] if pos is not None
                   else np.asarray(rec.get("transl"), float))
            out[(int(f), int(tid))] = facing_score(rec["global_orient"], pos)
    return out


def nearest(table: dict, frame: int, tid: int, *, max_gap: int = 6) -> float:
    """Score of the nearest posed frame of this track within ``max_gap``, else NaN."""
    best, best_gap = float("nan"), max_gap + 1
    for d in range(0, max_gap + 1):
        for f in ((frame - d, frame + d) if d else (frame,)):
            s = table.get((int(f), int(tid)))
            if s is not None and d < best_gap:
                best, best_gap = s, d
        if best_gap <= d:
            break
    return best
