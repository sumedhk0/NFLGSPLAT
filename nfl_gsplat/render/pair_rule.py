"""A two-camera id whose two tracks are not one player is drawn from the sideline alone.

WHY. The appearance pairing (08i) accepted play 1's id 9 -- a receiver
running 28 m along y in the sideline camera, an endzone track standing
still -- because its offset statistic was the mean difference vector,
which averages out when tracks cross. The timeline then averaged the two
cameras' ground points (garbage 7 m from either), refused the refit's
placement against that average, interpolated between refused records,
and the avatar sawtoothed by a metre every frame. The pairing has the
median-distance gate now; a play-dir paired before it gets this guard.

WHAT. For every id both cameras see, the median per-frame distance between
the sideline's and the endzone's ground points over the overlap; over
``MAX_MEDIAN_DIST_M`` the id's endzone rows are dropped (ground points,
views) and its two-view pose records must not be trusted (05p refits it
one-view when the fused cache no longer carries it).
"""
from __future__ import annotations

import numpy as np

MAX_MEDIAN_DIST_M: float = 4.0
MIN_OVERLAP: int = 6


def mispaired_ids(ground_side, ground_end, *, max_median_dist: float = MAX_MEDIAN_DIST_M,
                  min_overlap: int = MIN_OVERLAP):
    """``{pid: median distance}`` for ids whose sideline and endzone ground
    points (frame -> {pid: xy}, one dict per camera) sit further apart than
    ``max_median_dist`` at the median over at least ``min_overlap`` frames."""
    by_pid: dict = {}
    for f, d in ground_side.items():
        e = ground_end.get(f)
        if not e:
            continue
        for pid, xy in d.items():
            if pid in e:
                by_pid.setdefault(int(pid), []).append(float(np.hypot(*(np.asarray(xy, float) - np.asarray(e[pid], float)))))
    return {pid: float(np.median(v)) for pid, v in by_pid.items()
            if len(v) >= min_overlap and float(np.median(v)) > max_median_dist}
