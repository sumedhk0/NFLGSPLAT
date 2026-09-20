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

# Per FRAME: the endzone camera's depth (the field's x) wanders by up to 1.4 m while it pans during the play
# (2026-09-18: the offset endzone-minus-sideline is the same for still bodies as for runners in every 30-frame
# bin, so it is the pose, not a time offset), and that common-mode offset must not read as a mispair. What is
# left after the frame's median offset is removed is the row's own disagreement: a lineman whose endzone row is
# another man's (play 1 id 170: 2-3 m over 520-568) sits far beyond it. Over MISPAIR_FRAME_M the endzone row is
# dropped for that frame alone (the sideline draws him) and the two-view refit is not trusted there.
# MEASURED AND REJECTED on play 1 (2026-09-18, scratchpad/probe_mispair_frames_ab.py, play window 395-607): at
# 1.0 m the endzone ankle ruler did not move (du p50 35.3 -> 35.4 px, p90 104 -> 102) while the fast steps went
# 5 -> 15 and the census 1.25 -> 1.28: vetoing frame by frame flips a body between its two-view and its
# sideline-only placement from one frame to the next, and every flip is a step along the sideline's depth.
# The thirteenth correction that lost to what it corrected. Opt-in (None = off); a per-STRETCH veto might not
# flip, but the ruler shows nothing to win.
MISPAIR_FRAME_M: float | None = None
MIN_PAIRS_FOR_COMMON_MODE: int = 4    # fewer two-view bodies on a frame and no common mode is removed


def mispaired_frames(ground_side, ground_end, *, max_m: float = MISPAIR_FRAME_M, min_pairs: int = MIN_PAIRS_FOR_COMMON_MODE):
    """``{(frame, pid): residual}`` for the (frame, id) pairs whose endzone ground point sits further than
    ``max_m`` from the sideline's AFTER the frame's common-mode offset (the median endzone-minus-sideline
    vector over that frame's pairs, when there are ``min_pairs`` or more) is taken out. The common mode is
    the endzone camera's pose error on that frame; the residual is the row's own."""
    out: dict = {}
    for f, d in ground_side.items():
        e = ground_end.get(f)
        if not e:
            continue
        pairs = [(int(pid), np.asarray(e[pid], float)[:2] - np.asarray(xy, float)[:2]) for pid, xy in d.items() if pid in e]
        if not pairs:
            continue
        cm = np.median(np.stack([v for _p, v in pairs]), axis=0) if len(pairs) >= min_pairs else np.zeros(2)
        for pid, v in pairs:
            r = float(np.hypot(*(v - cm)))
            if r > max_m:
                out[(int(f), pid)] = r
    return out


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


# The same common mode as a CORRECTION rather than a veto. Measured on play 1 (2026-09-19 22:10, the play window
# 395-607, scratchpad/probe_ez_common_mode.py): the median sideline-minus-endzone offset over the frame's two-view
# bodies is +0.45 m along the field on every frame (p10 +0.31, p90 +0.63; 80 of 213 frames above 0.5 m) with the
# across offset drifting from 0 to -0.9 m after the catch as the endzone camera pans -- a bias of the endzone
# camera's ground points, not noise. A body the endzone alone places (an endzone-only frame, a beyond-span stretch)
# carries it whole; a two-view body does not (the sideline's point overrides). Shifting endzone-only points by the
# frame's common mode puts them where the sideline would have.
EZ_COMMON_MODE: bool = True           # adopted 2026-09-19 (v85): the seam 1.29 -> 0.91 m, the sideline blend closer at every checked frame
EZ_COMMON_MODE_MIN_PAIRS: int = 4      # fewer two-view bodies on a frame: the play-wide median offset is used


def common_mode_offsets(ground_side, ground_end, *, min_pairs: int = EZ_COMMON_MODE_MIN_PAIRS):
    """``({frame: offset}, global)``: per frame the median sideline-minus-endzone vector over the ids both cameras
    place (frames with fewer than ``min_pairs`` get the median over all frames' offsets, ``global``)."""
    per: dict = {}
    for f, d in ground_side.items():
        e = ground_end.get(f)
        if not e:
            continue
        v = [np.asarray(xy, float)[:2] - np.asarray(e[pid], float)[:2] for pid, xy in d.items() if pid in e]
        if len(v) >= min_pairs:
            per[int(f)] = np.median(np.stack(v), axis=0)
    glob = np.median(np.stack(list(per.values())), axis=0) if per else np.zeros(2)
    return per, glob


def common_mode_shift(ground, views, ground_side, ground_end, *, min_pairs: int = EZ_COMMON_MODE_MIN_PAIRS,
                      endzone: str = "endzone") -> dict:
    """Move every endzone-only point in ``ground`` (``views[f][pid] == (endzone,)``) by its frame's common-mode
    offset (``common_mode_offsets``; the global median where the frame has too few pairs). In place; returns
    ``{"moved": n, "frames_own": n, "global": (dx, dy)}``."""
    per, glob = common_mode_offsets(ground_side, ground_end, min_pairs=min_pairs)
    moved = 0; own = set()
    for f, d in ground.items():
        vf = views.get(f, {})
        off = per.get(int(f))
        for pid in list(d):
            if tuple(vf.get(pid, ())) != (endzone,):
                continue
            o = off if off is not None else glob
            d[pid] = np.asarray(d[pid], float) + np.array([o[0], o[1]] + [0.0] * (len(np.asarray(d[pid]).ravel()) - 2), float)[:len(np.asarray(d[pid]).ravel())]
            moved += 1
            if off is not None:
                own.add(int(f))
    return {"moved": moved, "frames_own": len(own), "global": (float(glob[0]), float(glob[1]))}
