"""Per-camera ground tracks paired across cameras by their trajectories.

WHY. 08b paired the two cameras' detections PER FRAME by ground distance and
linked the fused points through time. Measured on play 2 (2026-09-05): for
the same player the two cameras' foot points differ by 1.4 m (median) with
no systematic offset -- each camera is precise across its view and poor
along it, and a box bottom is a poor foot -- so with players 1-2 m apart the
per-frame pairing was a coin flip: 53 % of the ids seen by both cameras wore
different kits in the two views, only 10-12 of 22 players paired per frame,
the rest rendered twice as one-view ids and never triangulated.

WHAT. Link each camera on its own ground points (tracking.link3d), then pair
TRACKS: the cost of a sideline/endzone pair is the norm of the MEAN signed
offset over their time overlap. For the true pair that mean shrinks with the
overlap (the per-frame error has no bias); for a wrong pair it stays at the
two players' separation. A small frame-offset search absorbs a clip
misalignment; a track's kit (majority of its confident labels) vetoes a
cross-kit pair; fragments of one player may share a partner as long as they
are disjoint in time. Global ids come from the union of accepted pairs;
unpaired tracks keep their own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_OVERLAP: int = 15          # frames a pair must share before its mean offset means anything
MAX_OFFSET_M: float = 1.0      # accept a pair whose mean offset over the overlap is within this
FRAME_OFFSETS = range(-6, 7)   # endzone clip lag searched, frames (sideline f <-> endzone f + k)


@dataclass
class Pair:
    i: int              # sideline track index
    j: int              # endzone track index
    cost: float         # |mean offset| over the overlap, metres
    overlap: int        # frames in common


def camera_placements(det: dict, track, n_frames: int, *, on_field):
    """``det``: ``{frame: boxes [N, 4]}``; ``track``: a CameraTrack with per
    frame K, R, t, conf. Returns ``(placements {f: [M, 2]}, index {f: [M]})``:
    on-field ground points of the box bottoms in frames the camera is solved,
    and the box index each placement came from."""
    from nfl_gsplat.calibration.from_players import feet_of
    from nfl_gsplat.calibration.joint_views import ground_points

    placements, index = {}, {}
    for f in range(n_frames):
        if track.conf[f] <= 0:
            continue
        boxes = det.get(f)
        if boxes is None or len(boxes) == 0:
            continue
        g = ground_points((track.K[f], track.R[f], track.t[f]), feet_of(boxes))
        ok = on_field(g)
        if ok.any():
            placements[f] = g[ok]
            index[f] = np.flatnonzero(ok)
    return placements, index


def _series(track, smoothed_xy):
    return {int(f): p for f, p in zip(track.frames, smoothed_xy)}


def pair_tracks(tracks_s, tracks_e, *, fps: float = 59.94, min_overlap: int = MIN_OVERLAP,
                max_offset_m: float = MAX_OFFSET_M, frame_offsets=FRAME_OFFSETS,
                kit_veto: bool = True, smoothed: bool = True):
    """Returns ``(pairs, frame_offset)``. ``pairs`` is a list of ``Pair``
    accepted greedily by cost under the time-disjointness rule (a track's
    partners never overlap each other in time); ``frame_offset`` is the k
    with the most accepted pairs (ties: lowest mean cost), meaning sideline
    frame f sits beside endzone frame f + k."""
    from nfl_gsplat.tracking.link3d import smooth

    ser_s = [_series(t, smooth(t, fps=fps) if smoothed else np.asarray(t.xy)) for t in tracks_s]
    ser_e = [_series(t, smooth(t, fps=fps) if smoothed else np.asarray(t.xy)) for t in tracks_e]
    lab_s = [t.label for t in tracks_s]
    lab_e = [t.label for t in tracks_e]
    span_s = [(min(t.frames), max(t.frames)) for t in tracks_s]
    span_e = [(min(t.frames), max(t.frames)) for t in tracks_e]

    def candidates(k):
        out = []
        for i, a in enumerate(ser_s):
            for j, b in enumerate(ser_e):
                if kit_veto and lab_s[i] >= 0 and lab_e[j] >= 0 and lab_s[i] != lab_e[j]:
                    continue
                lo = max(span_s[i][0], span_e[j][0] - k)
                hi = min(span_s[i][1], span_e[j][1] - k)
                if hi - lo + 1 < min_overlap:
                    continue
                common = [f for f in range(lo, hi + 1) if f in a and (f + k) in b]
                if len(common) < min_overlap:
                    continue
                d = np.mean([b[f + k] - a[f] for f in common], axis=0)
                cost = float(np.linalg.norm(d))
                if cost <= max_offset_m:
                    out.append(Pair(i, j, cost, len(common)))
        return out

    def accept(cands):
        cands = sorted(cands, key=lambda p: (p.cost, -p.overlap))
        taken_s: dict[int, list] = {}
        taken_e: dict[int, list] = {}
        kept = []
        for p in cands:
            si = (span_s[p.i][0], span_s[p.i][1])
            ej = (span_e[p.j][0], span_e[p.j][1])
            # the new partner must be time-disjoint from every existing partner of each side
            if any(not (ej[1] < o[0] or ej[0] > o[1]) for o in taken_s.get(p.i, [])):
                continue
            if any(not (si[1] < o[0] or si[0] > o[1]) for o in taken_e.get(p.j, [])):
                continue
            taken_s.setdefault(p.i, []).append(ej)
            taken_e.setdefault(p.j, []).append(si)
            kept.append(p)
        return kept

    best, best_k = [], 0
    for k in frame_offsets:
        kept = accept(candidates(k))
        if (len(kept), -np.mean([p.cost for p in kept]) if kept else 0.0) > \
           (len(best), -np.mean([p.cost for p in best]) if best else 0.0):
            best, best_k = kept, k
    return best, best_k


def global_ids(n_s: int, n_e: int, pairs):
    """Union-find over the pairs: ``(gid_s [n_s], gid_e [n_e])``, ids dense from 0."""
    parent = list(range(n_s + n_e))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in pairs:
        ra, rb = find(p.i), find(n_s + p.j)
        if ra != rb:
            parent[rb] = ra
    roots = [find(a) for a in range(n_s + n_e)]
    order = {}
    for r in roots:
        order.setdefault(r, len(order))
    gids = np.array([order[r] for r in roots], int)
    return gids[:n_s], gids[n_s:]
