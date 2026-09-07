"""Per-camera tracks paired across cameras by appearance first, position second.

WHY. Pairing on box bottoms alone is ambiguous at about 1 m -- each
camera's ground point is poor along its own depth and players stand 1-2 m
apart -- so on play 1 the pairs made were 23-35 % cross-kit whatever the
gap, and a better endzone camera track did not change that. What the crops
carry that position does not: the kit (2-5 % wrong per box, near-certain
per track) and the jersey number (75 % per track). Two tracks that read the
same number in the same kit are the same player wherever their ground
points sit; two tracks in different kits, or with different numbers read
with confidence, are never the same player.

WHAT. For every sideline/endzone pair of per-camera tracks with a time
overlap: veto on a kit conflict or a number conflict; score the mean signed
offset over the overlap (``pair_tracks``' statistic); accept a number match
within ``gap_number`` metres, a kit-consistent pair within ``gap_position``;
greedy by (evidence, offset) under the time-disjoint rule; union-find gives
the global ids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAP_NUMBER_M: float = 3.5      # a number read in both cameras: position only has to be plausible
GAP_POSITION_M: float = 2.0    # no number: the kit must agree and the mean offset be this close
MIN_OVERLAP: int = 15
MIN_NUMBER_VOTES: int = 2      # OCR rows a track needs before its number counts as read


@dataclass
class CamTrack:
    cam: str
    tid: int
    frames: np.ndarray            # sorted
    xy: np.ndarray                # [n, 2] ground points, smoothed
    kit: int                      # 0 white, 1 coloured, -1 unknown
    number: int                   # -1 unknown


@dataclass
class Pair:
    s: int
    e: int
    evidence: str                 # "number" | "kit"
    offset: float
    overlap: int


def _series(t: CamTrack):
    return dict(zip(t.frames.tolist(), t.xy))


def pair_by_appearance(side: list, end: list, *, gap_number=GAP_NUMBER_M, gap_position=GAP_POSITION_M,
                       min_overlap=MIN_OVERLAP, lag: int = 0):
    """``[Pair]`` accepted; sideline frame f sits beside endzone frame f + lag."""
    ser_s = [_series(t) for t in side]
    ser_e = [_series(t) for t in end]
    cands = []
    for i, a in enumerate(side):
        for j, b in enumerate(end):
            if a.kit >= 0 and b.kit >= 0 and a.kit != b.kit:
                continue
            if a.number >= 0 and b.number >= 0 and a.number != b.number:
                continue
            common = [f for f in ser_s[i] if (f + lag) in ser_e[j]]
            if len(common) < min_overlap:
                continue
            d = np.mean([ser_e[j][f + lag] - ser_s[i][f] for f in common], axis=0)
            off = float(np.linalg.norm(d))
            if a.number >= 0 and b.number >= 0:
                if off <= gap_number:
                    cands.append(Pair(i, j, "number", off, len(common)))
            elif a.kit >= 0 and b.kit >= 0:
                if off <= gap_position:
                    cands.append(Pair(i, j, "kit", off, len(common)))
            # a pair with neither kit nor number known on one side is not made:
            # position alone was measured a coin flip
    rank = {"number": 0, "kit": 1}
    cands.sort(key=lambda p: (rank[p.evidence], p.offset, -p.overlap))
    span_s = [(int(t.frames.min()), int(t.frames.max())) for t in side]
    span_e = [(int(t.frames.min()), int(t.frames.max())) for t in end]
    taken_s: dict = {}
    taken_e: dict = {}
    kept = []
    for p in cands:
        si, ej = span_s[p.s], span_e[p.e]
        if any(not (ej[1] < o[0] or ej[0] > o[1]) for o in taken_s.get(p.s, [])):
            continue
        if any(not (si[1] < o[0] or si[0] > o[1]) for o in taken_e.get(p.e, [])):
            continue
        taken_s.setdefault(p.s, []).append(ej)
        taken_e.setdefault(p.e, []).append(si)
        kept.append(p)
    return kept


def global_ids(n_s: int, n_e: int, pairs):
    parent = list(range(n_s + n_e))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in pairs:
        ra, rb = find(p.s), find(n_s + p.e)
        if ra != rb:
            parent[rb] = ra
    roots = [find(a) for a in range(n_s + n_e)]
    order: dict = {}
    for r in roots:
        order.setdefault(r, len(order))
    g = np.array([order[r] for r in roots], int)
    return g[:n_s], g[n_s:]


def cam_tracks_from_frame(df, cams, *, kit_margin: float = 0.4, min_number_votes: int = MIN_NUMBER_VOTES,
                          smooth: int = 15):
    """``(side, end)`` lists of CamTrack from a tracks(_identity).parquet with
    per-camera ids (``cam``, ``track_id``), ``kit_margin`` when 08b wrote it,
    ``jersey_number_ocr`` when 08c's OCR ran (-1 otherwise)."""
    import pandas as pd

    from nfl_gsplat.calibration.from_players import feet_of
    from nfl_gsplat.calibration.joint_views import ground_points

    B = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    out = {"sideline": [], "endzone": []}
    for (cam, tid), g in df.groupby(["cam", "track_id"]):
        cam = str(cam)
        if cam not in out or int(tid) < 0:
            continue
        tr = cams[cam]
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy(int)
        ok = np.array([tr.conf[f] > 0 for f in fr])
        if ok.sum() < 3:
            continue
        pts = np.stack([ground_points((tr.K[f], tr.R[f], tr.t[f]), feet_of(b[None]))[0]
                        for f, b in zip(fr[ok], g[B].to_numpy()[ok])])
        fin = np.isfinite(pts).all(1)
        fr, pts = fr[ok][fin], pts[fin]
        if len(fr) < 3:
            continue
        if smooth > 1 and len(fr) >= smooth:
            ker = np.ones(smooth) / smooth
            pts = np.column_stack([np.convolve(pts[:, k], ker, mode="same") for k in range(2)])
        kit = -1
        if "kit_margin" in g:
            m = g["kit_margin"].to_numpy(float)
            m = m[np.isfinite(m) & (np.abs(m) >= kit_margin)]
            if len(m) >= 3:
                kit = int((m > 0).mean() > 0.5)
        number = -1
        if "jersey_number_ocr" in g:
            n = g["jersey_number_ocr"].to_numpy(int)
            n = n[n >= 0]
            if len(n) >= min_number_votes:
                vals, counts = np.unique(n, return_counts=True)
                number = int(vals[np.argmax(counts)])
        out[cam].append(CamTrack(cam, int(tid), fr, pts, kit, number))
    return out["sideline"], out["endzone"]
