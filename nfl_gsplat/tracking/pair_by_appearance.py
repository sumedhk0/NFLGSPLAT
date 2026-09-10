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
MIN_OVERLAP: int = 6           # 15 -> 6 measured twice on play 1: +5/+6 pairs, +1 player paired per frame, cross-kit still 0
# The offset statistic is the MEAN difference vector over the overlap (noise
# averages out for a true pair). It is blind to two tracks that cross: play 1
# id 9 paired a receiver running 28 m along y with an endzone track standing
# still -- per-frame distances up to 15 m, mean vector 2 m, accepted as a
# number match; the fused refit then triangulated two people. The median
# per-frame distance is the second gate: 1.0-1.9 m for the good pairs on
# play 1, 4.2 and 14.2 for the two wrong ones.
# 4.0 with box-bottom ground points and the wrong clip offset; with the ankle keypoints'
# points and the offset measured (play 1, 2026-09-09) a true pair sits 0.5 m apart at the
# median and the 2.5-4 m "pairs" were another player crossing or the neighbour in the lane.
MAX_MEDIAN_DIST_M: float = 2.0
# The endzone sees the field's y as its lateral axis (a few centimetres); the sideline's y is
# its depth (0.3-0.5 m off). Two tracks 2.4 m apart in y at the median are two people even
# when their mean offset passes the gate (play 1 id 11: 270 frames, dy +2.46 m).
MAX_LATERAL_M: float = 1.2
# Two tracks of ONE camera contesting the same partner's span are one person when they sit
# this close over their common frames (a duplicate box, or a fragment that continues past
# the other): play 1's endzone track 102 ran 206-550, its first 116 frames beside track 96
# (both 0.26 m from sideline 4), and the span rule threw its other 228 frames away.
SAME_PERSON_M: float = 1.0
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
                       min_overlap=MIN_OVERLAP, lag: int = 0, max_median_dist=MAX_MEDIAN_DIST_M,
                       same_person_m=SAME_PERSON_M, max_lateral_m=MAX_LATERAL_M):
    """``[Pair]`` accepted; sideline frame f sits beside endzone frame f + lag."""
    ser_s = [_series(t) for t in side]
    ser_e = [_series(t) for t in end]
    cands = []
    for i, a in enumerate(side):
        for j, b in enumerate(end):
            num_match = a.number >= 0 and b.number >= 0 and a.number == b.number
            if a.number >= 0 and b.number >= 0 and a.number != b.number:
                continue
            # two OCR reads agreeing on a number outrank a kit vote: play 1's endzone
            # track of KC 83 (432 frames, number read) carried the white kit and was
            # never paired
            if a.kit >= 0 and b.kit >= 0 and a.kit != b.kit and not num_match:
                continue
            common = [f for f in ser_s[i] if (f + lag) in ser_e[j]]
            if len(common) < min_overlap:
                continue
            diffs = np.array([ser_e[j][f + lag] - ser_s[i][f] for f in common])
            d = diffs.mean(axis=0)
            off = float(np.linalg.norm(d))
            if float(np.median(np.linalg.norm(diffs, axis=1))) > max_median_dist:
                continue                                       # the tracks cross: not one player
            if float(np.median(np.abs(diffs[:, 1]))) > max_lateral_m:
                continue                                       # apart along the endzone's precise axis
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
    taken_s: dict = {}                     # sideline i -> [endzone j held]
    taken_e: dict = {}                     # endzone j -> [sideline i held]
    kept = []
    for p in cands:
        si, ej = span_s[p.s], span_e[p.e]
        clash = False
        for j2 in taken_s.get(p.s, []):
            o = span_e[j2]
            if not (ej[1] < o[0] or ej[0] > o[1]) and not _same_person(ser_e[p.e], ser_e[j2], same_person_m):
                clash = True
                break
        if clash:
            continue
        for i2 in taken_e.get(p.e, []):
            o = span_s[i2]
            if not (si[1] < o[0] or si[0] > o[1]) and not _same_person(ser_s[p.s], ser_s[i2], same_person_m):
                clash = True
                break
        if clash:
            continue
        taken_s.setdefault(p.s, []).append(p.e)
        taken_e.setdefault(p.e, []).append(p.s)
        kept.append(p)
    return kept


def _same_person(ser_a: dict, ser_b: dict, tol_m: float) -> bool:
    """Two tracks of one camera are one person when they sit within ``tol_m`` of each other
    at the median over their common frames (none in common: not the same)."""
    common = [f for f in ser_a if f in ser_b]
    if len(common) < 3:
        return False
    d = np.linalg.norm(np.array([ser_a[f] - ser_b[f] for f in common]), axis=1)
    return float(np.median(d)) <= tol_m


def global_ids(n_s: int, n_e: int, pairs):
    return global_ids_checked(n_s, n_e, pairs)[:2]


# One person is in one place: two tracks of the SAME camera that overlap in time are two
# people, or one person twice (a ghost tail beside the re-acquired track). Either way one id
# for both puts two boxes on every overlapping frame (the fits skip those frames, the overlay
# draws one) -- play 1's sideline 17 was the centre AND the quarterback behind him for 165
# frames (the waiver's 1 m 'same person' test cannot tell them apart), and the fit sat 50 px
# between the two bodies. A short overlap is a continuation (the tracker re-acquires a body
# while the old track's tail still runs): allowed up to these limits.
CONTINUATION_MAX_FRAMES: int = 45
CONTINUATION_MAX_FRAC: float = 0.5     # ... and at most this fraction of the shorter track's span


def _overlap(a, b) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def global_ids_checked(n_s: int, n_e: int, pairs, spans_s=None, spans_e=None, *,
                       max_frames: int = CONTINUATION_MAX_FRAMES, max_frac: float = CONTINUATION_MAX_FRAC):
    """``(sideline ids, endzone ids, dropped pairs)`` by union-find over ``pairs``. With the
    tracks' spans ((first, last frame) per track, sideline then endzone) a union that would
    put two same-camera tracks overlapping in time beyond a short continuation into one id
    is refused and that pair dropped (in the pairs' order, so the better-ranked pair wins)."""
    parent = list(range(n_s + n_e))
    spans = None if spans_s is None else [tuple(s) for s in spans_s] + [tuple(s) for s in spans_e]
    members: dict = {a: [a] for a in range(n_s + n_e)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def compatible(ra, rb) -> bool:
        if spans is None:
            return True
        for a in members[ra]:
            for b in members[rb]:
                if (a < n_s) != (b < n_s):
                    continue                                   # different cameras
                ov = _overlap(spans[a], spans[b])
                shorter = min(spans[a][1] - spans[a][0] + 1, spans[b][1] - spans[b][0] + 1)
                if ov > max_frames or ov > max_frac * shorter:
                    return False
        return True

    dropped = []
    for p in pairs:
        ra, rb = find(p.s), find(n_s + p.e)
        if ra == rb:
            continue
        if not compatible(ra, rb):
            dropped.append(p)
            continue
        parent[rb] = ra
        members[ra].extend(members.pop(rb))
    roots = [find(a) for a in range(n_s + n_e)]
    order: dict = {}
    for r in roots:
        order.setdefault(r, len(order))
    g = np.array([order[r] for r in roots], int)
    return g[:n_s], g[n_s:], dropped


def cam_tracks_from_frame(df, cams, *, kit_margin: float = 0.4, min_number_votes: int = MIN_NUMBER_VOTES,
                          smooth: int = 15, ankles=None):
    """``(side, end)`` lists of CamTrack from a tracks(_identity).parquet with
    per-camera ids (``cam``, ``track_id``), ``kit_margin`` when 08b wrote it,
    ``jersey_number_ocr`` when 08c's OCR ran (-1 otherwise). ``ankles``
    ``{(cam, frame, track_id): xy}`` (render.play_timeline.ankle_ground) stands
    in for the box-bottom ground point where a camera has the player's ankles:
    the box point sits 0.3-0.5 m off the feet (play 1, against the two cameras'
    triangulated ankles), which the position gate feels."""
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
        pts = np.stack([np.asarray(ankles[(cam, int(f), int(tid))], float)[:2]
                        if ankles is not None and (cam, int(f), int(tid)) in ankles
                        else ground_points((tr.K[f], tr.R[f], tr.t[f]), feet_of(b[None]))[0]
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


STITCH_GAP_S: float = 1.0        # fragments further apart in time are not joined
STITCH_SPEED_M_S: float = 9.0    # a player covers at most this much per second across the gap
STITCH_FLOOR_M: float = 1.0      # and this much regardless (box-bottom noise)


def stitch_by_appearance(tracks: list, *, fps: float = 59.94, gap_s: float = STITCH_GAP_S,
                         speed: float = STITCH_SPEED_M_S, floor_m: float = STITCH_FLOOR_M):
    """Same-camera fragments joined end to start: no time overlap, a gap under
    ``gap_s``, no kit or number conflict, and the later fragment's first point
    within ``floor_m + speed * gap`` of the earlier one's last point (constant
    position across the gap: velocity extrapolation was measured to weld the
    wrong players). A number match on both sides is taken first; then kit
    agreement; a fragment with neither known joins nothing. Greedy by
    (evidence, distance), each end used once. Returns ``[(i_earlier, i_later)]``.

    WHY position alone is not enough (measured 2026-09-04, helmet set):
    position-only stitching cut pieces per player 4.51 -> 4.36 while purity
    fell 0.78 -> 0.75 -- it welded wrong players about as often as right
    ones. The kit and the number are the vetoes that were missing.

    NOT ADOPTED (measured 2026-09-07, play 1 v9, roster-named fragments as
    the truth): sideline 15 joins, 0 right and 1 wrong among named-both
    pairs, 14 with an unnamed side; endzone 11 joins, 1 right, 1 wrong;
    fragments per named player 1.69 -> 1.69 and 1.71 -> 1.68. With the kit
    and number vetoes it still welds a wrong pair per right one on the
    little evidence there is, and most fragments carry neither name nor
    number. Kept as tested code; 08i does not call it."""
    order = sorted(range(len(tracks)), key=lambda i: int(tracks[i].frames.min()))
    cands = []
    for a_idx in range(len(order)):
        a = tracks[order[a_idx]]
        a_end = int(a.frames.max())
        for b_idx in range(len(order)):
            b = tracks[order[b_idx]]
            b_start = int(b.frames.min())
            gap = (b_start - a_end) / fps
            if gap <= 0 or gap > gap_s:
                continue
            if a.kit >= 0 and b.kit >= 0 and a.kit != b.kit:
                continue
            if a.number >= 0 and b.number >= 0 and a.number != b.number:
                continue
            d = float(np.linalg.norm(b.xy[0] - a.xy[-1]))
            if d > floor_m + speed * gap:
                continue
            if a.number >= 0 and b.number >= 0:
                ev = "number"
            elif a.kit >= 0 and b.kit >= 0:
                ev = "kit"
            else:
                continue
            cands.append((0 if ev == "number" else 1, d, order[a_idx], order[b_idx], ev))
    cands.sort()
    used_end, used_start, out = set(), set(), []
    for _, d, i, j, ev in cands:
        if i in used_end or j in used_start:
            continue
        used_end.add(i)
        used_start.add(j)
        out.append((i, j, ev, d))
    return out


def chains_from_joins(n: int, joins):
    """Union-find over ``(i, j, ...)`` joins -> dense group id per track."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, *_ in joins:
        ra, rb = find(i), find(j)
        if ra != rb:
            parent[rb] = ra
    roots = [find(a) for a in range(n)]
    order: dict = {}
    for r in roots:
        order.setdefault(r, len(order))
    return np.array([order[r] for r in roots], int)
