"""The football in the film: a small blob that moves against the camera-compensated background, outside
every player's box, along a near-straight line in the image for the frames of its flight.

WHY. The ball has always been placed from hand-typed events (08y: release, catch, receiver), and on
2026-09-18 the receiver typed was the wrong man for a day. Nothing in the pipeline looked at the film
for the ball. This does: the flight is the stretch of frames on which a blob moves outside the boxes,
the passer is the box it leaves, the receiver the box it enters. It cannot see the ball inside a box
(the hands, the pocket), so it names the ends by extrapolation into the nearest box.

The pure parts live here (tested); scripts/09a_ball_in_film.py does the video work.
"""
from __future__ import annotations

import numpy as np

MIN_SPEED_PX: float = 6.0     # a ball in flight moves at least this per frame in the image (play 1: 17-19 px)
MAX_SPEED_PX: float = 60.0
TOL_PX: float = 14.0          # a candidate within this of the line is on the flight
MIN_INLIERS: int = 5          # a flight needs this many frames with a blob on the line
MIN_SPAN: int = 5             # ... spanning at least this many frames


MIN_DENSITY: float = 0.3      # inlier frames over the flight's span: a real flight is seen on a third of its frames at least
MAX_GAP: int = 12             # ... and never goes unseen longer than this (play 1: 10 frames behind the pocket's boxes)
GROW_ROUNDS: int = 2          # refit a quadratic to the inliers and re-collect within tol: the arc bends off the line


def _flight_from(inl: dict, speed: float, *, min_span: int) -> dict | None:
    fs = np.array(sorted(inl)); xs = np.array([inl[f][1][0] for f in fs]); ys = np.array([inl[f][1][1] for f in fs])
    if fs[-1] - fs[0] < min_span:
        return None
    deg = 2 if len(fs) >= 6 else 1
    return {"frames": [int(f) for f in fs], "x": np.polyfit(fs, xs, deg), "y": np.polyfit(fs, ys, deg), "speed": float(speed), "n": int(len(fs))}


def _grow(fl: dict, pts: list, *, tol: float, min_span: int, rounds: int = GROW_ROUNDS) -> dict:
    """Re-collect the candidates within ``tol`` of the fitted curve and refit, ``rounds`` times: the line
    of a RANSAC pair reaches part of a bending flight, the curve reaches the rest."""
    for _ in range(rounds):
        inl: dict = {}
        for f, x, y in pts:
            d = float(np.hypot(np.polyval(fl["x"], f) - x, np.polyval(fl["y"], f) - y))
            if d <= tol and d < inl.get(f, (tol + 1, None))[0]:
                inl[f] = (d, (x, y))
        if len(inl) <= fl["n"]:
            break
        fs = sorted(inl)
        sp = float(np.hypot(np.polyval(fl["x"], fs[-1]) - np.polyval(fl["x"], fs[0]), np.polyval(fl["y"], fs[-1]) - np.polyval(fl["y"], fs[0])) / max(1, fs[-1] - fs[0]))
        grown = _flight_from(inl, sp, min_span=min_span)
        if grown is None:
            break
        fl = grown
    return fl


def dense_enough(frames, *, min_density: float = MIN_DENSITY, max_gap: int = MAX_GAP) -> bool:
    fs = sorted(int(f) for f in frames)
    span = fs[-1] - fs[0] + 1
    gaps = np.diff(fs) if len(fs) > 1 else np.array([0])
    return len(fs) / float(span) >= min_density and int(gaps.max()) <= max_gap


def candidate_flights(cands: dict, *, min_speed: float = MIN_SPEED_PX, max_speed: float = MAX_SPEED_PX, tol: float = TOL_PX,
                      min_inliers: int = MIN_INLIERS, min_span: int = MIN_SPAN) -> list[dict]:
    """Every distinct straight constant-speed image track through ``{frame: [(x, y, area), ...]}`` with
    at least ``min_inliers`` frames holding a candidate within ``tol`` (RANSAC over candidate pairs on
    frames ``min_span`` or more apart; tracks sharing their inlier frames collapse to the fuller one),
    each grown along a quadratic in the frame (the arc and the perspective bend the flight off any one
    line: play 1's flight is 23 px off the line through its ends at mid-flight) and kept only when its
    inlier frames are dense (dense_enough: a junk line through blobs 80 frames apart is not a flight),
    most inliers first. ``x``/``y`` are np.polyval coefficients."""
    pts = [(int(f), float(x), float(y)) for f, cs in cands.items() for x, y, *_ in cs]
    if len(pts) < min_inliers:
        return []
    found: dict = {}
    for i, (f1, x1, y1) in enumerate(pts):
        for f2, x2, y2 in pts[i + 1:]:
            df = f2 - f1
            if abs(df) < min_span:
                continue
            vx, vy = (x2 - x1) / df, (y2 - y1) / df
            sp = float(np.hypot(vx, vy))
            if not (min_speed <= sp <= max_speed):
                continue
            inl: dict = {}
            for f, x, y in pts:
                d = float(np.hypot(x1 + vx * (f - f1) - x, y1 + vy * (f - f1) - y))
                if d <= tol and d < inl.get(f, (tol + 1, None))[0]:
                    inl[f] = (d, (x, y))
            if len(inl) < min_inliers:
                continue
            key = (min(inl), max(inl))
            if key not in found or len(inl) > len(found[key][0]):
                found[key] = (inl, sp)
    out = []
    for inl, sp in found.values():
        fl = _flight_from(inl, sp, min_span=min_span)
        if fl is None:
            continue
        fl = _grow(fl, pts, tol=tol, min_span=min_span)
        if dense_enough(fl["frames"]):
            out.append(fl)
    # a track whose frames lie inside another's is that track seen shorter
    out.sort(key=lambda fl: -fl["n"])
    kept: list = []
    for fl in out:
        s = set(fl["frames"])
        if any(s <= set(k["frames"]) for k in kept):
            continue
        kept.append(fl)
    return kept


def fit_flight(cands: dict, boxes_by_frame: dict | None = None, *, min_speed: float = MIN_SPEED_PX, max_speed: float = MAX_SPEED_PX,
               tol: float = TOL_PX, min_inliers: int = MIN_INLIERS, min_span: int = MIN_SPAN, reach: int = 20) -> dict | None:
    """The ball's flight among candidate_flights: with ``boxes_by_frame`` the track that leaves a box and
    enters a box (name_ends) beats one that does not -- a pass goes from a hand to a hand, while a
    player without a box moves in the open for as long as he likes (play 1: a far-field runner at 14
    px/frame outscored the real flight by inliers alone) -- then the most inliers. None when nothing
    moves like a ball."""
    fls = candidate_flights(cands, min_speed=min_speed, max_speed=max_speed, tol=tol, min_inliers=min_inliers, min_span=min_span)
    if not fls:
        return None
    if boxes_by_frame is None:
        return fls[0]

    def score(fl):
        e = name_ends(fl, boxes_by_frame, reach=reach)
        return ((e["passer"] is not None) + (e["receiver"] is not None), fl["n"])
    return max(fls, key=score)


def track_at(flight: dict, f: int) -> tuple[float, float]:
    return float(np.polyval(flight["x"], f)), float(np.polyval(flight["y"], f))


def boxes_holding(boxes, x: float, y: float, *, pad: float = 0.0) -> list:
    """The ids of the boxes (``[(pid, x1, y1, x2, y2), ...]``) containing (x, y) grown by ``pad``,
    smallest box first."""
    hits = [(float((x2 - x1) * (y2 - y1)), pid) for pid, x1, y1, x2, y2 in boxes
            if x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad]
    return [pid for _a, pid in sorted(hits)]


def box_holding(boxes, x: float, y: float, *, pad: float = 0.0, teams: dict | None = None, prefer=None):
    """The id of the box containing (x, y): of ``prefer``'s team when ``teams`` names one among the
    hits (a completed pass ends in the offence's hands; in tight coverage the defender's box holds the
    same point -- play 1: BAL 55 draped on the receiver), else the smallest; None when none."""
    hits = boxes_holding(boxes, x, y, pad=pad)
    if not hits:
        return None
    if teams is not None and prefer is not None:
        own = [pid for pid in hits if teams.get(pid) == prefer]
        if own:
            return own[0]
    return hits[0]


def name_ends(flight: dict, boxes_by_frame: dict, *, reach: int = 20, pad: float = 4.0, teams: dict | None = None) -> dict:
    """Walk the fitted track backwards from its first frame until it lies inside a box (the passer's
    hand: the release frame and id) and forwards from its last until it does again (the receiver's
    hands: the catch frame and id); ``reach`` frames each way at most. ``boxes_by_frame`` is
    ``{frame: [(pid, x1, y1, x2, y2), ...]}``. With ``teams`` ({pid: team}) the receiver is the passer's
    teammate among the boxes holding the catch point; ``others`` lists the other boxes holding it.
    Missing ends are None."""
    f0, f1 = flight["frames"][0], flight["frames"][-1]
    out = {"release": None, "passer": None, "catch": None, "receiver": None, "others": []}
    for f in range(f0, f0 - reach - 1, -1):
        x, y = track_at(flight, f)
        pid = box_holding(boxes_by_frame.get(f, []), x, y, pad=pad)
        if pid is not None:
            out["release"], out["passer"] = f, pid
            break
    offence = teams.get(out["passer"]) if (teams is not None and out["passer"] is not None) else None
    for f in range(f1, f1 + reach + 1):
        x, y = track_at(flight, f)
        pid = box_holding(boxes_by_frame.get(f, []), x, y, pad=pad, teams=teams, prefer=offence)
        if pid is not None:
            out["catch"], out["receiver"] = f, pid
            out["others"] = [q for q in boxes_holding(boxes_by_frame.get(f, []), x, y, pad=pad) if q != pid]
            break
    return out


def nearest_box(boxes, x: float, y: float):
    """``(pid, distance)`` of the box whose centre is nearest (x, y); (None, inf) without boxes."""
    best = (None, float("inf"))
    for pid, x1, y1, x2, y2 in boxes:
        d = float(np.hypot(0.5 * (x1 + x2) - x, 0.5 * (y1 + y2) - y))
        if d < best[1]:
            best = (pid, d)
    return best
