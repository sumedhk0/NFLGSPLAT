"""Bodies seen by the endzone camera alone are not drawn: they are the
sideline's players seen from the other side, unpaired.

WHY. The sideline camera is the precise one (refined to the paint every
frame, players static to 0.04 m pre-snap); the endzone camera's ground
points are poor along the field (its depth). Cross-camera pairing on box
bottoms is ambiguous at about 1 m -- players stand 1-2 m apart -- so a
third to a half of the players stay unpaired per frame in each camera and
the timeline drew both copies: on play 1 with the footage-driven endzone
track, 37 ids a frame for 22 players and 7 officials, 18 of them
endzone-only, each a ghost metres from the sideline's copy at the
endzone's x. One avatar per sideline track is the right count; the endzone
still pairs where it can, for triangulation and depth.

WHAT. An id whose boxes are all from the endzone camera and which is never
a two-view id is a ghost when the sideline camera SHOULD have seen it: at
least ``MIN_INSIDE`` of its frames project inside the sideline image. The
sideline's 12 deg lens follows the ball, so deep and far-side players are
endzone-only for real -- play 1 v9 dropped them with the ghosts and drew
five red avatars where KC had eleven (2026-09-07); with the frustum test
they stay. Without ground points and the sideline camera the rule falls
back to dropping every endzone-only id.
"""
from __future__ import annotations

import numpy as np

MIN_INSIDE: float = 0.5        # share of an id's frames inside the sideline image to call it a ghost
MARGIN_PX: float = 20.0        # inside means this far from the frame edge


def _inside_sideline(xy, track, f, *, margin: float = MARGIN_PX) -> bool:
    if track.conf[f] <= 0:
        return False
    X = np.array([float(xy[0]), float(xy[1]), 0.0])
    p = track.K[f] @ (track.R[f] @ X + track.t[f])
    if p[2] <= 0:
        return False
    u, v = p[0] / p[2], p[1] / p[2]
    return margin <= u <= track.width - margin and margin <= v <= track.height - margin


SAME_BODY_M: float = 1.2         # a sideline body this close is the same man under another id


HOLD_M: float | None = 0.8       # a beyond-span stretch whose join jumps farther than this from where the sideline first (or
                                 # last) has the man is the endzone's depth error, not the man: dropped


def beyond_sideline_span(ground, df, sideline, *, gap: int = 30, cam: str = "sideline",
                         margin: float = MARGIN_PX, side_ground=None, same_body_m: float = SAME_BODY_M,
                         hold_m: float | None = HOLD_M, report: dict | None = None):
    """``ground`` (frame -> {pid: xy}) without the frames of an id that lie
    beyond its sideline detections by more than ``gap`` frames, where the
    sideline could see the spot. Returns ``(ground, dropped)``.

    WHY. The appearance pairing joins a sideline fragment to an endzone
    track by number or kit; the endzone track outlives the fragment, and
    for the rest of its life the paired id is drawn from the endzone alone:
    play 1 v16, id 68 = an endzone track of 510 frames paired to 18
    sideline frames, drawn 474 frames as a second copy of a player the
    sideline tracks under other ids (the dedupe does not catch it: the
    endzone's copy stands metres from the sideline's along x). One avatar
    per sideline track means the sideline span is the avatar's life; the
    endzone refines position inside it. Beyond the span the id is kept only
    where the sideline camera could not have seen it (outside its image).

    ``hold_m``: the frames on one side of the span are also dropped, all of them, when the endzone's
    point for the man at the frame adjacent to the span stands farther than this from the sideline's
    own first (or last) point for him: the endzone's ground point is poor along the field, so the
    man it draws before the sideline has him can stand metres from where the sideline then finds
    him, and the smoother turns the jump into a glide. Play 1 v53 (footage 2026-09-17): id 198
    drawn 30 frames on empty turf beside the tackle, 2.4 m from its man; id 40 a phantom defender
    pre-snap 2.7 m off; the held linemen 37 and 38 join within 0.3-0.6 m and stay. The jump is
    measured at the join, not per frame, because over a long lead-in a real man moves on his own.
    ``report`` (optional dict) gets ``{pid: (beyond_frames, jump_m_at_join, dropped)}``."""
    sub = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    # Keyed by the PLAYER, because `ground` is: ground_positions keys by global_player_id, and the two
    # ids are equal only until a track is relabelled onto another player (08s). Grouping by track_id
    # here would silently stop span-limiting exactly the relabelled ids -- the same confusion that made
    # ground_positions report zero endzone frames for every one of them (2026-09-12). Latent rather
    # than live so far, because 08s relabels endzone rows and this reads the sideline's.
    span = sub.groupby("global_player_id")["frame"].agg(["min", "max"])
    lo = {int(pid): int(r["min"]) - gap for pid, r in span.iterrows()}
    hi = {int(pid): int(r["max"]) + gap for pid, r in span.iterrows()}
    side_at = None
    if side_ground is not None:
        side_at = {int(f): {int(p): np.asarray(v, float) for p, v in d.items()} for f, d in side_ground.items()}

    def side_point(pid, edge, step):
        """The sideline's own point for ``pid`` at ``edge`` or within 5 frames into the span."""
        for g in range(edge, edge + 6 * step, step):
            q = side_at.get(g, {}).get(pid)
            if q is not None:
                return q
        return None

    def jump_at(pid, edge, step):
        """Metres between the endzone's point for ``pid`` just outside the span (within 3 frames of
        ``edge``, stepping away from the span) and the sideline's own point at the edge; NaN when
        either is missing."""
        sp = side_point(pid, edge, -step)
        if sp is None:
            return float("nan")
        for g in range(edge + step, edge + 4 * step, step):
            q = ground.get(g, {}).get(pid)
            if q is not None:
                return float(np.linalg.norm(np.asarray(q, float) - sp))
        return float("nan")

    jumps = {}
    if side_at is not None:
        for pid in lo:
            a, b = lo[pid] + gap, hi[pid] - gap                 # the span itself
            jumps[pid] = (jump_at(pid, a, -1), jump_at(pid, b, +1))   # (lead-in, tail)

    out = {}
    dropped = 0
    kept_gap = 0
    stats: dict = {}
    for f, d in ground.items():
        f = int(f)
        keep = {}
        for pid, xy in d.items():
            pid = int(pid)
            if pid in lo and not (lo[pid] + gap <= f <= hi[pid] - gap):
                # outside the sideline span: within ``gap`` frames of it the endzone alone draws the
                # man (the sideline's tracker is late or early by that much); beyond the gap only
                # where the sideline could not have seen the spot
                if not (lo[pid] <= f <= hi[pid]) and sideline is not None                         and f < len(sideline.conf) and _inside_sideline(xy, sideline, f, margin=margin):
                    # Beyond its sideline span this id is drawn from the endzone alone. That is a second
                    # copy of a man the sideline tracks under another id ONLY if the sideline has a body
                    # there; where it has none, the sideline simply lost him and this is the only body he
                    # has. Play 1: four Kansas City players are seen by the endzone through a sideline gap,
                    # and dropping them left nine of eleven on the field.
                    if side_at is None:
                        dropped += 1                      # without the sideline's bodies, the old rule stands
                        continue
                    near = min((float(np.linalg.norm(np.asarray(xy, float) - q))
                                for j, q in side_at.get(f, {}).items() if j != pid), default=np.inf)
                    if near <= same_body_m:
                        dropped += 1
                        continue
                    kept_gap += 1
                # the hold test: the jump at the join between the endzone's stretch and the sideline's
                # span (see ``hold_m`` above); the whole side of the span goes with its join
                jl, jt = jumps.get(pid, (float("nan"), float("nan")))
                jump = jl if f < lo[pid] + gap else jt
                st = stats.setdefault(pid, [0, jump, 0])
                st[0] += 1
                if np.isfinite(jump) and (not np.isfinite(st[1]) or jump > st[1]):
                    st[1] = jump
                if hold_m is not None and np.isfinite(jump) and jump > hold_m:
                    st[2] += 1
                    dropped += 1
                    continue
            keep[pid] = xy
        out[f] = keep
    if report is not None:
        for pid, (n, jump, nfar) in stats.items():
            report[pid] = (n, float(jump), nfar)
    return out, dropped


def endzone_only_ids(df, views, *, cam: str = "endzone", ground=None, sideline=None,
                     min_inside: float = MIN_INSIDE) -> set:
    """Ids to leave out: every box from ``cam``, no two-view frame, and (when
    ``ground`` frame -> {pid: xy} and the ``sideline`` CameraTrack are given)
    at least ``min_inside`` of their frames inside the sideline image."""
    two_view = set()
    for d in views.values():
        for pid, v in d.items():
            if len(set(v)) >= 2:
                two_view.add(int(pid))
    out = set()
    for pid, g in df.groupby("global_player_id"):
        pid = int(pid)
        if pid < 0 or pid in two_view:
            continue
        cams = set(g["cam"])
        if cams != {cam}:
            continue
        if ground is None or sideline is None:
            out.add(pid)
            continue
        n = inside = 0
        for f in g["frame"].to_numpy(int):
            xy = ground.get(int(f), {}).get(pid)
            if xy is None:
                continue
            n += 1
            inside += int(_inside_sideline(xy, sideline, int(f)))
        if n and inside / n >= min_inside:
            out.add(pid)
    return out
