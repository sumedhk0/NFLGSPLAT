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


def beyond_sideline_span(ground, df, sideline, *, gap: int = 30, cam: str = "sideline",
                         margin: float = MARGIN_PX):
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
    where the sideline camera could not have seen it (outside its image)."""
    sub = df[(df["cam"] == cam) & (df["track_id"] >= 0)]
    span = sub.groupby("track_id")["frame"].agg(["min", "max"])
    lo = {int(pid): int(r["min"]) - gap for pid, r in span.iterrows()}
    hi = {int(pid): int(r["max"]) + gap for pid, r in span.iterrows()}
    out = {}
    dropped = 0
    for f, d in ground.items():
        f = int(f)
        keep = {}
        for pid, xy in d.items():
            pid = int(pid)
            if pid in lo and not (lo[pid] <= f <= hi[pid]) and sideline is not None \
                    and f < len(sideline.conf) and _inside_sideline(xy, sideline, f, margin=margin):
                dropped += 1
                continue
            keep[pid] = xy
        out[f] = keep
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
