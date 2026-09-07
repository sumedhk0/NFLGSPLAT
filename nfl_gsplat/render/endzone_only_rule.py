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
a two-view id is left out of the timeline (the same shape as the edge
rule). A player the sideline never sees at all is lost to the render; on
play 1 that is nobody the sideline's frame does not show.
"""
from __future__ import annotations


def endzone_only_ids(df, views, *, cam: str = "endzone") -> set:
    """Ids to leave out: every box from ``cam`` and no two-view frame.
    ``views`` is ground_positions' frame -> {pid: cameras}."""
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
        if cams == {cam}:
            out.add(pid)
    return out
