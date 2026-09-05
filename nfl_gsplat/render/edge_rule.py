"""Bodies whose boxes are cut by the frame edge: not on the field.

WHY. A box clipped by the top or bottom of the frame has no foot or no
head; its ground point is the line the edge maps to, so every such body
lands on that one line. Play 2 drew a column of red bodies at x = -24 m,
8-9 m behind the offence: officials seen at the top of the endzone view,
endzone-only, default-posed. Play 1's equivalents are photographers beyond
the end line, an official at the goal line, and three people on the
sideline.

MEASURED 2026-09-05 (the /btw fork's rule_b.py): play 2 drops 19
endzone-only ids (10 % of timeline states, all behind the offence); play 1
drops 23 (3 sideline, 20 late in the end zone); no sure identity (roster
name with a team+number unique in the play) among them; the one real
player in play 1's late group is two-view and stays. No time guard needed.
"""
from __future__ import annotations

EDGE_PX: float = 8.0


def edge_clipped_ids(df, tracks, views) -> set:
    """Ids to leave out of the timeline: seen by ONE camera only, never a
    two-view frame, and every one of their boxes touches the top or bottom
    edge of that camera's frame (within ``EDGE_PX``). ``views`` is
    ground_positions' frame -> {pid: cameras}; ``tracks[cam].height`` the
    frame height."""
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
        if len(cams) != 1:
            continue
        cam = next(iter(cams))
        h = float(tracks[cam].height)
        top = g["bbox_y1"].to_numpy() <= EDGE_PX
        bottom = g["bbox_y2"].to_numpy() >= h - EDGE_PX
        if len(g) and bool((top | bottom).all()):
            out.add(pid)
    return out
