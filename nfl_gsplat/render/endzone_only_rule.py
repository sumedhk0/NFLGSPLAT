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
SAME_BODY_GAP_M: float | None = 0.8    # ... and inside the 30-frame gap, this close (2026-09-18: 37's lead-in went, exactly-eleven +3 pre-snap, +10 live; 37's
                                       # lead-in at 340-369 stood 0.55 m from 166, the sideline's id for the same lineman)


PRESNAP_JOIN_MAX: int = 10       # the pre-snap per-frame test applies when the join is within this many frames of the snap
PRESNAP: str = "drop"            # a pre-snap beyond frame farther than HOLD_M from the join point: "drop" it, or "hold" the
                                 # man AT the join point (a set man has not moved; the join is where the sideline first has him)
HOLD_M: float | None = 0.8       # a beyond-span stretch whose join jumps farther than this from where the sideline first (or
                                 # last) has the man is the endzone's depth error, not the man: dropped


def beyond_sideline_span(ground, df, sideline, *, gap: int = 30, cam: str = "sideline",
                         margin: float = MARGIN_PX, side_ground=None, same_body_m: float = SAME_BODY_M,
                         hold_m: float | None = HOLD_M, report: dict | None = None, snap: int | None = None,
                         presnap: str = PRESNAP, presnap_join_max: int = PRESNAP_JOIN_MAX,
                         same_body_gap_m: float | None = SAME_BODY_GAP_M):
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
    measured at the join, not per frame, because over a long lead-in a real man moves on his own --
    except BEFORE THE SNAP (``snap``), when a set man does not move: there every frame is measured
    against the join point itself, which catches the ghost that slides onto its man (play 1 id 40:
    2.7 m off at 368, 0.67 m at the join 397, joined "cleanly" and drew a phantom defender for 30
    frames). Without ``snap`` the join test alone applies.
    ``report`` (optional dict) gets ``{pid: (beyond_frames, jump_m_at_join, dropped, jump_lead_in, jump_tail)}``."""
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
                # inside the gap: a sideline body of ANOTHER id this close is the same man (the sideline
                # already draws him under that id; the endzone's lead-in is his second copy)
                if same_body_gap_m is not None and side_at is not None and (lo[pid] <= f <= hi[pid]):
                    near_gap = min((float(np.linalg.norm(np.asarray(xy, float) - q))
                                    for j, q in side_at.get(f, {}).items() if j != pid), default=np.inf)
                    if near_gap <= same_body_gap_m:
                        st = stats.setdefault(pid, [0, float("nan"), 0])
                        st[0] += 1
                        st[2] += 1
                        dropped += 1
                        continue
                # the hold test: the jump at the join between the endzone's stretch and the sideline's
                # span (see ``hold_m`` above); the whole side of the span goes with its join
                jl, jt = jumps.get(pid, (float("nan"), float("nan")))
                lead_in = f < lo[pid] + gap
                jump = jl if lead_in else jt
                held = None
                if snap is not None and f < snap and side_at is not None:
                    # before the snap the man stands still: this frame's own distance from the join --
                    # when the join itself is at the snap (within PRESNAP_JOIN_MAX frames of it); a join
                    # deep in the play is where a man who has since run stands, and says nothing about
                    # where he stood set (play 1 id 74: join at snap+28, 1.14 m off, a real lineman
                    # dropped for the whole pre-snap; id 40's phantom joins at snap+5 and is caught)
                    edge = lo[pid] + gap if lead_in else hi[pid] - gap
                    if abs(edge - snap) <= presnap_join_max:
                        sp = side_point(pid, edge, -1 if lead_in else 1)
                        if sp is not None:
                            jump = float(np.linalg.norm(np.asarray(xy, float) - sp))
                            held = sp
                    else:
                        jump = float("nan")                      # nothing to hold to: the same-body test above stands
                st = stats.setdefault(pid, [0, jump, 0])
                st[0] += 1
                if np.isfinite(jump) and (not np.isfinite(st[1]) or jump > st[1]):
                    st[1] = jump
                if hold_m is not None and np.isfinite(jump) and jump > hold_m:
                    st[2] += 1
                    if held is not None and presnap == "hold":
                        keep[pid] = np.asarray(held, float)        # the set man, where the sideline first has him
                        continue
                    dropped += 1
                    continue
            keep[pid] = xy
        out[f] = keep
    if report is not None:
        for pid, (n, jump, nfar) in stats.items():
            jl, jt = jumps.get(pid, (float("nan"), float("nan")))
            report[pid] = (n, float(jump), nfar, float(jl), float(jt))
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


HOLE_HOLD_M: float | None = 0.8     # an endzone-filled hole frame farther than this from the sideline's own line
                                    # through the hole takes the line (play 1 id 37 at 569-576: five 0.4 m steps)
HOLE_MAX: int = 17                  # holes up to this long (2 * HOLE_REACH + 1) take the line between their ends;
HOLE_REACH: int = 8                 # longer ones extrapolate from the nearer end (the sideline's own velocity over
HOLE_VEL_FRAMES: int = 4            # HOLE_VEL_FRAMES) for the frames within HOLE_REACH of it, the ones the hole rule draws


def hold_holes(ground, side_ground, *, hold_m: float | None = HOLE_HOLD_M, max_hole: int = HOLE_MAX,
               reach: int = HOLE_REACH, vel_frames: int = HOLE_VEL_FRAMES):
    """``ground`` (frame -> {pid: xy}) with every endzone-filled HOLE frame -- inside a sideline span,
    the sideline has no point that frame, the merged ground has the endzone's -- moved onto the
    sideline's own straight line between its points either side of the hole when it stands farther
    than ``hold_m`` from that line. Returns ``(ground, moved [metres; NaN for a frame beyond reach of both ends, which is removed])``.

    WHY. The span rule (beyond_sideline_span) holds a span's EDGES; inside a span the hole rule draws
    an endzone-placed frame within HOLE_REACH of a sideline sighting, and the endzone's ground point
    is poor along the field: play 1 id 37 at 569-576 stepped 0.42-0.44 m a frame for five frames on
    an endzone-only hole in its sideline track, two metres out and back (07l on the live window). A
    man does not leave a straight line by a metre in eight frames; the sideline's interpolation
    across a short hole is the better guess, and the endzone still gives depth where both see him."""
    side_frames: dict = {}
    for f, d in side_ground.items():
        for pid in d:
            side_frames.setdefault(int(pid), []).append(int(f))
    for pid in side_frames:
        side_frames[pid].sort()
    out = {f: dict(d) for f, d in ground.items()}
    moved = []
    if hold_m is None:
        return out, moved
    for pid, fs in side_frames.items():
        arr = np.asarray(fs)
        for f in range(fs[0], fs[-1] + 1):
            if pid in side_ground.get(f, {}) or pid not in out.get(f, {}):
                continue
            i = int(np.searchsorted(arr, f))
            fa, fb = int(arr[i - 1]), int(arr[i])
            a = np.asarray(side_ground[fa][pid], float); b = np.asarray(side_ground[fb][pid], float)
            if fb - fa <= max_hole:
                line = a + (b - a) * (f - fa) / float(fb - fa)
            else:
                # a long hole: the hole rule draws only the frames within ``reach`` of a sideline
                # sighting; those follow that sighting at the sideline's own velocity there
                near_a = f - fa <= reach
                if not near_a and not (fb - f <= reach):
                    # beyond reach of both ends the endzone's point has nothing to hold to: not drawn
                    # (play 1 id 37 at 560-565 glided two metres into its held frames, v56)
                    del out[f][pid]
                    moved.append(float("nan"))
                    continue
                edge, step = (fa, -1) if near_a else (fb, +1)
                back = edge + step * vel_frames
                prev = None
                for g in range(edge + step, back + step, step):
                    if pid in side_ground.get(g, {}):
                        prev = (g, np.asarray(side_ground[g][pid], float))
                if prev is None:
                    v = np.zeros(2)
                else:
                    v = (np.asarray(side_ground[edge][pid], float) - prev[1]) / float(edge - prev[0])
                line = np.asarray(side_ground[edge][pid], float) + v * (f - edge)
            d = float(np.linalg.norm(np.asarray(out[f][pid], float) - line))
            if d > hold_m:
                out[f][pid] = line
                moved.append(d)
    return out, moved


FORMATION_STILL_M: float | None = None   # a man whose pre-snap sideline points stay within this of their median is set:
                                         # drawn there from the clip start to the snap (off until measured, 2026-09-18)
FORMATION_MIN_FRAMES: int = 5
FORMATION_MARGIN: int = 10               # frames before the snap the hold stops (the line moves before the ball does)


FORMATION_EMPTY_M: float = 0.8           # a held spot must be this far from every sideline body that frame
# The general hold was measured and rejected (fragments a metre from the men they double get held
# too, 2026-09-18). The one role it is right for is the quarterback under centre: the sideline sees
# him for a handful of frames behind the line (play 1: id 33, five frames), his spot is fixed, and
# no fragment doubles him there -- so for these roles the hold runs with its own bar, 0.4 m (the
# centre stands 0.7 m in front of him and must not count as the spot being taken).
FORMATION_ROLES: tuple = ("QB",)
FORMATION_ROLE_STILL_M: float | None = None   # off: play 1 id 33 (role QB) stands 5 m beside the centre, not behind him (2026-09-18)
FORMATION_ROLE_EMPTY_M: float = 0.4
FORMATION_ROLE_MIN_FRAMES: int = 3


def formation_hold(ground, side_ground, *, start: int, snap: int, still_m: float | None = FORMATION_STILL_M,
                   min_frames: int = FORMATION_MIN_FRAMES, margin: int = FORMATION_MARGIN,
                   empty_m: float = FORMATION_EMPTY_M, only_ids=None):
    """``ground`` with every set man drawn at his median pre-snap sideline point on the pre-snap frames
    the sideline missed him on. Set: at least ``min_frames`` sideline points in [start, snap - margin]
    that all lie within ``still_m`` of their median (a man in motion, a shifting defender or a
    switched track fails this). Returns ``(ground, {pid: frames_added})``.

    WHY. Before the snap nobody moves for seconds, yet play 1's line was drawn by fragments that
    came and went (KC 8-10 of 11 on 213-300: 82 drawn 300-307, 31 at 220, 166 from 291), each man
    flickering in and out of a formation that stood still. Where the sideline has a man set, it has
    him for the whole window; the tracker's gaps are its own, not the man's."""
    out = {f: dict(d) for f, d in ground.items()}
    added: dict = {}
    if still_m is None:
        return out, added
    lo, hi = int(start), int(snap) - int(margin)
    pts: dict = {}
    for f, d in side_ground.items():
        if lo <= int(f) <= hi:
            for pid, xy in d.items():
                pts.setdefault(int(pid), []).append(np.asarray(xy, float))
    for pid, arr in pts.items():
        if only_ids is not None and pid not in only_ids:
            continue
        if len(arr) < min_frames:
            continue
        a = np.stack(arr)
        med = np.median(a, axis=0)
        if np.max(np.linalg.norm(a - med, axis=1)) > still_m:
            continue
        for f in range(lo, hi + 1):
            if pid in out.setdefault(f, {}):
                continue
            # only an EMPTY spot is filled: a fragment that sat on another man (a rider) held for
            # the whole window became a duplicate of him (play 1 id 32: BAL 11.0 -> 12.05 pre-snap)
            near = min((float(np.linalg.norm(np.asarray(q, float) - med)) for j, q in side_ground.get(f, {}).items() if int(j) != pid),
                       default=np.inf)
            if near < empty_m:
                continue
            out[f][pid] = med.copy()
            added[pid] = added.get(pid, 0) + 1
    return out, added


QB_HOLD: bool = True             # the quarterback under centre, held at the spot he steps back from
QB_BEHIND_M: tuple = (0.5, 2.5)  # his first sideline point lies this far behind the centre (the offence's side) ...
QB_ACROSS_M: float = 1.0         # ... and this close to the centre's line
QB_WINDOW: tuple = (-25, 15)     # ... on a track that starts this close to the snap


def qb_hold(ground, side_ground, *, start: int, snap: int, centre_xy, sign: float, team_ids, first_frame: dict,
            behind_m: tuple = QB_BEHIND_M, across_m: float = QB_ACROSS_M, window: tuple = QB_WINDOW):
    """``(ground, pid, n_added)``: the quarterback under centre, drawn from ``start`` to the frame
    before his sideline track begins, at that track's first point.

    WHY. Under centre the quarterback stands inside the centre's detection box in both cameras
    (play 1: no keypoint at his helmet, the endzone's second box there is the centre's own), so no
    id carries him until he steps back at the snap -- the sideline then picks him up 0.8 m behind
    the centre (play 1 id 80 at 377, snap 393). A set quarterback has not moved: the spot he steps
    back from is the spot he stood on. ``team_ids``: the offence's ids; ``first_frame``: pid -> first
    sideline frame; the candidate is the offence id whose track starts within ``window`` of the snap,
    ``behind_m`` behind the centre along the field (toward the offence, ``sign``) and within
    ``across_m`` of the centre's line, with no sideline point before the window."""
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    best = None
    for pid, f0 in first_frame.items():
        pid = int(pid); f0 = int(f0)
        if pid not in team_ids or not (snap + window[0] <= f0 <= snap + window[1]):
            continue
        pt = side_ground.get(f0, {}).get(pid)
        if pt is None:
            continue
        behind = (float(pt[0]) - cx) * float(sign)
        across = abs(float(pt[1]) - cy)
        if behind_m[0] <= behind <= behind_m[1] and across <= across_m:
            if best is None or across < best[0]:
                best = (across, pid, f0, np.asarray(pt, float))
    if best is None:
        return ground, None, 0
    _a, pid, f0, pt = best
    out = {f: dict(d) for f, d in ground.items()}
    n = 0
    for f in range(int(start), f0):
        if pid not in out.setdefault(f, {}):
            out[f][pid] = pt.copy()
            n += 1
    return out, pid, n
