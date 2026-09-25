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
SAME_BODY_MAX_SPAN: int | None = None      # OFF (measured worse, 2026-09-22: 0.498 -> 0.671). With a value: the same-body test only for an id whose sideline span is at most this many frames: the copy
                                          # case is a short sideline fragment paired to a long endzone track (v16: 18 frames);
                                          # a long-lived sideline id's endzone tail is his own man (the centre behind the QB, 557+)
SAME_BODY_SAME_TEAM: bool = False         # ... and only a sideline man of the SAME team can be the copy (a Raven 1 m from a KC
                                          # tackle is not the tackle's second copy: id 1's 112 rows beside KC 12 and 76, 2026-09-21)
SAME_BODY_APART_IOU: float | None = 0.3   # ... unless the endzone boxes both men on the frame, overlapping below this: two men
                                          # (play 1 ids 4 and 40 beside KC 65 from 524, 1.0-1.5 m apart, one sideline box)
HOLD_M: float | None = 0.8       # a beyond-span stretch whose join jumps farther than this from where the sideline first (or
                                 # last) has the man is the endzone's depth error, not the man: dropped


def beyond_sideline_span(ground, df, sideline, *, gap: int = 30, cam: str = "sideline",
                         margin: float = MARGIN_PX, side_ground=None, same_body_m: float = SAME_BODY_M,
                         hold_m: float | None = HOLD_M, report: dict | None = None, snap: int | None = None,
                         presnap: str = PRESNAP, presnap_join_max: int = PRESNAP_JOIN_MAX,
                         same_body_gap_m: float | None = SAME_BODY_GAP_M, apart_iou: float | None = SAME_BODY_APART_IOU,
                         teams: dict | None = None, same_body_max_span: int | None = SAME_BODY_MAX_SPAN):
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
    # the endzone's boxes per frame, for the same-body test: two boxes apart on the frame are two men
    ez_boxes: dict = {}
    if apart_iou is not None and {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(df.columns) and "endzone" in set(df["cam"].astype(str)):
        for r in df[df["cam"] == "endzone"].itertuples():
            ez_boxes.setdefault(int(r.frame), {})[int(r.global_player_id)] = (float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2))

    def same_team(pid, j):
        """``j`` can be ``pid``'s second copy: the same team, or either team unknown (``teams`` None = the old rule)."""
        if teams is None or not SAME_BODY_SAME_TEAM:
            return True
        a, b = teams.get(int(pid)), teams.get(int(j))
        return a is None or b is None or a == b

    def boxes_apart(f, pid, j):
        """Both ``pid`` and ``j`` have an endzone box on ``f`` and the two overlap below ``apart_iou``."""
        a, b = ez_boxes.get(f, {}).get(pid), ez_boxes.get(f, {}).get(j)
        if a is None or b is None:
            return False
        w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = w * h
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return union > 0 and inter / union < apart_iou
    # Keyed by the PLAYER, because `ground` is: ground_positions keys by global_player_id, and the two
    # ids are equal only until a track is relabelled onto another player (08s). Grouping by track_id
    # here would silently stop span-limiting exactly the relabelled ids -- the same confusion that made
    # ground_positions report zero endzone frames for every one of them (2026-09-12). Latent rather
    # than live so far, because 08s relabels endzone rows and this reads the sideline's.
    span = sub.groupby("global_player_id")["frame"].agg(["min", "max"])
    lo = {int(pid): int(r["min"]) - gap for pid, r in span.iterrows()}
    hi = {int(pid): int(r["max"]) + gap for pid, r in span.iterrows()}
    span_len = {int(pid): int(r["max"]) - int(r["min"]) + 1 for pid, r in span.iterrows()}

    def same_body_applies(pid):
        """The same-body test is for a short sideline span (a fragment pairing); a long one is trusted."""
        return same_body_max_span is None or span_len.get(int(pid), 0) <= int(same_body_max_span)
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
                    near, near_j = min(((float(np.linalg.norm(np.asarray(xy, float) - q)), j)
                                        for j, q in side_at.get(f, {}).items() if j != pid and same_team(pid, j)), default=(np.inf, None))
                    if near <= same_body_m and same_body_applies(pid) and not boxes_apart(f, pid, near_j):
                        dropped += 1
                        continue
                    kept_gap += 1
                # inside the gap: a sideline body of ANOTHER id this close is the same man (the sideline
                # already draws him under that id; the endzone's lead-in is his second copy)
                if same_body_gap_m is not None and side_at is not None and (lo[pid] <= f <= hi[pid]):
                    near_gap = min((float(np.linalg.norm(np.asarray(xy, float) - q))
                                    for j, q in side_at.get(f, {}).items() if j != pid and same_team(pid, j)), default=np.inf)
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
HOLE_CHAIN_STEP_M: float | None = 0.6   # inside a LONG hole the endzone's own points are kept on a continuous chain from the
                                        # hole's near end: first point within HOLE_HOLD_M of the sideline's there, then this far
                                        # per frame of gap (the endzone's depth jitter breaks a 0.3 m chain); None = off
HOLE_CHAIN_GAP: int = 5                 # ... across gaps of at most this many frames
# A point off the chain is SKIPPED, not the chain's end (the chain still ends after HOLE_CHAIN_GAP frames without a
# point on it), and a step is bounded ACROSS the endzone camera's line of sight by HOLE_CHAIN_ACROSS_M plus
# HOLE_CHAIN_SPEED_M per frame of gap (the endzone measures across well; a man covers ~0.17 m a frame at 59.94 fps)
# as well as ALONG it by step_m per frame of gap (its depth jitter). Play 1 (2026-09-25): Madubuike (id 4) crawled
# on the turf through a 443-528 sideline hole; one box on the tackle beside him at 491 (1.3 m across, inside the
# isotropic 1.8 m three-frame bound) was taken, the true point at 493 then failed and ended the chain, and the rule
# deleted 493-516 while the endzone boxed him on: the timeline held him still a metre off. The across bound also
# refuses the box merged with the guard over him at 518-522 (1.2 m across). Read at call time by the loader.
# ADOPTED 2026-09-25 (v113): on play 1 only two long holes carry endzone points and only their ids move. Endzone film
# (05q strips): Madubuike drawn over #92 on the turf instead of on #74's block 490-518; the centre on #52 512-528
# (the old bridge drifted half a metre onto Thuney), 500-508 a bridge 0.25 m right of him (no box on him there).
# Live 07l unchanged (steps 2 / hops 0 / census 0.16); depth ruler across p99 1.18 -> 0.86 m, over 1.5 m 2.8 -> 2.1 %;
# frames the hole rule deleted 42 -> 11. Flags off reproduce v112 exactly (13,424 body-frames).
HOLE_CHAIN_SKIP: bool = True
HOLE_CHAIN_ACROSS_M: float | None = 0.2   # None: the shipped isotropic bound (step_m per frame of gap)
HOLE_CHAIN_SPEED_M: float = 0.17


def camera_ground_xy(track, step: int = 20):
    """A camera's centre on the ground plane (the median over its solved frames every ``step``), or None."""
    if track is None:
        return None
    cs = []
    for f in range(0, len(track.conf), int(step)):
        if track.conf[f] <= 0:
            continue
        _, pose = track.at(f)
        R = np.asarray(pose.R, float)
        t = np.asarray(pose.t, float).reshape(3)
        cs.append((-R.T @ t)[:2])
    return np.median(np.asarray(cs), axis=0) if cs else None


def hole_chain(ground, side_ground, pid, fa, fb, *, hold_m: float, step_m: float, gap: int, skip: bool = False,
               across_m: float | None = None, speed_m: float = HOLE_CHAIN_SPEED_M, cam_xy=None) -> set:
    """Frames strictly inside the hole (fa, fb) of ``pid`` whose endzone points form a continuous chain from either end:
    the first point (within ``gap`` frames of the end) within ``hold_m`` of the sideline's point at that end, then each
    next point within ``gap`` frames and ``step_m`` per frame of gap of the previous. The chain is the man's own track.
    With ``across_m`` and ``cam_xy`` (the endzone camera's ground point) a step is bounded along the camera's line of
    sight by ``step_m`` and across it by ``across_m + speed_m`` per frame of gap; with ``skip`` a point off the chain is
    passed over instead of ending it."""
    keep: set = set()
    for edge, step in ((int(fa), +1), (int(fb), -1)):
        sp = side_ground.get(edge, {}).get(pid)
        if sp is None:
            continue
        prev_f, prev_x, first = edge, np.asarray(sp, float), True
        f = edge + step
        while fa < f < fb:
            q = ground.get(f, {}).get(pid)
            if q is None:
                f += step
                if abs(f - prev_f) > gap + 1:
                    break
                continue
            q = np.asarray(q, float)
            n = abs(f - prev_f)
            if first:
                ok = float(np.linalg.norm(q - prev_x)) <= hold_m
            elif across_m is not None and cam_xy is not None:
                ray = prev_x - np.asarray(cam_xy, float)
                ray = ray / max(float(np.linalg.norm(ray)), 1e-9)
                d = q - prev_x
                ok = (abs(float(d @ ray)) <= step_m * n
                      and abs(float(d[0] * ray[1] - d[1] * ray[0])) <= across_m + speed_m * n)
            else:
                ok = float(np.linalg.norm(q - prev_x)) <= step_m * n
            if not ok:
                if not skip:
                    break
                f += step
                if abs(f - prev_f) > gap + 1:
                    break
                continue
            keep.add(f)
            prev_f, prev_x, first = f, q, False
            f += step
    return keep


def hold_holes(ground, side_ground, *, hold_m: float | None = HOLE_HOLD_M, max_hole: int = HOLE_MAX,
               reach: int = HOLE_REACH, vel_frames: int = HOLE_VEL_FRAMES, chain_step_m: float | None = HOLE_CHAIN_STEP_M,
               chain_gap: int = HOLE_CHAIN_GAP, chain_skip: bool = False, chain_across_m: float | None = None,
               chain_speed_m: float = HOLE_CHAIN_SPEED_M, cam_xy=None):
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
    chained: dict = {}                                   # pid -> frames on an endzone chain inside a long hole: kept as they are
    if chain_step_m is not None:
        for pid, fs in side_frames.items():
            for fa, fb in zip(fs, fs[1:]):
                if fb - fa - 1 > max_hole:
                    chained.setdefault(pid, set()).update(
                        hole_chain(ground, side_ground, pid, fa, fb, hold_m=hold_m, step_m=chain_step_m, gap=chain_gap,
                                   skip=chain_skip, across_m=chain_across_m, speed_m=chain_speed_m, cam_xy=cam_xy))
    for pid, fs in side_frames.items():
        arr = np.asarray(fs)
        for f in range(fs[0], fs[-1] + 1):
            if pid in side_ground.get(f, {}) or pid not in out.get(f, {}):
                continue
            if f in chained.get(pid, ()):
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
QB_SAME_M: float = 0.5           # a teammate the sideline draws this close to the held spot before the snap IS the
                                 # quarterback under another id (play 1: 204, a 159-px box on the spot on 90 of 180
                                 # pre-snap frames, flickering): he goes, the held body stands for him on every frame.
                                 # Measured on play 1's pre-snap 213-383 with the line vouch on (2026-09-18), mean
                                 # |KC - 11| a frame: off 0.602, 0.8 -> 0.538, 0.5 -> 0.520 (exact-eleven frames 80 ->
                                 # 91, frames at twelve or more 61 -> 34, at ten or fewer 30 -> 46: 204 stood in for
                                 # nobody on some frames but for the held man on most)
QB_SAME_BEHIND_M: float = 0.3    # ... provided he stands at least this far behind the centre: the centre himself and
                                 # the linemen beside him are never the quarterback (a first cut without this took the
                                 # centre out on 18 frames and a guard on 10)


def qb_hold(ground, side_ground, *, start: int, snap: int, centre_xy, sign: float, team_ids, first_frame: dict,
            behind_m: tuple = QB_BEHIND_M, across_m: float = QB_ACROSS_M, window: tuple = QB_WINDOW,
            same_m: float | None = QB_SAME_M, same_behind_m: float = QB_SAME_BEHIND_M, removed: dict | None = None):
    """``(ground, pid, n_added)``: the quarterback under centre, drawn from ``start`` to the frame
    before his sideline track begins, at that track's first point. A teammate within ``same_m`` of
    that point on a held frame is the same man under another id and is taken out (``removed``, an
    optional dict, gets ``{pid: frames}``), so one body stands there with one pose.

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
        d = out.setdefault(f, {})
        if same_m is not None:
            for other in [j for j, q in d.items() if int(j) != pid and int(j) in team_ids
                          and float(np.linalg.norm(np.asarray(q, float) - pt)) <= same_m
                          and (float(q[0]) - cx) * float(sign) >= same_behind_m]:
                del d[other]
                if removed is not None:
                    removed[int(other)] = removed.get(int(other), 0) + 1
        if pid not in d:
            d[pid] = pt.copy()
            n += 1
    return out, pid, n


# ---- the hidden linemen: the endzone's across axis separates the line, the sideline's cannot -------------
# Before the snap the offensive line stands across the field, stacked along the sideline camera's depth
# axis: from the sideline the guards hide behind the centre and their tracks start late (play 1: 38 at
# 345, 37 at 370), while the endzone camera, looking along the line, sees every one of them side by side
# for the whole pre-snap (38 on 169 frames, 37 on 166). Those endzone-only bodies were then deleted by the
# dedupe as copies of the centre -- along the sideline's depth axis they sit inside its box -- and Kansas
# City stood 8-10 of 11 for three seconds. A ghost from the endzone (a second copy of a man the sideline
# draws under another id) shares its man's ACROSS position, which the endzone measures well; a hidden
# lineman stands a body's width across from every sideline-backed teammate. So on the pre-snap frames an
# endzone-only body on the line (within LINE_VOUCH_LINE_M along the field of the offence's side of the
# LOS) is vouched for -- exempt from the dedupe -- when no sideline-backed body of its team stands within
# LINE_VOUCH_ACROSS_M of it ACROSS the field.
# Measured on play 1's pre-snap window 213-383 (2026-09-18): KC 9.75 -> 11.18 bodies a frame, frames at
# exactly eleven a side 28 -> 80, frames with KC at ten or fewer 129 -> 30; 38 vouched on 115 frames,
# 74 on 157 (both real, checked on the endzone footage: a red skeleton on a red lineman), 86 on 87.
# The 61 frames now at twelve carry the sideline's own twins (82/166 on one man, 204 on the quarterback's
# spot), which stood there before too under a nine-man count. 0.5 and 0.7 read the same.
LINE_VOUCH_ACROSS_M: float | None = 0.7
LINE_VOUCH_LINE_M: float = 2.0
# A PURE endzone-only id (no two-view frame, left out whole as a ghost) can be drawn on its vouched frames only,
# when it has at least LINE_VOUCH_REVIVE_MIN of them (play 1's left guard 86 has 87; the Baltimore ghost 92 has
# 22). MEASURED AND NOT ADOPTED (2026-09-18, pre-snap 213-383): the guard is drawn on every frame (KC at ten or
# fewer 40 -> 4) but the fragments that stood in his slot (166, 204, 82, 31) now count as extras: exact-eleven
# frames 112 -> 103, mean |KC-11| 0.345 -> 0.433. It pays once those fragments are folded (82 is his own
# sideline partial box 0.55 m away). None = off; 30 to try it.
LINE_VOUCH_REVIVE_MIN: int | None = None    # off (see above); 30 to try it, with LINE_VOUCH_FOLD_M folding his sideline fragment
LINE_VOUCH_FOLD_M: float = 0.8              # a same-team sideline-only body this close to a revived man is his fragment
LINE_VOUCH_OFFSET_M: float = 1.0          # the line stands about this far on its side of the LOS


def line_vouch(ground, views, side_ground, *, start: int, snap: int, teams: dict, los_x: float, sign: float,
               across_m: float | None = LINE_VOUCH_ACROSS_M, line_m: float = LINE_VOUCH_LINE_M,
               offset_m: float = LINE_VOUCH_OFFSET_M, margin: int = FORMATION_MARGIN, endzone: str = "endzone"):
    """``({frame: {pid}}, {pid: n_frames})`` of the pre-snap endzone-only bodies on the line that no
    sideline-backed teammate stands within ``across_m`` of, across the field. See above."""
    keep: dict = {}
    counts: dict = {}
    if across_m is None or views is None or side_ground is None:
        return keep, counts
    line_x = float(los_x) + float(sign) * float(offset_m)
    for f in range(int(start), int(snap) - int(margin) + 1):
        vf = views.get(f, {})
        side = side_ground.get(f, {})
        for pid, xy in ground.get(f, {}).items():
            pid = int(pid)
            v = tuple(vf.get(pid, ()))
            if v != (endzone,):
                continue                                    # the sideline sees him: the dedupe's business
            xy = np.asarray(xy, float)
            if abs(xy[0] - line_x) > line_m:
                continue                                    # not on the line
            tm = teams.get(pid)
            near = min((abs(float(np.asarray(q, float)[1]) - xy[1]) for j, q in side.items()
                        if int(j) != pid and teams.get(int(j)) == tm), default=np.inf)
            if near < across_m:
                continue                                    # a teammate the sideline draws stands there
            keep.setdefault(f, set()).add(pid)
            counts[pid] = counts.get(pid, 0) + 1
    return keep, counts


# ---- the pocket during the play: the rusher the sideline cannot see ------------------------------------
# After the snap the sideline camera loses a pass rusher behind the lineman he is engaged with (play 1 at 492
# and 546: eleven Ravens on the film, ten sideline boxes on ten distinct men, the eleventh unboxed at the
# pocket's edge; 2026-09-19), so Baltimore reads 10.8 a frame on the play window with no identity error to
# fix. The endzone camera, looking along the line, sees the rushers side by side. line_vouch's test carries
# over: an endzone-only body in the pocket region (from POCKET_VOUCH_BEHIND_M in front of the LOS to
# POCKET_VOUCH_DEPTH_M behind it, on the offence's side) is vouched on a play frame when no sideline-backed
# body of ITS team stands within POCKET_VOUCH_ACROSS_M of it across the field -- a ghost of a drawn rusher
# shares his across position (the endzone measures it well), a hidden rusher does not. A pure endzone-only id
# is then revived on its vouched frames when it has POCKET_VOUCH_REVIVE_MIN of them (the loader). Off (None)
# until measured on the census and the film.
POCKET_VOUCH_ACROSS_M: float | None = None
POCKET_VOUCH_DEPTH_M: float = 8.0
POCKET_VOUCH_BEHIND_M: float = 2.0
POCKET_VOUCH_REVIVE_MIN: int = 10


def pocket_vouch(ground, views, side_ground, *, snap: int, end: int, teams: dict, los_x: float, sign: float,
                 across_m: float | None = POCKET_VOUCH_ACROSS_M, depth_m: float = POCKET_VOUCH_DEPTH_M,
                 behind_m: float = POCKET_VOUCH_BEHIND_M, endzone: str = "endzone"):
    """``({frame: {pid}}, {pid: n_frames})`` of the endzone-only bodies in the pocket on the play frames
    ``snap..end`` that no sideline-backed teammate stands within ``across_m`` of across the field."""
    keep: dict = {}
    counts: dict = {}
    if across_m is None or views is None or side_ground is None:
        return keep, counts
    lo = float(los_x) * float(sign) - float(behind_m)          # along the field, on the offence's side, in sign units
    hi = float(los_x) * float(sign) + float(depth_m)
    for f in range(int(snap), int(end) + 1):
        vf = views.get(f, {})
        side = side_ground.get(f, {})
        for pid, xy in ground.get(f, {}).items():
            pid = int(pid)
            if tuple(vf.get(pid, ())) != (endzone,):
                continue
            xy = np.asarray(xy, float)
            along = float(xy[0]) * float(sign)
            if not (lo <= along <= hi):
                continue                                    # not in the pocket
            tm = teams.get(pid)
            near = min((abs(float(np.asarray(q, float)[1]) - xy[1]) for j, q in side.items()
                        if int(j) != pid and teams.get(int(j)) == tm), default=np.inf)
            if near < across_m:
                continue                                    # a teammate the sideline draws stands there: a ghost
            keep.setdefault(f, set()).add(pid)
            counts[pid] = counts.get(pid, 0) + 1
    return keep, counts


# ---- a set man's holes before the snap ---------------------------------------------------------------
# Inside its own pre-snap span an id has holes the hole rule cannot touch: hold_holes moves endzone-filled
# frames, and a sideline-only id has none to move (play 1's left guard, id 82: sideline points on 51 of
# the 91 frames 217-307, a 69-px box on a man hidden behind the centre, no endzone id); a vouched
# endzone-only guard (38) has the endzone's own gaps (drawn in seven runs, holes of 3-24 frames: he
# flickered in v65). Before the snap a set man does not move, so a hole between two of his points is
# filled with the straight line between them, whatever its length -- but only where that line point is
# EMPTY: no same-team body within PRESNAP_FILL_ACROSS_M across and PRESNAP_FILL_ALONG_M along the field
# (a twin's hole is its man's frame; filling it would draw the ghost -- the first cut without the empty
# test filled 19, 34, 195 and 204 and read |KC - 11| 0.520 -> 0.538). Nothing is added beyond an id's
# first or last pre-snap point (the span rule's business). ``frames_of`` restricts an id to holes between
# the frames given (the vouched ones, for a vouched id).
# Measured on play 1's pre-snap 213-383 (2026-09-18, line vouch and quarterback rule on), mean |KC - 11| a
# frame: off 0.520; the vouched ids' holes only (38: 28 frames) 0.462, exact-eleven frames 91 -> 96; plus
# every empty spot (36 more frames: 172, 98, 82 ...) 0.427, exact 102, frames at ten or fewer 46 -> 28,
# at twelve or more 34 -> 41. The loader's presnap_fill takes False / "vouched" / True.
# DEAD CODE 2026-09-20..25 (5779113 nested the loader's block under the switched-off short-team vouch); moved back
# 2026-09-25 with the default OFF -- the state v110-v112 shipped -- until re-measured on today's tables.
PRESNAP_FILL: bool = False
PRESNAP_FILL_ACROSS_M: float = 0.7
PRESNAP_FILL_ALONG_M: float = 2.0


def fill_presnap_holes(ground, *, start: int, snap: int, teams: dict, margin: int = FORMATION_MARGIN,
                       only_ids=None, frames_of: dict | None = None, across_m: float = PRESNAP_FILL_ACROSS_M,
                       along_m: float = PRESNAP_FILL_ALONG_M):
    """``(ground, {pid: frames_added})``: see above."""
    out = {f: dict(d) for f, d in ground.items()}
    added: dict = {}
    lo, hi = int(start), int(snap) - int(margin)
    have: dict = {}
    for f in range(lo, hi + 1):
        for pid in ground.get(f, {}):
            pid = int(pid)
            if only_ids is not None and pid not in only_ids:
                continue
            if frames_of is not None and pid in frames_of and f not in frames_of[pid]:
                continue
            have.setdefault(pid, []).append(f)
    for pid, fs in have.items():
        fs.sort()
        if len(fs) < 2:
            continue
        arr = np.asarray(fs)
        tm = teams.get(pid)
        for f in range(fs[0], fs[-1] + 1):
            d = out.setdefault(f, {})
            if pid in d:
                continue
            i = int(np.searchsorted(arr, f))
            fa, fb = int(arr[i - 1]), int(arr[i])
            a = np.asarray(ground[fa][pid], float); b = np.asarray(ground[fb][pid], float)
            pt = a + (b - a) * (f - fa) / float(fb - fa)
            taken = any(int(j) != pid and teams.get(int(j)) == tm
                        and abs(float(q[1]) - pt[1]) < across_m and abs(float(q[0]) - pt[0]) < along_m
                        for j, q in d.items())
            if taken:
                continue
            d[pid] = pt
            added[pid] = added.get(pid, 0) + 1
    return out, added


# The eleventh man the sideline never boxed. On play 1 (2026-09-20) Baltimore read ten on 119 of the 213 play frames while
# two Ravens had endzone boxes and no sideline ones at all: 146 in coverage downfield (out of the sideline's frame from
# 468) and 112 (jersey 90, a rusher) in the pile from 524. The endzone-only rule drops them whole because a body the
# sideline could have seen is, nine times in ten, the sideline's own player unpaired. The count is the tenth time: a
# team drawn short on a frame has a man missing, and an endzone-only body of that team standing clear of every drawn
# teammate is him. The pocket vouch (rejected 09-19: KC copies revived) had no such gate: KC read eleven or twelve,
# so nothing of KC's is vouched here. Off (None) until measured on the census, the rulers and the endzone blend.
SHORT_TEAM_VOUCH_CLEAR_M: float | None = None   # the endzone-only body must stand this far from every drawn teammate
SHORT_TEAM_VOUCH_REVIVE_MIN: int = 10           # frames vouched before the id is drawn at all
SHORT_TEAM_FULL: int = 11


def short_team_vouch(ground, views, *, lo: int, hi: int, teams: dict, clear_m: float | None = SHORT_TEAM_VOUCH_CLEAR_M,
                     full: int = SHORT_TEAM_FULL, endzone: str = "endzone", exclude=None):
    """``({frame: {pid}}, {pid: n_frames})``: on frames ``lo..hi`` where a team's bodies that are NOT endzone-only
    number fewer than ``full``, its endzone-only bodies (``views[f][pid] == (endzone,)``) standing at least
    ``clear_m`` from every non-endzone-only teammate. ``exclude``: ids never vouched (officials, staff)."""
    keep: dict = {}
    counts: dict = {}
    if clear_m is None:
        return keep, counts
    exclude = set(int(p) for p in (exclude or ()))
    for f in range(int(lo), int(hi) + 1):
        g = ground.get(f)
        if not g:
            continue
        vf = views.get(f, {})
        drawn: dict = {}
        ez: dict = {}
        for pid, xy in g.items():
            pid = int(pid)
            tm = teams.get(pid)
            if tm is None:
                continue
            if tuple(vf.get(pid, ())) == (endzone,):
                if pid not in exclude:
                    ez.setdefault(tm, []).append((pid, np.asarray(xy, float)[:2]))
            else:
                drawn.setdefault(tm, []).append(np.asarray(xy, float)[:2])
        for tm, cands in ez.items():
            if len(drawn.get(tm, [])) >= full:
                continue
            for pid, xy in cands:
                near = min((float(np.hypot(*(q - xy))) for q in drawn.get(tm, [])), default=np.inf)
                if near >= clear_m:
                    keep.setdefault(f, set()).add(pid)
                    counts[pid] = counts.get(pid, 0) + 1
    return keep, counts
