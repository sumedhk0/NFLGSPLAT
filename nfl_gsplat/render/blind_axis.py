"""The endzone's blind axis, held from the camera that sees it.

WHY. A body only the endzone camera sees stands on the endzone's own foot point. That camera looks
along the field, from ~88 m behind the goal line, so a few pixels of box or ankle error at the foot is
metres of error ALONG THE FIELD (x) and almost none across it (y). On play 1 (2026-09-16, 05q strip of
id 40 at 387-394) such a body stood on empty turf between two real men, and its wander along x was one
of the seven live-play hops the rulers still counted.

WHAT. For each RUN of consecutive body-frames the endzone alone sees, take the id's x from the frames
the SIDELINE saw it (x is the sideline's across-image axis, which it measures well): interpolated when
a sighting lies on both sides of the run within ``window`` frames, held when only one side has a
sighting within ``reach`` frames (a longer one-sided hold would pin a moving man to where he stood a
second ago -- measured 2026-09-16: it made an eight-frame fragment hop 0.4 m/frame). Then slide each
point along the endzone camera's own ground ray to that x. The point stays on its ray, so its
projection into the endzone image is unchanged by construction; only the coordinate that camera
cannot see moves. A run is slid whole or not at all (its median slide against ``max_move_m``): a
per-frame cap flickered bodies on and off the slide. The depth snap (render.depth_snap) is the same
idea the other way round.

MEASURED AND NOT ADOPTED (play 1, 2026-09-16, both A/Bs at the timeline, live play 300-460). Per-frame
version: live steps > 0.25 m/frame 7 -> 9, census 1.50 -> 1.50 (both teams nearer eleven), endzone
lower joints p50 51.8 -> 51.3 px. Per-run version with reach 6: live steps 5 -> 7, census 1.50 ->
1.60 (KC 10.98 -> 11.09), endzone/sideline/jitter unchanged; 611 body-frames slid, median 0.60 m.
Id 40, the body the strip showed on empty turf: his x under the hold differs from the endzone's own
by 0.1-0.3 m on 388-394 -- the endzone's x drift there (-27.1 -> -25.2 in ten frames, 6 m/s) matches
the sideline sightings on both sides, so the "hop" is a man running and the ghost is a lateral
offset the hold does not touch. The change loses on the ruler it was built for; it stays opt-in
(``load_play_timeline(blind_axis=True)``) for a play where endzone-only x drift is the defect.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.render.depth_snap import camera_ground_centre

WINDOW: int = 30            # a sideline sighting further than this on BOTH sides is not this run's evidence
REACH: int = 6              # a one-sided hold reaches at most this far
MAX_MOVE_M: float = 4.0     # never slide a run whose median slide is longer than this
MIN_AXIS_COS: float = 0.5   # the ray must run mostly along the blind axis for the slide to be well posed


def _runs(frames):
    """Consecutive-integer runs of a sorted frame list: ``[(first, last), ...]``."""
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def hold_blind_axis(ground: dict, views: dict, track, *, frame_shift: int = 0, window: int = WINDOW,
                    reach: int = REACH, max_move_m: float = MAX_MOVE_M, axis: int = 0,
                    seen_by: str = "sideline", only_by: str = "endzone"):
    """``(ground, moves)``: ``ground`` (frame -> {pid: xy}) with every run of body-frames whose
    ``views`` entry is exactly ``(only_by,)`` slid along ``track``'s ground ray (that camera's pose
    at ``frame + frame_shift``) to the ``axis`` coordinate taken from the id's ``seen_by`` sightings;
    ``moves`` lists the metres each touched body-frame moved. Untouched: runs with no sighting
    within ``window`` on both sides nor within ``reach`` on one, rays running across the axis,
    slides toward the camera, and runs whose median slide exceeds ``max_move_m``."""
    seen_frames: dict[int, list] = {}
    only_frames: dict[int, list] = {}
    for f, d in views.items():
        for pid, vs in d.items():
            if pid not in ground.get(f, {}):
                continue
            if seen_by in vs:
                seen_frames.setdefault(int(pid), []).append(int(f))
            elif tuple(vs) == (only_by,):
                only_frames.setdefault(int(pid), []).append(int(f))
    out = {f: dict(d) for f, d in ground.items()}
    moves = []
    for pid, fs_only in only_frames.items():
        fs = np.asarray(sorted(seen_frames.get(pid, [])))
        if len(fs) == 0:
            continue
        for a, b in _runs(sorted(fs_only)):
            i = int(np.searchsorted(fs, a))
            before = int(fs[i - 1]) if i > 0 else None
            after = int(fs[i]) if i < len(fs) else None
            two_sided = before is not None and after is not None and a - before <= window and after - b <= window
            if not two_sided:
                if before is not None and a - before <= reach:
                    after = None
                elif after is not None and after - b <= reach:
                    before = None
                else:
                    continue
            xb = ground[before][pid][axis] if before is not None else None
            xa = ground[after][pid][axis] if after is not None else None
            slid = []
            for f in range(a, b + 1):
                if xb is not None and xa is not None:
                    w = (f - before) / float(after - before)
                    target = (1.0 - w) * xb + w * xa
                else:
                    target = xb if xb is not None else xa
                fc = int(f) + int(frame_shift)
                if fc < 0 or fc >= len(track.conf) or track.conf[fc] <= 0:
                    slid = []
                    break
                centre = camera_ground_centre(track, fc)
                p = np.asarray(ground[f][pid], float)
                u = p - centre
                n = float(np.linalg.norm(u))
                if n < 1e-6 or abs(u[axis] / n) < MIN_AXIS_COS:
                    slid = []
                    break
                u = u / n
                s = (float(target) - centre[axis]) / u[axis]
                if s <= 0:
                    slid = []
                    break
                q = centre + s * u
                slid.append((f, q, float(np.linalg.norm(q - p))))
            if not slid or float(np.median([m for _, _, m in slid])) > max_move_m:
                continue
            for f, q, m in slid:
                out[f][pid] = q
                moves.append(m)
    return out, moves
