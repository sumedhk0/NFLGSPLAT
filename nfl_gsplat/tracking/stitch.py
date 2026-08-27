"""Join track fragments that belong to the same player.

BoT-SORT drops a track when a player is occluded -- routine in football, where
twenty-two bodies converge on one point -- and starts a new id when they emerge.
On play_001's sideline feed that produced 150 tracks for 22 players, and it split
the play in two: jersey numbers read best LATE, when players are near the camera
and turned toward it, while formation alignment only means anything BEFORE the
snap. The two strongest identity signals therefore landed on different track ids
belonging to the same person, and neither could reinforce the other.

Stitching is done in FIELD METRES, not pixels. A gap of fifty pixels means
nothing on its own -- it is metres near the far sideline and centimetres near
the camera -- whereas a player's speed is a physical constant. A sprinter covers
about 10 m/s, so a fragment starting 8 m away half a second after another ended
cannot be the same person, and one starting 2 m away can.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Fastest a player moves, with headroom. Used as a physical bound on how far a
# fragment may travel during the gap, rather than a fixed distance that would be
# too tight for a receiver and too loose for a lineman.
MAX_SPEED_M_S: float = 11.0

# Beyond this the gap is too long to trust: players cross, and a plausible-
# looking join between two different people is worse than two honest fragments,
# because it welds one player's jersey votes onto another's alignment.
MAX_GAP_S: float = 1.5

# Even a stationary player has foot-point noise; this is the floor on how close
# two fragments must be, independent of the gap.
MIN_RADIUS_M: float = 1.5


def track_endpoints(positions):
    """``{track: (first_frame, last_frame, first_xy, last_xy)}``.

    ``positions`` maps track -> list of ``(frame, x, y)`` in field metres.
    """
    out = {}
    for track_id, rows in positions.items():
        if not rows:
            continue
        ordered = sorted(rows, key=lambda r: r[0])
        first, last = ordered[0], ordered[-1]
        out[int(track_id)] = (int(first[0]), int(last[0]),
                              np.array(first[1:], float),
                              np.array(last[1:], float))
    return out


def stitch(positions, *, fps: float = 59.94, max_gap_s: float = MAX_GAP_S,
           max_speed: float = MAX_SPEED_M_S, min_radius_m: float = MIN_RADIUS_M):
    """Link fragments into players. Returns ``{track_id: player_id}``.

    Greedy in time: fragments are considered in the order they end, and each is
    joined to the best still-unclaimed fragment that starts after it. A global
    assignment is not used here on purpose -- a fragment can only continue into
    ONE successor and only forward in time, so the greedy order is already the
    constraint the solver would enforce, and it stays readable.
    """
    ends = track_endpoints(positions)
    if not ends:
        return {}

    order = sorted(ends, key=lambda t: ends[t][1])          # by last frame
    successor: dict[int, int] = {}
    claimed: set[int] = set()

    for a in order:
        _fa, la, _pa, end_xy = ends[a]
        best, best_cost = None, float("inf")
        for b in ends:
            if b == a or b in claimed:
                continue
            fb, _lb, start_xy, _pb = ends[b]
            gap_frames = fb - la
            if gap_frames <= 0:                              # must be forward
                continue
            gap_s = gap_frames / float(fps)
            if gap_s > max_gap_s:
                continue
            reach = max(min_radius_m, max_speed * gap_s)
            dist = float(np.hypot(*(start_xy - end_xy)))
            if dist > reach:
                continue
            # Prefer the nearest continuation, then the soonest.
            cost = dist + 0.5 * gap_s
            if cost < best_cost:
                best, best_cost = b, cost
        if best is not None:
            successor[a] = best
            claimed.add(best)

    # Walk the chains: every fragment inherits the id of the earliest fragment
    # it descends from, so a player keeps one identity across all their pieces.
    player_of: dict[int, int] = {}
    starts = [t for t in ends if t not in claimed]
    for root in starts:
        node = root
        while True:
            player_of[node] = root
            node = successor.get(node)
            if node is None or node in player_of:
                break

    for t in ends:                                            # any stragglers
        player_of.setdefault(t, t)

    n_players = len(set(player_of.values()))
    _LOG.info("stitching: %d fragments -> %d players (%d joins)",
              len(ends), n_players, len(ends) - n_players)
    return player_of


def apply_to_tracks(tracks_df, player_of, *, column: str = "player_track"):
    """Add a stitched-id column to a tracks frame."""
    out = tracks_df.copy()
    out[column] = [player_of.get(int(t), int(t)) for t in out["track_id"]]
    return out
