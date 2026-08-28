"""End-to-end identity for a play: tracks in, named players out.

The pieces existed separately -- grounding, stitching, jersey votes, formation
alignment, one-to-one assignment, cross-camera merge -- and were wired together
by hand each time, which meant the result could not be reproduced from the
command line. This is that wiring, in the order the evidence actually flows:

1. GROUND every detection through the calibrated camera, so distances are in
   metres rather than pixels. A gap of fifty pixels means nothing on its own;
   a metre is a metre anywhere on the field.
2. STITCH fragments into players. The tracker drops a player on occlusion and
   restarts them under a new id, which splits the two identity signals apart:
   jersey numbers read LATE, when players are near the camera and turned toward
   it, while formation alignment only means anything BEFORE the snap. Stitched,
   both land on one player and can reinforce each other.
3. CREDIT TRUNCATED READS to the numbers they could have come from, because OCR
   drops digits far more often than it invents them.
4. ASSIGN one-to-one against the 22 players the league says were on the field.
5. MERGE the cameras by jersey, then let geometry REFUTE any pairing where the
   two views disagree about where the player stood.
"""
from __future__ import annotations

import collections

import numpy as np

from nfl_gsplat.identity import formation as fm
from nfl_gsplat.identity.jersey_vote import (assign, credit_truncations,
                                             restrict_to_known)
from nfl_gsplat.identity.merge_cameras import (check_separation,
                                               drop_contradicted, merge)
from nfl_gsplat.tracking.stitch import stitch
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# A detection grounding outside this box is behind the camera or across the
# stadium; it is a bad foot point, not a player.
FIELD_HALF_X_M: float = 60.0
FIELD_HALF_Y_M: float = 28.0

# Formation only means something before the snap, while players are still set.
PRE_SNAP_FRAMES: int = 180

# Below this many grounded players a frame cannot be split into two elevens.
MIN_PLAYERS_FOR_FORMATION: int = 20


def ground_tracks(track, tracks_df, cam: str):
    """``({track: [(frame, x, y)]}, {frame: (ids, xy)})`` in field metres.

    Only VERIFIED frames are used: a frame with ``conf == 0`` had its pose
    filled in from a neighbour, so every position derived from it is a guess.
    """
    df = tracks_df[(tracks_df["cam"] == cam) & (tracks_df["track_id"] >= 0)]
    verified = set(int(f) for f in np.flatnonzero(np.asarray(track.conf) > 0))
    positions: dict[int, list] = collections.defaultdict(list)
    by_frame: dict[int, tuple] = {}
    for frame, rows in df[df["frame"].isin(verified)].groupby("frame"):
        intr, pose = track.at(int(frame))
        k_inv = np.linalg.inv(intr.K())
        centre = -pose.R.T @ pose.t
        ids, xy = [], []
        for r in rows.itertuples():
            ray = pose.R.T @ (k_inv @ np.array([r.foot_u, r.foot_v, 1.0]))
            if abs(ray[2]) < 1e-9:          # parallel to the turf, never lands
                continue
            p = (centre - centre[2] / ray[2] * ray)[:2]
            if abs(p[0]) < FIELD_HALF_X_M and abs(p[1]) < FIELD_HALF_Y_M:
                positions[int(r.track_id)].append((int(frame), p[0], p[1]))
                ids.append(int(r.track_id))
                xy.append(p)
        by_frame[int(frame)] = (ids, np.array(xy) if xy else np.zeros((0, 2)))
    return dict(positions), by_frame


def formation_positions(by_frame, player_of, los_x, offense, defense, *,
                        pre_snap_frames: int = PRE_SNAP_FRAMES):
    """``({player: position}, n_frames_used)`` from pre-snap alignment.

    A frame contributes only if it splits cleanly into eleven and eleven. That
    is strict on purpose: a frame where the detector missed two players would
    otherwise assign every role one seat over.
    """
    votes: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    used = 0
    for frame, (ids, xy) in sorted(by_frame.items()):
        if frame >= pre_snap_frames or len(ids) < MIN_PLAYERS_FOR_FORMATION:
            continue
        off_idx, def_idx, trench = fm.split_sides(xy, los_x)
        off_idx, def_idx = fm.resolve_trench(off_idx, def_idx, trench, xy, los_x)
        if len(off_idx) != 11 or len(def_idx) != 11:
            continue
        used += 1
        for k, role in fm.assign_offense_roles(xy[off_idx], offense, los_x).items():
            votes[player_of.get(ids[off_idx[k]], ids[off_idx[k]])][role] += 1
        for k, role in fm.assign_defense_roles(xy[def_idx], defense, los_x).items():
            votes[player_of.get(ids[def_idx[k]], ids[def_idx[k]])][role] += 1
    return {t: c.most_common(1)[0][0] for t, c in votes.items()}, used


def resolve_camera(track, tracks_df, cam, on_field, votes_by_track, *,
                   team_by_track=None, los_x=None, offense=None, defense=None,
                   fps: float = 59.94, truncation_credit: float | None = None):
    """Named players for ONE camera. Returns ``(identities, positions, stitch)``.

    ``positions`` is per stitched player, keyed by frame, ready for the
    cross-camera geometry check.
    """
    raw_positions, by_frame = ground_tracks(track, tracks_df, cam)
    player_of = stitch(raw_positions, fps=fps)

    votes: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for tid, counter in votes_by_track.items():
        votes[player_of.get(int(tid), int(tid))].update(counter)

    teams = {}
    for tid, team in (team_by_track or {}).items():
        teams.setdefault(player_of.get(int(tid), int(tid)), team)

    roles: dict[int, str] = {}
    used = 0
    if los_x is not None and offense is not None and defense is not None:
        roles, used = formation_positions(by_frame, player_of, los_x,
                                          offense, defense)

    jerseys = [int(j) for j in on_field["jersey_number"]]
    kwargs = {} if truncation_credit is None else {"weight": truncation_credit}
    pool = {t: credit_truncations(c, jerseys, **kwargs)
            for t, c in restrict_to_known(votes, jerseys).items()}
    for t in roles:                      # alignment-only players still compete
        pool.setdefault(t, collections.Counter())

    identities = assign(pool, on_field, team_of_track=teams or None,
                        position_of_track=roles or None)
    _LOG.info("%s: %d fragments -> %d players, %d clean pre-snap frames, "
              "%d identified", cam, len(raw_positions),
              len(set(player_of.values())), used, len(identities))

    positions: dict[int, dict[int, np.ndarray]] = collections.defaultdict(dict)
    for tid, rows in raw_positions.items():
        for frame, x, y in rows:
            positions[player_of.get(tid, tid)][frame] = np.array([x, y])
    return identities, dict(positions), player_of


def resolve_play(per_camera, cam_a: str = "sideline", cam_b: str = "endzone"):
    """Merge per-camera identities and drop the ones geometry refutes.

    ``per_camera`` maps camera -> ``(identities, positions)``.
    """
    merged = merge({cam: ident for cam, (ident, _pos) in per_camera.items()})
    positions = {cam: pos for cam, (_ident, pos) in per_camera.items()}
    checks = {}
    if cam_a in positions and cam_b in positions:
        checks = check_separation(merged, positions, cam_a, cam_b)
        merged = drop_contradicted(merged, checks)
    return merged, checks
