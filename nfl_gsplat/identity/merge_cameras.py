"""Merge per-camera identities into one roster, and read correspondence off it.

Geometric cross-camera matching was measured three times and failed three times
(see ``docs/RESULTS_first_end_to_end.md``): median residual 3.3-4.5 m against a
player spacing of 1-2 m, so the nearest track in the other camera is frequently
the wrong man. The failure is not a calibration fault -- the axis biases are
0.2-0.6 m -- it is that positional precision is comparable to player spacing, and
no assignment rule recovers information that was never measured.

Identity inverts the dependency. If the sideline feed independently names a track
#85 and the endzone feed independently names one #85, they are the same player,
with no geometry involved at all. Correspondence falls out of the merge for free,
and it is available on every frame either camera sees rather than only the 86
where both are verified.

Two cameras also cover for each other. The sideline sees numbers late, when
players turn toward it; the endzone is zoomed far tighter and reads backs the
sideline never sees. A player missed by one is often read easily by the other, so
the union is substantially larger than either -- 17 of 22 on play_001 against 10
and 14 separately.

The overlap is NOT the free correctness check it first appears to be, and this
module originally claimed that it was. Two feeds agreeing on a number proves
less than it looks, because each is forced to emit every jersey at most once
regardless of evidence; measured on play_001, two of the four testable
agreements were 11 m and 16 m apart. So geometry returns, in the only role it
can actually play here: it cannot CREATE a correspondence, but it can REFUTE
one. See :func:`check_separation`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class PlayerIdentity:
    """One player, as seen by however many cameras identified them."""

    jersey: int
    player: str
    team: str
    height_m: float
    weight_lb: float = 0.0
    tracks: dict[str, int] = field(default_factory=dict)   # camera -> track id
    votes: dict[str, int] = field(default_factory=dict)    # camera -> jersey votes

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted(self.tracks))

    @property
    def corroborated(self) -> bool:
        """True when two or more cameras named this player independently.

        A HINT, not a proof, and the difference was measured rather than
        reasoned about. Of the four corroborated pairs testable on play_001,
        two turned out to be 11 m and 16 m apart on the field -- different
        people wearing the same label. Each camera solves a FORCED one-to-one
        assignment against the same 22 jerseys, so every number gets emitted at
        most once whether or not the evidence supports it, and two feeds can
        agree on a label with neither being right.

        Use :func:`check_separation` before trusting this.
        """
        return len(self.tracks) > 1

    @property
    def total_votes(self) -> int:
        return sum(self.votes.values())


def merge(per_camera: dict[str, list]) -> dict[int, PlayerIdentity]:
    """``{camera: [TrackIdentity, ...]}`` -> ``{jersey: PlayerIdentity}``.

    Merging on JERSEY rather than on track is what makes this safe: each camera
    has already solved a one-to-one assignment against the same 22 players, so a
    jersey identifies the same person in every feed by construction.
    """
    out: dict[int, PlayerIdentity] = {}
    for camera, identities in per_camera.items():
        for ident in identities:
            jersey = int(ident.jersey)
            existing = out.get(jersey)
            if existing is None:
                out[jersey] = PlayerIdentity(
                    jersey=jersey, player=str(ident.player),
                    team=str(ident.team), height_m=float(ident.height_m),
                    weight_lb=float(getattr(ident, "weight_lb", 0.0)),
                    tracks={camera: int(ident.track_id)},
                    votes={camera: int(ident.votes)})
                continue
            # Both cameras resolved this jersey against the same participation
            # table, so the names must agree. If they ever do not, the tables
            # differ and every identity downstream is suspect -- fail rather
            # than silently keep whichever arrived first.
            if existing.player != str(ident.player):
                raise CalibrationError(
                    f"jersey #{jersey} is {existing.player!r} in "
                    f"{existing.cameras} but {ident.player!r} in {camera!r}. "
                    "The cameras were resolved against different rosters; "
                    "re-run players_on_play for the same game and play id.")
            existing.tracks[camera] = int(ident.track_id)
            existing.votes[camera] = int(ident.votes)

    agreed = sum(1 for p in out.values() if p.corroborated)
    _LOG.info("merged %d cameras: %d players, %d corroborated by 2+",
              len(per_camera), len(out), agreed)
    return out


def correspondence(merged: dict[int, PlayerIdentity], cam_a: str, cam_b: str):
    """``{track in cam_a: track in cam_b}`` for players both cameras named.

    This is the cross-camera correspondence that geometry could not deliver.
    """
    out = {}
    for player in merged.values():
        if cam_a in player.tracks and cam_b in player.tracks:
            out[player.tracks[cam_a]] = player.tracks[cam_b]
    return out


def coverage(merged: dict[int, PlayerIdentity], on_field):
    """``(identified, missing)`` against the 22 players known to be playing."""
    named = set(merged)
    missing = [(int(r["jersey_number"]), str(r["full_name"]), str(r["team"]),
                str(r["position"]))
               for _, r in on_field.iterrows()
               if int(r["jersey_number"]) not in named]
    return len(named), missing


# Two cameras agreeing on a NUMBER is much weaker evidence than it looks, and
# measurement is what showed it. Of the 4 corroborated pairs testable on
# play_001, two were 11 m and 16 m apart in field metres -- not a precision
# problem, a different human being. Each camera solves a FORCED one-to-one
# assignment against the same 22 jerseys, so both are compelled to emit every
# number at most once; two feeds can therefore land on the same label with
# neither being right. Agreement is a hint, not a proof.
#
# Geometry cannot CREATE correspondence (measured three ways, all negative) but
# it can DESTROY a false one, and that asymmetry is the useful part: a claim
# that two tracks are one player is refutable by a single well-grounded frame.
MAX_SEPARATION_M: float = 4.0

# Below this many shared frames the median is not meaningful and the pair is
# reported as untestable rather than as passing. Silence is not a pass.
MIN_SHARED_FRAMES: int = 20


@dataclass(frozen=True)
class Consistency:
    jersey: int
    separation_m: float | None      # None when untestable
    n_frames: int

    @property
    def testable(self) -> bool:
        return self.separation_m is not None

    @property
    def contradicted(self) -> bool:
        return self.testable and self.separation_m > MAX_SEPARATION_M


def check_separation(merged, positions, cam_a: str, cam_b: str, *,
                     max_separation_m: float = MAX_SEPARATION_M,
                     min_shared_frames: int = MIN_SHARED_FRAMES):
    """Median field-metre gap between the two tracks each player was given.

    ``positions`` is ``{camera: {track: {frame: (x, y)}}}`` in field metres.
    Returns ``{jersey: Consistency}`` for players both cameras named.

    Identity is derived from jersey pixels and formation alignment and never
    compares the cameras' positions, so this is a genuinely independent test of
    the identities AND of both calibrations at once.
    """
    import numpy as np

    out: dict[int, Consistency] = {}
    for jersey, player in merged.items():
        if cam_a not in player.tracks or cam_b not in player.tracks:
            continue
        pa = positions.get(cam_a, {}).get(player.tracks[cam_a], {})
        pb = positions.get(cam_b, {}).get(player.tracks[cam_b], {})
        shared = sorted(set(pa) & set(pb))
        if len(shared) < min_shared_frames:
            out[jersey] = Consistency(jersey, None, len(shared))
            continue
        gaps = [float(np.hypot(*(np.asarray(pa[f]) - np.asarray(pb[f]))))
                for f in shared]
        out[jersey] = Consistency(jersey, float(np.median(gaps)), len(shared))

    bad = [c.jersey for c in out.values() if c.contradicted]
    if bad:
        _LOG.warning("cross-camera geometry CONTRADICTS %d identities: %s "
                     "(more than %.1f m apart)", len(bad),
                     ", ".join(f"#{j}" for j in sorted(bad)), max_separation_m)
    return out


def drop_contradicted(merged, checks, votes_win_by: int = 3):
    """Keep the better-evidenced camera for each contradicted player.

    When geometry refutes a pairing at least one side is wrong, and the vote
    count says which: on play_001 the sideline called a track Kyler Murray on
    ZERO jersey votes -- from alignment alone, via the sole-candidate rule that
    bypasses the vote thresholds -- while the endzone read #1 seventy-three
    times on a different track eleven metres away.

    If neither side wins by ``votes_win_by`` the player is dropped entirely. A
    wrong identity is worse than a missing one: it silently attaches the wrong
    body shape, the wrong avatar and the wrong correspondence, and nothing
    downstream can detect it.
    """
    out, dropped = {}, []
    for jersey, player in merged.items():
        check = checks.get(jersey)
        if check is None or not check.contradicted:
            out[jersey] = player
            continue
        best = max(player.votes, key=lambda c: player.votes[c])
        rest = max((v for c, v in player.votes.items() if c != best), default=0)
        if player.votes[best] - rest < votes_win_by:
            dropped.append(jersey)
            continue
        out[jersey] = PlayerIdentity(
            jersey=player.jersey, player=player.player, team=player.team,
            height_m=player.height_m, weight_lb=player.weight_lb,
            tracks={best: player.tracks[best]},
            votes={best: player.votes[best]})
    if dropped:
        _LOG.warning("dropped %d players whose cameras disagree with no clear "
                     "winner: %s", len(dropped),
                     ", ".join(f"#{j}" for j in sorted(dropped)))
    return out
