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
the union is substantially larger than either -- and the OVERLAP is a free
correctness check nothing else in the pipeline provides, since the two cameras
share no pixels, no tracker state and no calibration.
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
    tracks: dict[str, int] = field(default_factory=dict)   # camera -> track id
    votes: dict[str, int] = field(default_factory=dict)    # camera -> jersey votes

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted(self.tracks))

    @property
    def corroborated(self) -> bool:
        """True when two or more cameras named this player independently.

        This is the strongest identity claim the pipeline can make. The cameras
        share no pixels, no tracker and no calibration, so agreement cannot come
        from a common-mode error.
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
