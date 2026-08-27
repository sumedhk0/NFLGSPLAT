"""Infer each player's POSITION from where they line up.

Jersey OCR reads a number off a 60-180 px crop and gets it right maybe a tenth
of the time; it resolved 9 of 22 players on play_001 even with the candidate set
narrowed to the 22 known to be on the field. Formation carries far more
information and needs no pixels at all beyond a foot point:

* the play-by-play gives the LINE OF SCRIMMAGE as a yard line, the team in
  POSSESSION, and the formation ("SHOTGUN");
* participation gives the exact PERSONNEL on each side -- "1 C, 2 G, 1 QB,
  1 RB, 2 T, 1 TE, 3 WR" against "3 CB, 2 DE, 1 DT, 1 FS, 2 ILB, 1 OLB, 1 SS";
* the calibration puts every detection on the field in metres.

Football alignment is highly constrained. The offensive line stands on the ball,
the centre over it, guards inside tackles. A shotgun quarterback is about five
yards back, the running back beside him, receivers wide. The defensive line
mirrors across the ball, linebackers behind them, corners out on the receivers,
safeties deepest. Those rules assign a POSITION to nearly every player from
geometry alone -- and position, joined to the roster, narrows identity to the
handful of players who play it.

This does not replace jersey reading; it constrains it. A track whose geometry
says "left tackle" can only be one of the two tackles on the field, and a jersey
vote that agrees is then near-certain rather than merely likely.
"""
from __future__ import annotations

import collections
import re

import numpy as np

from nfl_gsplat.calibration.field_landmarks import GOAL_LINE_X_M, YARD_TO_M
from nfl_gsplat.errors import SetupError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Positions that stand ON the line of scrimmage, offence and defence.
OFFENSIVE_LINE = ("C", "G", "T")
DEFENSIVE_LINE = ("DE", "DT", "NT")
LINEBACKERS = ("ILB", "OLB", "MLB", "LB")
DEFENSIVE_BACKS = ("CB", "FS", "SS", "S", "DB")

# A shotgun back stands beside the quarterback, not merely deep. Measured on
# play_001, "deepest remaining" put the back 9-14 m wide across three frames --
# a receiver who happened to be deep.
RB_MAX_FROM_QB_M: float = 4.0


def parse_personnel(text: str) -> collections.Counter:
    """``"1 C, 2 G, 1 QB"`` -> ``Counter({'C': 1, 'G': 2, 'QB': 1})``."""
    out: collections.Counter = collections.Counter()
    for count, label in re.findall(r"(\d+)\s+([A-Za-z]+)", str(text or "")):
        out[label.upper()] += int(count)
    return out


def line_of_scrimmage_x(yrdln: str, posteam: str, *,
                        own_goal_x_m: float = -GOAL_LINE_X_M) -> float:
    """World X of the line of scrimmage, from the play-by-play yard line.

    ``yrdln`` is like ``"ARI 26"`` -- a side of the field and a yard number,
    which counts UP from that side's own goal line to midfield and back down.
    Deriving this from the play-by-play rather than from where players happen to
    be standing matters: a density estimate is pulled toward whichever side has
    more men near the ball, and on play_001 that biased it 1.1 m.

    ``own_goal_x_m`` is the X of the possessing team's own goal line, which
    depends on which way they are driving and must come from the calibration.
    """
    match = re.match(r"\s*([A-Z]{2,3})\s+(\d+)", str(yrdln or ""))
    if not match:
        raise SetupError(
            f"cannot parse yard line {yrdln!r}; expected something like "
            "'ARI 26'. Without it the line of scrimmage would have to be "
            "guessed from player density, which is biased.")
    side, yards = match.group(1), int(match.group(2))
    direction = 1.0 if own_goal_x_m < 0 else -1.0
    if side.upper() == str(posteam).upper():
        # own side: yards measured from the possessing team's goal line
        return own_goal_x_m + direction * yards * YARD_TO_M
    # opponent's side: yards measured back from the far goal line
    return -own_goal_x_m - direction * yards * YARD_TO_M


def split_sides(xy, los_x: float, *, driving: float = +1.0, tol_m: float = 0.9):
    """``(offense_idx, defense_idx)`` by which side of the ball each player is on.

    ``driving`` is +1 when the offence advances toward +X. The offence lines up
    BEHIND the ball, so it occupies the smaller X in that case.

    ``tol_m`` exists because the two lines straddle the ball: on play_001 the
    offensive and defensive linemen sit within about 1.3 m of each other, so a
    hard cut at the ball splits the trenches arbitrarily. Players inside the
    tolerance are assigned by nearest-side counting rather than by sign.
    """
    xy = np.asarray(xy, float)
    rel = (xy[:, 0] - los_x) * driving
    offense = list(np.flatnonzero(rel < -tol_m))
    defense = list(np.flatnonzero(rel > tol_m))
    trench = list(np.flatnonzero(np.abs(rel) <= tol_m))
    return offense, defense, trench


def resolve_trench(offense, defense, trench, xy, los_x, *, driving: float = +1.0,
                   n_per_side: int = 11):
    """Give the players straddling the ball to whichever side is short.

    Both lines stand on the ball, so a cut by sign alone splits the trenches
    arbitrarily -- measured on play_001, three linemen fell inside a 0.9 m
    tolerance. But each side fields exactly ``n_per_side``, and that is enough
    to settle it: on that frame the defence already had its eleven, so all three
    were offensive linemen.

    When counting cannot decide (both sides short), the tie is broken by which
    side of the ball each player is actually on, which is the best available
    evidence and no worse than the arbitrary cut it replaces.
    """
    offense, defense = list(offense), list(defense)
    need_off = n_per_side - len(offense)
    need_def = n_per_side - len(defense)
    rel = (np.asarray(xy, float)[:, 0] - los_x) * driving

    if need_off <= 0 and need_def <= 0:
        return offense, defense
    if need_def <= 0:
        return offense + list(trench), defense
    if need_off <= 0:
        return offense, defense + list(trench)

    # Both short: nearest side wins, offence first for ties on the ball itself.
    for i in sorted(trench, key=lambda k: rel[k]):
        if len(offense) < n_per_side and (rel[i] <= 0 or len(defense) >= n_per_side):
            offense.append(i)
        else:
            defense.append(i)
    return offense, defense


def assign_offense_roles(xy, personnel, los_x: float, *, driving: float = +1.0):
    """Positions for the offence, from alignment. Returns ``{index: position}``.

    Order matters: the line is identified first because everything else is
    defined relative to it -- the quarterback is behind the CENTRE, receivers are
    wide of the TACKLES.
    """
    xy = np.asarray(xy, float)
    idx = list(range(len(xy)))
    rel_x = (xy[:, 0] - los_x) * driving
    want = collections.Counter(personnel)
    roles: dict[int, str] = {}

    n_line = sum(want[p] for p in OFFENSIVE_LINE)
    # The line is the tightest CONTIGUOUS run in Y among players near the ball,
    # not simply the n closest to it. Split receivers line up ON the line of
    # scrimmage too -- perfectly legal, and on play_001 that put a "tackle" nine
    # metres wide and threaded the tight end through the middle of the line.
    # Interior linemen stand shoulder to shoulder, so contiguity is what
    # actually distinguishes them.
    near = sorted((i for i in idx if abs(rel_x[i]) <= 2.5),
                  key=lambda i: xy[i, 1])
    if len(near) < n_line:
        near = sorted(idx, key=lambda i: abs(rel_x[i]))[:n_line]
        near.sort(key=lambda i: xy[i, 1])
    best, best_span = near[:n_line], float("inf")
    for start in range(0, max(1, len(near) - n_line + 1)):
        window = near[start:start + n_line]
        if len(window) < n_line:
            break
        span = xy[window[-1], 1] - xy[window[0], 1]
        if span < best_span:
            best, best_span = window, span
    on_line = list(best)
    # Along the line, order by cross-field position: T G C G T outward from the
    # centre, which is what "inside" and "outside" mean.
    on_line.sort(key=lambda i: xy[i, 1])
    mid = len(on_line) // 2
    labels = (["T"] * (want["T"] // 2) + ["G"] * (want["G"] // 2) + ["C"] * want["C"]
              + ["G"] * (want["G"] - want["G"] // 2) + ["T"] * (want["T"] - want["T"] // 2))
    for i, label in zip(on_line, labels[:len(on_line)]):
        roles[i] = label

    rest = [i for i in idx if i not in roles]
    if not rest:
        return roles

    centre_y = float(np.median([xy[i, 1] for i in on_line])) if on_line else 0.0

    # Quarterback: deepest behind the ball, near the centre of the formation.
    backfield = sorted(rest, key=lambda i: rel_x[i])
    qb = None
    if want["QB"] and backfield:
        qb = min(backfield[:3], key=lambda i: abs(xy[i, 1] - centre_y))
        roles[qb] = "QB"
        rest.remove(qb)

    # Running back: beside the quarterback, NOT merely the next deepest. Taking
    # depth alone put the "back" fourteen metres wide on play_001 -- a receiver
    # who happened to be deep. If nobody is within reach the back has motioned
    # out or the formation is empty, and guessing would attach a real player's
    # identity to the wrong body; leaving it unassigned is recoverable.
    if want["RB"] and qb is not None and rest:
        near = min(rest, key=lambda i: float(np.hypot(*(xy[i] - xy[qb]))))
        if float(np.hypot(*(xy[near] - xy[qb]))) <= RB_MAX_FROM_QB_M:
            roles[near] = "RB"
            rest.remove(near)
        else:
            _LOG.info("no back within %.1f m of the quarterback; leaving the "
                      "RB unassigned rather than labelling a receiver",
                      RB_MAX_FROM_QB_M)

    # Receivers: widest from the formation centre. Tight ends are what is left,
    # which is right by construction -- they align between the tackles and the
    # numbers.
    rest.sort(key=lambda i: -abs(xy[i, 1] - centre_y))
    for label in (["WR"] * want["WR"] + ["TE"] * want["TE"]):
        if not rest:
            break
        roles[rest.pop(0)] = label
    return roles


def assign_defense_roles(xy, personnel, los_x: float, *, driving: float = +1.0):
    """Positions for the defence, by depth then width."""
    xy = np.asarray(xy, float)
    idx = list(range(len(xy)))
    rel_x = (xy[:, 0] - los_x) * driving
    want = collections.Counter(personnel)
    roles: dict[int, str] = {}

    n_dl = sum(want[p] for p in DEFENSIVE_LINE)
    by_depth = sorted(idx, key=lambda i: rel_x[i])
    for i in by_depth[:n_dl]:
        roles[i] = "DT" if want["DT"] and abs(xy[i, 1]) < 3.0 else "DE"

    rest = [i for i in by_depth if i not in roles]
    n_lb = sum(want[p] for p in LINEBACKERS)
    for i in rest[:n_lb]:
        roles[i] = "ILB" if abs(xy[i, 1]) < 5.0 and want["ILB"] else "OLB"
    rest = rest[n_lb:]

    # Corners take the width, safeties the depth -- that is the difference.
    rest.sort(key=lambda i: -abs(xy[i, 1]))
    for i in rest[:want["CB"]]:
        roles[i] = "CB"
    for i in rest[want["CB"]:]:
        roles[i] = "FS" if rel_x[i] == max((rel_x[j] for j in rest[want["CB"]:]),
                                           default=0.0) else "SS"
    return roles
