"""Break the paint ambiguity with the one thing paint cannot fake: people.

The problem this exists for. A football field is a plane covered in parallel
lines, and more than one camera explains it. Measured on one All-22 clip, a
62 degree camera 160 m out and a 13 degree camera 87 m out both reproduced the
paint to a few pixels -- and every check available from the paint itself said
both were fine, because both were, as far as the paint is concerned.

What the paint cannot say is how BIG anything is. Players can. A person standing
on the turf has their feet on the plane, so the feet ray fixes where they are;
the head ray then implies how tall they are, and that number is only right for
the right camera. An NFL player in pads and helmet is about 1.85 m. A camera
that has the field twice as far away makes everyone twice as tall, and nothing
about the lines objects.

This needs no tracking, no identity, and no second view -- just person boxes,
which the detector already produces at 94% recall. It is deliberately a SCORE
rather than a constraint: it ranks cameras that the paint cannot separate, and
says nothing when the detections are too few or too poor to speak.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# An NFL player with helmet and pads. The spread across a roster is real but
# small next to the errors this is meant to catch, which are factors, not
# centimetres.
EXPECTED_PLAYER_M: float = 1.85

# Heights outside this are not people, and usually mean the box is a referee
# kneeling, two players merged, or a detection on the crowd.
PLAUSIBLE_PLAYER_M: tuple[float, float] = (1.2, 2.6)

MIN_PLAYERS: int = 6


def implied_heights(K, R, t, boxes, *, z_plane: float = 0.0):
    """Height of each person box, in metres, given a camera and the ground.

    The feet are on the plane, so the bottom-centre ray fixes the ground point.
    The head is directly above it, so the top-centre ray meets the vertical
    through that point at the person's height.
    """
    K = np.asarray(K, float)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0)

    centre = -R.T @ t
    Kinv = np.linalg.inv(K)
    out = []
    for x1, y1, x2, y2 in boxes:
        foot = np.array([(x1 + x2) / 2.0, y2, 1.0])
        head = np.array([(x1 + x2) / 2.0, y1, 1.0])
        d_foot = R.T @ (Kinv @ foot)
        d_head = R.T @ (Kinv @ head)
        if abs(d_foot[2]) < 1e-9:
            continue
        s = (z_plane - centre[2]) / d_foot[2]
        if s <= 0:
            continue
        ground = centre + s * d_foot          # where the player stands
        # The head ray must pass over that same ground point; solve for the
        # height at which it does, using whichever horizontal axis is better
        # conditioned for this ray.
        axis = 0 if abs(d_head[0]) > abs(d_head[1]) else 1
        if abs(d_head[axis]) < 1e-9:
            continue
        s_head = (ground[axis] - centre[axis]) / d_head[axis]
        if s_head <= 0:
            continue
        out.append(float(centre[2] + s_head * d_head[2] - z_plane))
    return np.asarray(out)


def height_score(K, R, t, boxes, *, z_plane: float = 0.0,
                 expected_m: float = EXPECTED_PLAYER_M):
    """``(cost, n_used, median_height)``; lower cost is a better camera.

    ``inf`` when too few boxes give a usable height, so a camera is never
    preferred on the strength of two detections.
    """
    h = implied_heights(K, R, t, boxes, z_plane=z_plane)
    keep = h[(h > PLAUSIBLE_PLAYER_M[0]) & (h < PLAUSIBLE_PLAYER_M[1])]
    if len(keep) < MIN_PLAYERS:
        return float("inf"), int(len(keep)), float("nan")
    median = float(np.median(keep))
    # Both terms matter: the wrong SCALE moves the median, and a wrong pose
    # spreads the heights even when the median happens to land near 1.85.
    spread = float(np.subtract(*np.percentile(keep, [75, 25])))
    return abs(median - expected_m) + spread, int(len(keep)), median
