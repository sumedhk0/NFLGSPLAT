"""How much of the field a camera can see. A DIAGNOSTIC, not a gate.

WHY IT EXISTS, AND WHY IT IS NOT A GATE. This was written as a filter, on the
reasoning that All-22 footage is named for showing all twenty-two players and so
must show most of the field. Measured on the real play, that reasoning was
wrong, and the filter would have rejected the right camera and kept the wrong
one:

    lens      coverage   players' implied height   players' ground points
    12.2 deg    0.13     1.85 m (p10 1.5, p90 2.1)  on the field, 17 x 28 m
    62.5 deg    0.80     5.2 m  (p90 over 6)        y -69..-40 m, OFF the field

The boxes themselves settle it without any solve: the detector's person boxes
are ~140 px tall at 1080p. A 1.85 m player about 100 m away is 140 px only
behind a focal length near 7500 px -- a 12 to 15 degree lens. Behind a 62.5
degree lens they would be 31 px. All-22 sideline film frames the FORMATION,
about twenty metres of turf, and pans with the play; it does not show the
whole 120 yard field at once.

So this is the fifth measured case of a sensible prior making things worse
because the thing it was correcting was already right. The number is still
worth printing -- a camera that sees 80% of the field while placing players
five metres tall is describing a different world, and the two facts read
together say so at a glance -- but it decides nothing.
"""
from __future__ import annotations

import numpy as np

# An NFL field including both end zones: 120 yd by 53.3 yd.
FIELD_HALF_X_M: float = 54.86
FIELD_HALF_Y_M: float = 24.38

_GRID = 41


def field_coverage(K, R, t, width: int, height: int, *, z_plane: float = 0.0,
                   half_x: float = FIELD_HALF_X_M,
                   half_y: float = FIELD_HALF_Y_M) -> float:
    """Fraction of the field that falls inside the frame, 0 to 1.

    Points behind the camera do not count, which is what makes this meaningful
    for a camera pointed away from the field rather than merely zoomed in.
    """
    K = np.asarray(K, float)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    xs = np.linspace(-half_x, half_x, _GRID)
    ys = np.linspace(-half_y, half_y, _GRID)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel(),
                           np.full(gx.size, float(z_plane))])
    cam = pts @ R.T + t
    in_front = cam[:, 2] > 1e-6
    if not in_front.any():
        return 0.0
    uv = cam[in_front] @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    return float(inside.sum()) / float(len(pts))
