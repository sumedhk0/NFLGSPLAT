"""How much of the field a camera can actually see.

WHY THIS EXISTS. All-22 footage is named for what it shows: all twenty-two
players, which means substantially the whole field. That is a hard external fact
about the footage, and nothing in the pipeline was using it.

It should have been. The sideline solve settled, repeatedly and across
independent frame samples, on a camera 95 m out behind a 12.2 degree lens. At
that range such a lens spans about 21 m of turf -- a fifth of the field -- and
the solve duly placed all twenty-three detected players inside a 17 by 28 m
patch. Three separate samples agreed on it, which looked like evidence and was
not: they agreed because the same wrong labelling recurs, not because it is
right. Meanwhile a 62.5 degree candidate, which is what an All-22 sideline
camera actually looks like, was thrown away for scoring badly on player height.

The checks in place could not catch this. Paint residual cannot -- a field of
parallel lines fits a telephoto view of one corner as happily as a wide view of
the whole. Player height cannot either: it constrains how far away the camera
is, and a camera can be the right distance behind quite the wrong lens.

Coverage is a different question from either, and a blunt one: project the field
into the image and count how much of it lands on the sensor. It needs no
detections, no labelling and no tracking -- only the camera and the frame size.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# An NFL field including both end zones: 120 yd by 53.3 yd.
FIELD_HALF_X_M: float = 54.86
FIELD_HALF_Y_M: float = 24.38

# How much of the field an All-22 camera must see. Deliberately well below what
# a correct camera scores, so this rejects the plainly impossible rather than
# adjudicating between plausible ones -- the same role the mount check plays.
MIN_ALL22_COVERAGE: float = 0.35

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


def sees_enough_of_the_field(K, R, t, width: int, height: int, *,
                             minimum: float = MIN_ALL22_COVERAGE) -> bool:
    """Could this camera have produced All-22 footage at all?"""
    got = field_coverage(K, R, t, width, height)
    if got < minimum:
        _LOG.info("camera sees %.0f%% of the field, under %.0f%%",
                  100 * got, 100 * minimum)
    return got >= minimum
