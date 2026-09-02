"""Choose the two cameras of a play together, not one at a time.

WHY. A camera can pass every test available from ONE view and still be wrong in
a way only the other view can see. Measured on a production play, the sideline
and endzone cameras each looked fine alone -- both physically possible, both
placing players at believable heights -- and disagreed about the width of the
field by a factor of three:

    sideline   players spanned y  -8.4 ..  6.4 m
    endzone    players spanned y -27.2 .. 17.5 m

Only three of twenty-two players matched across the pair, because the two
cameras were describing different worlds. Nothing in either view objects: the
field's own lines fix Y only if the rows were labelled correctly, and a wrong
labelling is self-consistent within the view that made it.

WHAT SETTLES IT. Two views of the same moment must place the SAME PLAYERS IN
THE SAME PLACES. That is a joint property, unavailable to either camera alone,
and it is exactly what the reconstruction needs to be true.

HOW, AND WHY NOT THE OBVIOUS WAY. The obvious score is triangulation residual:
pair the detections across views and see how well the rays meet. Measured, that
does not work here, and the reason is the camera geometry rather than the code.
The sideline and endzone cameras are nearly perpendicular, so a sideline ray is
close to a vertical plane at one X and an endzone ray close to a vertical plane
at one Y -- and two such planes always meet. Every pairing triangulates almost
perfectly, whatever the cameras are, so residual ranks nothing. On the synthetic
case a threefold Y stretch still matched most players.

So the score is the diagnostic that found the bug instead: put each view's
players on the turf independently, using the ray through their feet, and ask
whether the two ground clouds are the SAME CLOUD, in metres. A stretched or
mislabelled camera moves its cloud tens of metres and matches nothing. This uses
no triangulation, so the degeneracy above cannot hide in it.

This is the same rule that has settled every ambiguity in this pipeline: score
the thing actually wanted, not a proxy for it. Paint residual preferred cameras
that were kilometres away; player height fixed the scale but not the aspect;
cross-view ground agreement is what remains.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.from_paint import ray_to_plane
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# A pair must reconcile at least this many players to be believed at all. Real
# formations put 22 on the field and both cameras see most of them; a handful of
# matches is what a wrong pair produces by accident.
MIN_MATCHED: int = 8

# How far apart two views may place the same player and still be the same
# player. Generous next to the errors this catches, which are tens of metres,
# and next to the width of a person, which is where foot-point noise lives.
MAX_GAP_M: float = 2.5

FIELD_HALF_X_M: float = 56.0
FIELD_HALF_Y_M: float = 26.0


def ground_points(cam, feet_uv, *, z_plane: float = 0.0):
    """Where each detection stands, from the ray through its feet.

    Feet are on the turf, so this needs no second view and no triangulation --
    which is the point: it is an independent statement from each camera about
    where the players are, and independent statements can be compared.
    """
    K, R, t = cam
    feet_uv = np.asarray(feet_uv, float).reshape(-1, 2)
    out = np.full((len(feet_uv), 2), np.nan)
    for i, uv in enumerate(feet_uv):
        try:
            p = ray_to_plane(K, R, t, uv, z_plane)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if p is not None and np.all(np.isfinite(p)):
            out[i] = p
    return out


def match_count(cam_a, feet_a, cam_b, feet_b, *, max_gap_m: float = MAX_GAP_M,
                field_half_x: float = FIELD_HALF_X_M,
                field_half_y: float = FIELD_HALF_Y_M, z_plane: float = 0.0):
    """How many players the two cameras place in the same spot, and where.

    ``feet_*`` are BOTTOM-CENTRE image points -- where each person meets the
    turf. Head points would not do: they are not on the plane, so a single view
    cannot place them without assuming a height.
    """
    from scipy.optimize import linear_sum_assignment

    ga = ground_points(cam_a, feet_a, z_plane=z_plane)
    gb = ground_points(cam_b, feet_b, z_plane=z_plane)
    ok_a = (np.isfinite(ga).all(1) & (np.abs(ga[:, 0]) <= field_half_x)
            & (np.abs(ga[:, 1]) <= field_half_y))
    ok_b = (np.isfinite(gb).all(1) & (np.abs(gb[:, 0]) <= field_half_x)
            & (np.abs(gb[:, 1]) <= field_half_y))
    ia = np.flatnonzero(ok_a)
    ib = np.flatnonzero(ok_b)
    if not len(ia) or not len(ib):
        return 0, np.zeros((0, 2))
    cost = np.linalg.norm(ga[ia, None] - gb[None, ib], axis=2)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < max_gap_m
    placed = 0.5 * (ga[ia[r[keep]]] + gb[ib[c[keep]]])
    return int(keep.sum()), placed


def choose_pair(cands_a, cands_b, feet_a, feet_b, *, min_matched=MIN_MATCHED,
                **kwargs):
    """``(cam_a, cam_b, matched, info)`` -- the pair that reconciles the most.

    ``cands_*`` are lists of ``{frame: (K, R, t)}`` camera sets for one view;
    ``feet_*`` map frame to bottom-centre detections in that view. Frames are
    paired by index, so both views must be sampled at the same moments.
    """
    frames = sorted(set(feet_a) & set(feet_b))
    if not frames:
        raise CalibrationError("the two views share no sampled frames.")

    best = None
    for i, cam_a in enumerate(cands_a):
        for j, cam_b in enumerate(cands_b):
            per_frame = []
            for f in frames:
                if not len(feet_a[f]) or not len(feet_b[f]):
                    continue
                ka = min(cam_a, key=lambda k: abs(k - f))
                kb = min(cam_b, key=lambda k: abs(k - f))
                n, _placed = match_count(cam_a[ka], feet_a[f],
                                         cam_b[kb], feet_b[f], **kwargs)
                per_frame.append(n)
            if not per_frame:
                continue
            score = float(np.median(per_frame))
            _LOG.info("pair (%d, %d): %.1f players reconciled", i, j, score)
            if best is None or score > best[2]:
                best = (cam_a, cam_b, score, {"a": i, "b": j})
    if best is None:
        raise CalibrationError("no camera pair could be scored.")
    if best[2] < min_matched:
        raise CalibrationError(
            f"the best camera pair reconciles only {best[2]:.0f} players, "
            f"under {min_matched}. The two views are describing different "
            "worlds; neither camera can be trusted for reconstruction.")
    _LOG.info("chose pair %s reconciling %.1f players", best[3], best[2])
    return best
