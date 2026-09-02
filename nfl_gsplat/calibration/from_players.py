"""Calibrate the second camera from the first one, using the players.

WHY THIS EXISTS. The endzone view does not calibrate from paint, and the
measurement is not close. Across three frame samples times four detector
settings times two orientations -- twenty-four solves of one production endzone
clip -- exactly two distinct cameras came out, both from the same sample, and
NEITHER put a single player at a believable height: one sat 192 m down the
field's long axis, the other out past the sideline on the wrong axis entirely.
Every pairing of them with every sideline candidate reconciled zero players.

The reason is geometric, not a tuning failure. Looking down the length of the
field, the lines the labeller needs are the wrong way round: the hash marks that
fit_rows must group into rows are a pair of receding, nearly vertical strings of
dots, and the far half of the field compresses into a few dozen pixels. Turning
the frame a quarter turn swaps the families back (see orientation.py) but does
not restore the resolution that perspective threw away.

WHAT WORKS INSTEAD. The sideline camera is solved and verified by the players
themselves (1.85 m, on the turf). So use it. Players stand on the turf, so the
sideline view says where each of them is in the world; the endzone view sees the
same people. That is a set of world-to-image correspondences on a PLANE, which
is a homography, which is machinery this package already has -- and better
conditioned than paint ever was, because twenty spread points carry scale where
a pencil of parallel lines does not.

HOW THE SEARCH IS SEEDED, AND WHY BY THE BOXES. Which endzone detection is which
sideline player is unknown, so the fit starts from a guess and alternates
matching with re-fitting. The first version guessed the lens from a grid of
three fields of view and, on the real play, converged to a 119 degree camera
five metres above the turf that reconciled five players. The lens does not have
to be guessed: the person boxes are a ruler. A 1.85 m player at range d is
f * 1.85 / d pixels tall, so the median box height at a candidate mount fixes
the focal length directly. That ruler is what just settled the sideline camera
(140 px boxes at 100 m mean a 12 degree lens, whatever the paint prefers), and
it is used here for the same reason.

HOW A RESULT IS BELIEVED. Inlier count alone was not enough: a degenerate
homography can collect matches and then imply a camera no mount could be. So a
result must (a) reconcile at least MIN_INLIERS players, (b) decompose into a
camera at a physically possible height and range, and (c) put the players it
sees at a plausible height -- the same player_scale check that ranks sideline
candidates. Candidates are walked in order of inliers until one passes all
three; if none does, it says so rather than returning the least bad.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.decompose_homography import homography_to_krt
from nfl_gsplat.calibration.joint_views import ground_points
from nfl_gsplat.calibration.player_scale import EXPECTED_PLAYER_M, height_score
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Where a broadcast endzone camera can be: behind an end zone, on or near the
# field's long axis, high enough to see over the players. Deliberately coarse --
# these only have to get the matching started; the lens comes from the boxes.
SEED_X_M: tuple[float, ...] = (-95.0, -75.0, -60.0, 60.0, 75.0, 95.0)
SEED_Z_M: tuple[float, ...] = (12.0, 25.0, 40.0)

# The box-implied focal is tried at these multiples, because the boxes give the
# lens at the SEED range, which is itself a guess.
SEED_FOCAL_SCALES: tuple[float, ...] = (0.7, 1.0, 1.4)

# How close a predicted player must land to a detection to be called that
# player, in pixels. Starts loose so a rough seed can find anything at all, and
# tightens as the fit improves.
GATE_START_PX: float = 220.0
GATE_FINAL_PX: float = 45.0
ICP_ITERS: int = 12

# Below this many reconciled players the answer is not believed at all.
MIN_INLIERS: int = 8

# Player-height cost above which a camera is not believed however many players
# it reconciled. Same bar as calibrate_clip.MAX_PLAYER_COST, and for the same
# reason: it is the one check a wrong lens cannot pass.
MAX_PLAYER_COST: float = 0.60

FIELD_HALF_X_M: float = 56.0
FIELD_HALF_Y_M: float = 26.0

# What a broadcast mount can physically be. Generous -- these only have to reject
# a degenerate fit, not adjudicate between plausible cameras.
MOUNT_HEIGHT_M: tuple[float, float] = (3.0, 90.0)
MAX_MOUNT_RANGE_M: float = 300.0


def feet_of(boxes):
    """Bottom-centre of each person box: where they meet the turf."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])


def focal_from_boxes(boxes_by_frame, distance_m: float,
                     expected_m: float = EXPECTED_PLAYER_M) -> float:
    """The focal length the boxes imply if the players are ``distance_m`` away.

    A person ``expected_m`` tall at range ``d`` is ``f * expected_m / d`` pixels
    tall, so the median box height fixes ``f`` at any assumed range.
    """
    heights = [np.median(np.asarray(b, float).reshape(-1, 4)[:, 3]
                         - np.asarray(b, float).reshape(-1, 4)[:, 1])
               for b in boxes_by_frame.values() if len(b)]
    if not heights:
        raise CalibrationError("no person boxes to read a lens from.")
    return float(np.median(heights)) * distance_m / expected_m


def seed_homography(centre, focal_px: float, width: int, height: int,
                    target=(0.0, 0.0, 0.0)):
    """Turf-to-image homography for a camera at ``centre`` looking at ``target``."""
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    if n < 1e-9:
        return None
    right = right / n
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    K = np.array([[focal_px, 0.0, width / 2.0],
                  [0.0, focal_px, height / 2.0],
                  [0.0, 0.0, 1.0]])
    P = K @ np.column_stack([R, -R @ centre])
    return P[:, [0, 1, 3]]


def _project(H, xy):
    q = np.column_stack([np.asarray(xy, float), np.ones(len(xy))]) @ np.asarray(H, float).T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = q[:, :2] / q[:, 2:3]
    out[q[:, 2] <= 0] = np.nan        # behind the camera
    return out


def _match(H, world_xy, uv, gate_px):
    """Pair predicted player positions with detections, one to one."""
    from scipy.optimize import linear_sum_assignment

    pred = _project(H, world_xy)
    ok = np.isfinite(pred).all(1)
    if ok.sum() == 0 or len(uv) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    idx = np.flatnonzero(ok)
    cost = np.linalg.norm(pred[idx, None] - np.asarray(uv, float)[None], axis=2)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < gate_px
    return world_xy[idx[r[keep]]], np.asarray(uv, float)[c[keep]]


def fit_from_seed(H, world_by_frame, uv_by_frame, *, iters: int = ICP_ITERS,
                  gate_start: float = GATE_START_PX,
                  gate_final: float = GATE_FINAL_PX):
    """Alternate matching and re-fitting from a seed. Returns ``(H, inliers)``.

    The gate shrinks geometrically: a rough seed needs a loose gate to match
    anything at all, and a converged fit needs a tight one to stop counting
    coincidences.
    """
    import cv2

    gates = np.geomspace(gate_start, gate_final, iters)
    best = (H, 0)
    for gate in gates:
        src, dst = [], []
        for f, world_xy in world_by_frame.items():
            if f not in uv_by_frame or not len(world_xy) or not len(uv_by_frame[f]):
                continue
            w, u = _match(H, world_xy, uv_by_frame[f], gate)
            if len(w):
                src.append(w)
                dst.append(u)
        if not src:
            break
        src = np.concatenate(src)
        dst = np.concatenate(dst)
        if len(src) < 4:
            break
        H_new, mask = cv2.findHomography(src, dst, cv2.RANSAC, 12.0)
        if H_new is None:
            break
        H = H_new
        n = int(mask.sum()) if mask is not None else len(src)
        if n > best[1]:
            best = (H, n)
    return best


def _player_cost(K, R, t, boxes_by_frame) -> float:
    costs = []
    for boxes in boxes_by_frame.values():
        if len(boxes):
            cost, _n, _median = height_score(K, R, t, boxes)
            if np.isfinite(cost):
                costs.append(cost)
    return float(np.median(costs)) if costs else float("inf")


def solve_second_view(cam_known, feet_known, boxes_unknown, width: int,
                      height: int, *, seeds=None, min_inliers: int = MIN_INLIERS,
                      max_player_cost: float = MAX_PLAYER_COST):
    """``(K, R, t, inliers)`` for the second camera, from the players alone.

    ``cam_known`` is the solved camera of the first view; ``feet_known`` maps
    frame to bottom-centre detections in it; ``boxes_unknown`` maps frame to
    full person boxes in the view to solve -- boxes, not feet, because their
    height seeds the lens and gates the answer. Frames are paired by index.

    ``seeds`` overrides the search as ``[(centre_xyz, focal_px), ...]``.
    """
    world_by_frame = {}
    for f, uv in feet_known.items():
        g = ground_points(cam_known[min(cam_known, key=lambda k: abs(k - f))], uv)
        ok = (np.isfinite(g).all(1) & (np.abs(g[:, 0]) <= FIELD_HALF_X_M)
              & (np.abs(g[:, 1]) <= FIELD_HALF_Y_M))
        if ok.sum() >= 4:
            world_by_frame[f] = g[ok]
    if not world_by_frame:
        raise CalibrationError(
            "the known camera placed no players on the field, so it cannot "
            "calibrate the other view.")
    feet_unknown = {f: feet_of(b) for f, b in boxes_unknown.items() if len(b)}
    if not feet_unknown:
        raise CalibrationError("no person boxes in the view to solve.")

    # Aim at the formation, not the field's origin: the play may be anywhere.
    formation = np.concatenate(list(world_by_frame.values())).mean(0)
    target = np.array([formation[0], formation[1], 0.0])

    if seeds is None:
        seeds = []
        for x in SEED_X_M:
            for z in SEED_Z_M:
                centre = np.array([x, 0.0, z])
                f0 = focal_from_boxes(boxes_unknown,
                                      float(np.linalg.norm(centre - target)))
                seeds += [(centre, f0 * s) for s in SEED_FOCAL_SCALES]

    results = []
    for centre, focal in seeds:
        H0 = seed_homography(centre, focal, width, height, target=target)
        if H0 is None:
            continue
        H, n = fit_from_seed(H0, world_by_frame, feet_unknown)
        if n >= min_inliers:
            results.append((n, (tuple(np.round(centre, 1)), round(focal)), H))
    results.sort(key=lambda r: -r[0])

    # Most inliers first, but a result must also be a camera that could exist
    # AND make the players it sees a believable height. A wrong seed can collect
    # matches on a degenerate fit; one did, and decomposition raised. Another
    # converged to a 119 degree lens 5 m up that reconciled five players and
    # passed the mount check, which is why the height check is here.
    best_cost, best_n = float("inf"), results[0][0] if results else 0
    for n, seed, H in results:
        try:
            K, R, t = homography_to_krt(H, width=width, height=height)
        except (ValueError, np.linalg.LinAlgError):
            _LOG.info("seed %s: %d inliers but no camera could exist", seed, n)
            continue
        centre = -R.T @ t
        if not np.all(np.isfinite(centre)):
            continue
        if not (MOUNT_HEIGHT_M[0] <= centre[2] <= MOUNT_HEIGHT_M[1]):
            continue
        if np.linalg.norm(centre[:2]) > MAX_MOUNT_RANGE_M:
            continue
        cost = _player_cost(K, R, t, boxes_unknown)
        if not np.isfinite(cost) or cost > max_player_cost:
            _LOG.info("seed %s: %d inliers, centre %s, but player cost %.2f",
                      seed, n, np.round(centre, 1), cost)
            best_cost = min(best_cost, cost)
            continue
        _LOG.info("second view from players: %d inliers, player cost %.2f, "
                  "seed %s, centre %s", n, cost, seed, np.round(centre, 1))
        return K, R, t, n

    raise CalibrationError(
        f"the players could not calibrate the second view: {len(results)} "
        f"seeds reached {min_inliers} inliers (best {best_n}), but none "
        f"implied a camera that could exist with players a believable height "
        f"(best player cost {best_cost:.2f}, bar {max_player_cost}).")
