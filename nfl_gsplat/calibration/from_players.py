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

WHAT WORKS INSTEAD. The sideline camera IS solid -- three independent frame
samples, at two different detector settings, agreed on (30, -95, 47) with
players at plausible heights. So use it. Players stand on the turf, so the
sideline view says where each of them is in the world; the endzone view sees the
same people. That is a set of world-to-image correspondences on a PLANE, which
is a homography, which is machinery this package already has.

And the correspondences are better conditioned than paint ever was: twenty-two
points spread across the field, rather than a pencil of parallel lines that
carries no scale. The thing that made paint ambiguous is exactly what players
fix.

WHAT IS UNKNOWN, AND HOW IT IS SEARCHED. Which endzone detection is which
sideline player is not known. Rather than search that combinatorially, the solve
is seeded from where a broadcast endzone camera can physically be -- behind an
end zone, on the long axis -- and alternates matching with re-fitting, keeping
whichever seed reconciles the most players. A seed that is wrong does not
converge to a plausible camera; it reconciles nothing, and says so.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.decompose_homography import homography_to_krt
from nfl_gsplat.calibration.joint_views import ground_points
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Where a broadcast endzone camera can be: behind an end zone, on or near the
# field's long axis, high enough to see over the players. Deliberately coarse --
# these only have to get the matching started.
SEED_X_M: tuple[float, ...] = (-95.0, -75.0, -60.0, 60.0, 75.0, 95.0)
SEED_Z_M: tuple[float, ...] = (12.0, 25.0, 40.0)
SEED_FOV_DEG: tuple[float, ...] = (20.0, 35.0, 50.0)

# How close a predicted player must land to a detection to be called that
# player, in pixels. Starts loose so a rough seed can find anything at all, and
# tightens as the fit improves.
GATE_START_PX: float = 220.0
GATE_FINAL_PX: float = 45.0
ICP_ITERS: int = 12

# Below this many reconciled players the answer is not believed at all.
MIN_INLIERS: int = 8

FIELD_HALF_X_M: float = 56.0
FIELD_HALF_Y_M: float = 26.0

# What a broadcast mount can physically be. Generous -- these only have to reject
# a degenerate fit, not adjudicate between plausible cameras.
MOUNT_HEIGHT_M: tuple[float, float] = (3.0, 90.0)
MAX_MOUNT_RANGE_M: float = 300.0


def seed_homography(centre, fov_deg: float, width: int, height: int,
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
    f = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
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


def solve_second_view(cam_known, feet_known, feet_unknown, width: int,
                      height: int, *, seeds=None, min_inliers: int = MIN_INLIERS):
    """``(K, R, t, inliers)`` for the second camera, from the players alone.

    ``cam_known`` is the solved camera of the first view; ``feet_*`` map frame to
    bottom-centre person detections in each view. Frames are paired by index.
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

    if seeds is None:
        seeds = [(x, 0.0, z, fov) for x in SEED_X_M for z in SEED_Z_M
                 for fov in SEED_FOV_DEG]
    results = []
    for x, y, z, fov in seeds:
        H0 = seed_homography((x, y, z), fov, width, height)
        if H0 is None:
            continue
        H, n = fit_from_seed(H0, world_by_frame, feet_unknown)
        if n >= min_inliers:
            results.append((n, (x, y, z, fov), H))
    results.sort(key=lambda r: -r[0])

    # Most inliers first, but a homography that cannot BE a camera does not win
    # on count. A wrong seed can accumulate matches on a degenerate fit -- one
    # did, and the decomposition raised rather than returning nonsense, which is
    # the failure working correctly. Walk down until one is physically possible.
    for n, seed, H in results:
        try:
            K, R, t = homography_to_krt(H, width=width, height=height)
        except (ValueError, np.linalg.LinAlgError):
            _LOG.info("seed %s: %d inliers but no camera could exist",
                      np.round(seed, 1), n)
            continue
        centre = -R.T @ t
        if not np.all(np.isfinite(centre)):
            continue
        if not (MOUNT_HEIGHT_M[0] <= centre[2] <= MOUNT_HEIGHT_M[1]):
            continue
        if np.linalg.norm(centre[:2]) > MAX_MOUNT_RANGE_M:
            continue
        _LOG.info("second view from players: %d inliers, seed %s, centre %s",
                  n, np.round(seed, 1), np.round(centre, 1))
        return K, R, t, n

    got = results[0][0] if results else 0
    raise CalibrationError(
        f"the players could not calibrate the second view: {len(results)} "
        f"seeds reached {min_inliers} inliers (best {got}), but none implied a "
        "camera that could exist.")
