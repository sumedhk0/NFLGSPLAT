"""Calibrate the second camera from the first one, using the players.

WHY THIS EXISTS. The endzone view does not calibrate from paint, and the
measurement is not close. Across three frame samples times four detector
settings times two orientations -- twenty-four solves of one production endzone
clip -- exactly two distinct cameras came out, both from the same sample, and
NEITHER put a single player at a believable height. Every pairing of them with
every sideline candidate reconciled zero players.

The reason is geometric, not a tuning failure, and it is visible in the frame
itself (look at one). The endzone camera is high, steep, and TIGHT: the two
hash rows, 5.64 m apart, sit 156 px/m apart, so the frame shows about twelve
metres of field width and twenty of length, and it pans and zooms to follow
the play. The hash marks the paint labeller needs are two receding strings of
dots rather than rows, and the far half of the field is a few dozen pixels.

WHAT WORKS INSTEAD. The sideline camera is solved and verified by the players
themselves (1.85 m, on the turf). Players stand on the turf, so the sideline
says where each of them is in the world; the endzone sees the same people;
that is a set of world-to-image correspondences on a plane. Twenty spread
points carry scale where a pencil of parallel lines does not.

WHAT WAS TRIED AND MEASURED, IN ORDER.

  1. A free homography, ICP-refined from a grid of lenses. Converged from all
     eighteen seeds to a 119 degree camera 5 m above the turf that reconciled
     5 of 21. An 8-DOF fit on a 17 x 28 m patch of noisy feet is
     ill-conditioned; RANSAC picks any of a large family.
  2. Fixed K from the box ruler (a 1.85 m player at range d is f*1.85/d px
     tall; the ruler that settled the sideline), pose by PnP. Every seed
     returned nothing: the seed AIMED at the sideline's formation centroid,
     and the endzone's tight frame is centred elsewhere -- under f ~ 9000 a
     220 px gate is 1.4 m on the turf, so nothing matched. Then, with an aim
     search, PnP RANSAC never moved the camera: its 15 px threshold is 0.1 m
     at that focal length. Both fixed by scaling every gate by f/d, i.e.
     working in metres on the turf.
  3. Per frame, an aim search on the ground, then PnP-ICP with the lens fixed
     and gates shrinking in metres. On the real play: 12 of 21 players
     reconciled per frame, the two views 0.81 m apart on the same player at
     the median. But on NOISELESS synthetic data the per-frame centres
     wandered by tens of metres and two frames in eight failed outright, with
     either PnP solver (ITERATIVE from a guess, or the planar IPPE: worse,
     160-1240 px plane-map error). That is not a solver problem. Behind a
     10000 px lens the view of a ten-player patch is nearly affine, and under
     weak perspective the camera's DISTANCE is not observable from points on
     a plane. Anything that tries to recover t per frame returns noise.
  4. The tripod solve (shared centre, per-frame R and f) on those matches.
     Reconciled one or two more players but placed them less accurately
     (gap 1.01 vs 0.81 m), kept fewer frames, and ran its centre out to
     114-144 m along the axis -- the same unobservable distance. Not adopted.

THIS VERSION fits only what the feet can determine. The mount CENTRE is a
prior: a coarse grid behind either end zone, held fixed. The LENS per frame is
the ruler applied to that frame's own boxes, which also tracks the zoom. The
ROTATION per frame is solved -- three parameters from eight to twenty points,
well conditioned even when perspective is weak -- after an aim search on the
turf, with gates shrinking in metres. Where exactly on the f/d family the
mount sits barely changes the plane map for the visible patch, which is why a
coarse grid is enough for placing players, and the ruler fixes f/d, which is
what makes the implied heights come out right.

HOW A RESULT IS BELIEVED. A mount must reconcile at least MIN_RECONCILED
players per frame at the median IN METRES (joint_views' ground-cloud test, the
only cross-view test that works for near-perpendicular cameras) and must put
the players it sees at a believable height. Mounts are ranked by reconciled
count, then by the gap. Nothing here is a prior about where the camera is
except the coarse seed grid, and the grid covers both ends of the field.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.joint_views import MAX_GAP_M, ground_points
from nfl_gsplat.calibration.player_scale import EXPECTED_PLAYER_M, height_score
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

# Where a broadcast endzone camera can be: behind either end zone, on the
# field's long axis, from a low platform to the upper deck. Coarse on purpose --
# the aim search and PnP do the rest, and the true mount on the measured play
# (about 80 m out, 12-20 m up, 14 degree pitch) sat between grid points.
MOUNT_X_M: tuple[float, ...] = (45.0, 60.0, 75.0, 95.0)
MOUNT_Z_M: tuple[float, ...] = (20.0, 35.0, 50.0, 65.0)

# The box-implied focal is tried at these multiples: the boxes give the lens at
# the SEED range, which is itself a guess.
SEED_FOCAL_SCALES: tuple[float, ...] = (0.8, 1.0, 1.25)

# Aim search, in metres on the turf around the formation. The tight endzone
# frame is centred on the action, not on the formation's centroid. The step
# has to be well under the spacing between players (2-4 m): with a 3 m step
# the best aim was still 2.8 degrees off, which at 77 m is 3.8 m on the turf,
# and the first assignment paired everyone with a neighbour. Least squares
# then converged, faithfully, to that wrong pairing at 0.75 m residual.
AIM_COARSE_M = np.arange(-15.0, 15.1, 3.0)     # first pass over the turf
AIM_FINE_M = np.arange(-2.0, 2.1, 1.0)         # around each of the best coarse aims
AIM_REFINE_TOP: int = 5
AIM_GATE_M: float = 1.5
AIM_STARTS: int = 8

# ICP gates, in metres on the turf, converted to pixels by f/d per mount. In
# pixels they would be meaningless: 15 px is 1 m behind a 1300 px lens and
# 0.1 m behind a 9000 px one.
GATE_START_M: float = 1.5
GATE_FINAL_M: float = 0.6
ICP_ITERS: int = 6
ICP_ROUNDS: int = 2               # re-match after each refine at the same gate

# The gate for COUNTING a match at the end is adaptive: a multiple of the
# median residual of the fit, clipped. Measured on the real play, the two
# views disagree by about 1 m on the same player -- the sideline's own
# world-point error, not the endzone fit -- so 70% of players agree within
# 1.5 m and only 40% within 0.7 m. A fixed 0.7 m gate, tuned on noiseless
# synthetic data, refused every real frame. Noiseless data still gets 0.5 m.
NOISE_GATE_MULT: float = 1.5
NOISE_GATE_PROBE_M: float = 2.0
MATCH_GATE_M: tuple[float, float] = (0.5, 1.5)
MIN_MATCHES_PER_FRAME: int = 4

# A frame whose best fit explains fewer than this fraction of its detections
# (and of the first view's players its camera puts in frame) has settled into
# a wrong pairing; better no camera for that frame than a wrong one, since
# the median over frames decides the mount anyway. Not higher: on the real
# play the detections include referees, a skycam and sideline staff, and the
# noise floor costs another third, so a correct frame explains 14 of 22 --
# 0.6 rejected it. The stretched-world case this once guarded is now caught
# by the zero-filled median over frames.
MIN_MATCH_FRAC: float = 0.45

# A mount must reconcile this many players per frame, at the median, for its
# answer to be believed at all. The real play gives 12; a wrong mount gives a
# handful by accident.
MIN_RECONCILED: int = 6

# And it must fit at least this fraction of the shared frames. A stretched
# sideline world was accepted on the strength of ONE lucky frame in eight,
# because the median was taken over the frames that fitted; frames that did
# not fit now count as zero, and too few fitted frames is a refusal.
MIN_FRAMES_FRAC: float = 0.35     # late in a play the formation disperses

# Player-height cost above which a mount is not believed however many players
# it reconciles. Looser than calibrate_clip's 0.60 on purpose: on the real play
# EVERY mount scored 0.52-0.68 (the endzone's steep, tight view makes feet
# noisy), so the bar is not what separates mounts -- reconciliation is -- and
# it only has to reject a camera that makes people 5 m tall.
MAX_PLAYER_COST: float = 0.75

FIELD_HALF_X_M: float = 56.0
FIELD_HALF_Y_M: float = 26.0


def feet_of(boxes):
    """Bottom-centre of each person box: where they meet the turf."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])


def focal_from_boxes(boxes_by_frame, distance_m: float, *,
                     pitch_deg: float = 0.0,
                     expected_m: float = EXPECTED_PLAYER_M) -> float:
    """The focal length the boxes imply if the players are ``distance_m`` away.

    A person ``expected_m`` tall at range ``d`` is ``f * expected_m / d`` pixels
    tall when seen level; looking down at ``pitch_deg`` foreshortens a vertical
    by about cos(pitch), which matters for the endzone's steep view.
    """
    heights = [np.median(np.asarray(b, float).reshape(-1, 4)[:, 3]
                         - np.asarray(b, float).reshape(-1, 4)[:, 1])
               for b in boxes_by_frame.values() if len(b)]
    if not heights:
        raise CalibrationError("no person boxes to read a lens from.")
    scale = max(np.cos(np.radians(pitch_deg)), 0.2)
    return float(np.median(heights)) * distance_m / (expected_m * scale)


def look_at(centre, target):
    """World-to-camera rotation for a camera at ``centre`` looking at ``target``."""
    centre = np.asarray(centre, float)
    fwd = np.asarray(target, float) - centre
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return None
    fwd = fwd / n
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    n = np.linalg.norm(right)
    if n < 1e-9:
        return None
    right = right / n
    down = np.cross(fwd, right)
    return np.vstack([right, down, fwd])


def _project(K, R, t, xy):
    p = np.c_[np.asarray(xy, float), np.zeros(len(xy))] @ R.T + t
    q = p @ K.T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    uv[p[:, 2] <= 0] = np.nan          # behind the camera
    return uv


def _match(K, R, t, world_xy, uv, gate_px):
    """Pair predicted player positions with detections, one to one."""
    from scipy.optimize import linear_sum_assignment

    pred = _project(K, R, t, world_xy)
    ok = np.isfinite(pred).all(1)
    if ok.sum() == 0 or len(uv) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    idx = np.flatnonzero(ok)
    cost = np.linalg.norm(pred[idx, None] - np.asarray(uv, float)[None], axis=2)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < gate_px
    return world_xy[idx[r[keep]]], np.asarray(uv, float)[c[keep]]


def _refine_rotation(K, centre, R0, world_xy, uv, gate_px):
    """Rotation only: the centre is a prior and the lens comes from the boxes.

    Three parameters, a robust loss at the gate, so a stray match cannot
    drag the aim.
    """
    import cv2
    from scipy.optimize import least_squares

    centre = np.asarray(centre, float)
    uv = np.asarray(uv, float)
    r0 = cv2.Rodrigues(np.asarray(R0, float))[0].ravel()

    def resid(r):
        R = cv2.Rodrigues(r)[0]
        pred = _project(K, R, -R @ centre, world_xy)
        d = pred - uv
        d[~np.isfinite(d)] = 10.0 * gate_px
        return d.ravel()

    sol = least_squares(resid, r0, loss="soft_l1", f_scale=float(gate_px),
                        max_nfev=200)
    R = cv2.Rodrigues(sol.x)[0]
    return R, -R @ centre


def fit_frame(centre, K, world_xy, uv, formation, px_per_m):
    """``(R, t, n_matched)`` or None. See ``_fit_frame`` for the reasons."""
    out, _why = _fit_frame(centre, K, world_xy, uv, formation, px_per_m)
    return out


def _fit_frame(centre, K, world_xy, uv, formation, px_per_m):
    """One frame: aim search on the turf, then rotation-only ICP.

    Returns ``(R, t, n_matched)`` or None. The aim search is what lets a seed
    pointed at the wrong patch of turf find the players at all; the shrinking
    gates are what stop it counting coincidences once it has. ICP is run from
    the best few aims, not the single best: on synthetic data half the frames
    converged exactly from the top aim and the other half settled into a
    wrong pairing of neighbours that a second start escapes.
    """
    centre = np.asarray(centre, float)

    def score_aim(dx, dy):
        R = look_at(centre, [formation[0] + dx, formation[1] + dy, 0.0])
        if R is None:
            return None
        t = -R @ centre
        w, _u = _match(K, R, t, world_xy, uv, AIM_GATE_M * px_per_m)
        return (len(w), R, t, dx, dy)

    coarse = [a for a in (score_aim(dx, dy) for dx in AIM_COARSE_M
                          for dy in AIM_COARSE_M) if a is not None]
    coarse.sort(key=lambda a: -a[0])
    seen, aims = set(), []
    for _n, _R, _t, cx, cy in coarse[:AIM_REFINE_TOP]:
        for ddx in AIM_FINE_M:
            for ddy in AIM_FINE_M:
                key = (round(cx + ddx, 3), round(cy + ddy, 3))
                if key in seen:
                    continue
                seen.add(key)
                a = score_aim(*key)
                if a is not None:
                    aims.append(a[:3])
    aims.sort(key=lambda a: -a[0])
    best = None
    recall_failed = 0
    for n0, R, t in aims[:AIM_STARTS]:
        if n0 < MIN_MATCHES_PER_FRAME:
            break
        for gate_m in np.geomspace(GATE_START_M, GATE_FINAL_M, ICP_ITERS):
            for _round in range(ICP_ROUNDS):
                w, u = _match(K, R, t, world_xy, uv, gate_m * px_per_m)
                if len(w) < MIN_MATCHES_PER_FRAME:
                    break
                R, t = _refine_rotation(K, centre, R, w, u, gate_m * px_per_m)
        # The noise floor of THIS fit sets the counting gate.
        w, u = _match(K, R, t, world_xy, uv, NOISE_GATE_PROBE_M * px_per_m)
        if len(w) < MIN_MATCHES_PER_FRAME:
            continue
        floor = float(np.median(np.linalg.norm(_project(K, R, t, w) - u,
                                               axis=1))) / px_per_m
        gate_m = float(np.clip(NOISE_GATE_MULT * floor, *MATCH_GATE_M))
        w, u = _match(K, R, t, world_xy, uv, gate_m * px_per_m)
        if len(w) < MIN_MATCHES_PER_FRAME:
            continue
        # Recall as well as precision: under this camera, how many of the
        # first view's players should be in the frame? A fit that explains
        # the detections with a small subset of a world that puts far more
        # players in frame has matched the wrong world -- measured, a
        # threefold-stretched sideline still found six congruent points.
        pred = _project(K, R, t, world_xy)
        in_frame = int((np.isfinite(pred).all(1) & (pred[:, 0] >= 0)
                        & (pred[:, 0] < 2 * K[0, 2]) & (pred[:, 1] >= 0)
                        & (pred[:, 1] < 2 * K[1, 2])).sum())
        if len(w) < MIN_MATCH_FRAC * in_frame:
            recall_failed += 1
            continue
        resid = float(np.median(np.linalg.norm(
            _project(K, R, t, w) - u, axis=1)))
        key = (len(w), -resid)
        if best is None or key > best[0]:
            best = (key, R, t)
        if len(w) == len(uv):
            break                       # every detection explained; done
    if best is None:
        return None, ("explains too few of the players in frame"
                      if recall_failed else "no start converged")
    if best[0][0] < MIN_MATCH_FRAC * len(uv):
        return None, "explains too few detections"
    (n, _r), R, t = best
    return (R, t, int(n)), "ok"


def _agreement(cams_known, feet_known, cams_unknown, feet_unknown, *,
               gap_m: float = MAX_GAP_M):
    """Per-frame reconciled counts and the metre gaps between the two views."""
    from scipy.optimize import linear_sum_assignment

    counts, gaps = [], []
    for f, cam_u in cams_unknown.items():
        if f not in feet_known or f not in feet_unknown:
            continue
        ga = ground_points(cams_known[min(cams_known, key=lambda k: abs(k - f))],
                           feet_known[f])
        gb = ground_points(cam_u, feet_unknown[f])
        oka = (np.isfinite(ga).all(1) & (np.abs(ga[:, 0]) <= FIELD_HALF_X_M)
               & (np.abs(ga[:, 1]) <= FIELD_HALF_Y_M))
        okb = (np.isfinite(gb).all(1) & (np.abs(gb[:, 0]) <= FIELD_HALF_X_M)
               & (np.abs(gb[:, 1]) <= FIELD_HALF_Y_M))
        if not oka.any() or not okb.any():
            counts.append(0)
            continue
        cost = np.linalg.norm(ga[oka][:, None] - gb[okb][None], axis=2)
        r, c = linear_sum_assignment(cost)
        keep = cost[r, c] < gap_m
        counts.append(int(keep.sum()))
        gaps.extend(cost[r, c][keep].tolist())
    return counts, np.asarray(gaps)


def _player_cost(cams, boxes_by_frame) -> float:
    costs = []
    for f, (K, R, t) in cams.items():
        if f in boxes_by_frame and len(boxes_by_frame[f]):
            cost, _n, _median = height_score(K, R, t, boxes_by_frame[f])
            if np.isfinite(cost):
                costs.append(cost)
    return float(np.median(costs)) if costs else float("inf")


def solve_second_view(cam_known, feet_known, boxes_unknown, width: int,
                      height: int, *, mounts=None,
                      focal_scales=SEED_FOCAL_SCALES,
                      min_reconciled: int = MIN_RECONCILED,
                      max_player_cost: float = MAX_PLAYER_COST):
    """Per-frame cameras for the second view, from the players alone.

    Returns ``(cams, info)``: ``cams`` maps frame to ``(K, R, t)`` -- per frame,
    because the endzone pans -- and ``info`` reports the mount, lens, players
    reconciled per frame, the metre gap between the views, and player height.

    ``cam_known`` maps frame to the solved camera of the first view;
    ``feet_known`` maps frame to bottom-centre detections in it;
    ``boxes_unknown`` maps frame to full person boxes in the view to solve --
    boxes, because their height fixes the lens and gates the answer. Frames
    are paired by index.

    ``mounts`` overrides the seed grid as ``[(x, z), ...]``; y is 0.
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
    frames = sorted(set(world_by_frame) & set(feet_unknown))
    if not frames:
        raise CalibrationError("the two views share no frames with players.")
    formation = np.concatenate([world_by_frame[f] for f in frames]).mean(0)
    target = np.array([formation[0], formation[1], 0.0])

    if mounts is None:
        mounts = [(sx * x, z) for sx in (-1.0, 1.0) for x in MOUNT_X_M
                  for z in MOUNT_Z_M]

    results = []
    for x, z in mounts:
        centre = np.array([x, 0.0, z], float)
        d = float(np.linalg.norm(centre - target))
        pitch = float(np.degrees(np.arctan2(z, np.linalg.norm(centre[:2]
                                                             - formation))))
        best = None
        for s in focal_scales:
            cams, why = {}, {}
            for fr in frames:
                # The lens per FRAME, from that frame's boxes: the endzone zooms.
                f = s * focal_from_boxes({fr: boxes_unknown[fr]}, d,
                                         pitch_deg=pitch)
                K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0],
                              [0.0, 0.0, 1.0]])
                out, reason = _fit_frame(centre, K, world_by_frame[fr],
                                         feet_unknown[fr], formation, f / d)
                why[reason] = why.get(reason, 0) + 1
                if out is not None:
                    cams[fr] = (K, out[0], out[1])
            _LOG.info("mount (%.0f, %.0f) x%.2f: %s", x, z, s, why)
            f = float(np.median([c[0][0, 0] for c in cams.values()])) if cams else 0.0
            if len(cams) < MIN_FRAMES_FRAC * len(frames):
                continue
            counts, gaps = _agreement(cam_known, feet_known, cams, feet_unknown)
            counts = counts + [0] * (len(frames) - len(counts))   # unfitted = 0
            score = (float(np.median(counts)),
                     -float(np.median(gaps)) if len(gaps) else -np.inf)
            if best is None or score > best[0]:
                best = (score, f, cams, counts, gaps)
        if best is None:
            _LOG.info("mount (%.0f, %.0f): fewer than half the frames fitted",
                      x, z)
            continue
        score, f, cams, counts, gaps = best
        cost = _player_cost(cams, boxes_unknown)
        _LOG.info("mount (%.0f, %.0f): lens %.0f px, %d frames, reconciles "
                  "%.0f/frame, gap %.2f m, player cost %.2f", x, z, f,
                  len(cams), score[0], -score[1], cost)
        results.append((score, cost, (x, z), f, cams, counts, gaps))

    if not results:
        raise CalibrationError(
            "the players could not calibrate the second view: no mount seed "
            "fitted a single frame.")
    results.sort(key=lambda r: r[0], reverse=True)
    for score, cost, mount, f, cams, counts, gaps in results:
        if score[0] < min_reconciled:
            break
        if not np.isfinite(cost) or cost > max_player_cost:
            _LOG.info("mount %s reconciles %.0f but player cost %.2f; skipped",
                      mount, score[0], cost)
            continue
        info = {"mount": mount, "focal": f, "frames": len(cams),
                "reconciled": score[0], "gap_m": -score[1],
                "player_cost": cost,
                "centre": np.array([mount[0], 0.0, mount[1]]),
                "counts": counts}
        _LOG.info("second view from players: mount %s, %d frames, reconciles "
                  "%.0f/frame, gap %.2f m, centre %s", mount, len(cams),
                  score[0], -score[1], np.round(info["centre"], 1))
        return cams, info

    best = results[0]
    raise CalibrationError(
        f"the players could not calibrate the second view: the best mount "
        f"{best[2]} reconciles {best[0][0]:.0f} players per frame (need "
        f"{min_reconciled}) with player cost {best[1]:.2f} (bar "
        f"{max_player_cost}).")
