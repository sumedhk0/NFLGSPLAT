"""Per-frame refinement and global bundle adjustment for a tripod camera.

The mosaic route (:mod:`nfl_gsplat.calibration.endzone_mosaic`) solves ONE
camera, at the reference frame, and every other frame inherits it through a
homography onto that reference. The inheritance decays: registering a frame far
down the play against the reference needs shared visible field, and a broadcast
camera pans a long way, so the overlap thins, the homography becomes poorly
conditioned, and the error lands in the rotation. Measured on SEA@AZ play_001,
propagated poses sat 26-55 px off -- roughly a yard on the ground -- while the
reference itself reprojects at 2.71 px.

This module gives frames their own solve instead.

The whole thing rests on the camera being on a TRIPOD. Two consequences:

* The centre never moves, so a frame has 4 unknowns (focal + rotation), not 7 --
  and a pencil of yard lines supplies 5 independent constraints, enough to pin
  them.
* The map between any two frames is exactly ``K_s R_s R_t^T K_t^-1``, with no
  dependence on the scene, since without translation there is no parallax. That
  makes a measured frame-to-frame homography a direct constraint on a pair of
  cameras.

Two hard-won rules are encoded here, because both produced confident wrong
answers before they were understood:

* An association is only usable if it is UNAMBIGUOUS. Yard lines repeat every
  YARD_LINE_SPACING_M, so a slightly wrong pose matches the neighbouring line
  just as happily, and a solve built on that is self-consistent and wrong. An
  early sweep did exactly this and reported 1.59 px while sitting a whole line
  out.
* A verification metric must carry a CONTROL. Four separate per-frame metrics
  looked plausible and were measuring nothing -- they reported ~20 px on a
  camera independently known to be correct to 2.7 px. :func:`verify_frame`
  therefore also scores a deliberately wrong model, and refuses to certify a
  frame whose metric cannot tell the two apart.
"""
from __future__ import annotations

import cv2
import numpy as np

from nfl_gsplat.calibration.field_landmarks import (HALF_WIDTH_M,
                                                    YARD_LINE_SPACING_M)
from nfl_gsplat.calibration.field_model_fit import detect_accumulated_lines
from nfl_gsplat.errors import CalibrationError
from nfl_gsplat.utils.logging import get_logger

_LOG = get_logger(__name__)

#: Sample count along a projected line. Fixed, because least_squares needs a
#: fixed-length residual and visibility changes as the camera moves.
#:
#: Sized so the samples stay DENSE across the full field width. Visibility is
#: tested by counting on-sensor samples, so density and span are coupled: at 30
#: samples, widening the span from 18 m to 49 m quietly tripled how much of a
#: line had to be visible to qualify (8 samples went from ~5 m of line to ~13 m)
#: and cost more frames than the wider span gained.
_N_SAMPLES = 80
#: Cost charged to a sample that projects off-sensor or behind the camera.
#: NOT zero: zeroing them let the optimiser push every point out of frame for a
#: free residual of 0 and destroy the camera (measured: focal ran away, error
#: became nan).
_OFF_SENSOR_COST = 200.0


def detect_frame_lines(paint, *, min_len_frac: float = 0.15,
                       merge_tol_px: float = 14.0) -> list[np.ndarray]:
    """Merged field lines in ONE frame, as normalised homogeneous lines.

    Delegates to detect_accumulated_lines rather than running a bare Hough pass.
    Raw Hough returns fragments -- thick paint yields both edges, every hash mark
    contributes -- which measured 60 to 138 "lines" per frame where 11 exist.
    Nearest-line association against that many candidates is meaningless, and it
    produced camera shifts of up to 445 px, a whole yard-line swap.

    ``min_len_frac`` is below the mosaic default because a single frame's lines
    are broken by players and shadows, so they run shorter than in an
    accumulated mosaic.
    """
    mask = np.asarray(paint)
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    if mask.max() > 1.5:
        mask = mask / 255.0
    # detect_accumulated_lines refuses when two parallel families tie in size,
    # since it cannot tell which is the yard-line pencil. On a single frame that
    # tie is an artefact of the length threshold, not a real ambiguity: measured
    # on play_001, frames 450/690/750 tied at 10-11 lines and resolved to 13-18
    # as soon as shorter segments were admitted. Swallowing the error instead
    # silently discarded those frames.
    segs = None
    for frac in (min_len_frac, 0.10, 0.07):
        if frac > min_len_frac:
            continue
        try:
            segs = detect_accumulated_lines(mask, vote_thresh=0.5,
                                            min_len_frac=frac,
                                            merge_tol_px=merge_tol_px)
            break
        except CalibrationError:
            continue
    if segs is None:
        return []
    out = []
    for sg in segs:
        ln = np.cross([sg.p0[0], sg.p0[1], 1.0], [sg.p1[0], sg.p1[1], 1.0])
        n = float(np.hypot(ln[0], ln[1]))
        if n > 1e-9:
            out.append(ln / n)
    return out


def project_line(focal, rot, centre, world_x, *, cx, cy,
                 y_span=(-HALF_WIDTH_M, HALF_WIDTH_M),
                 width=1920, height=1080):
    """Sample a world yard line into the image. Returns ``(uv, visible)``.

    ``y_span`` covers the FULL field width by default. A narrow window centred on
    midfield looks reasonable and quietly breaks whenever the camera pans across
    the field to follow a play: every sample lands off-sensor, the line reports
    "not visible", and there is nothing left to associate. Measured on play_001,
    a +-8 m window left frames 570 and 600 with no visible world lines at all
    despite ten clean detections in each, and cut frames 540 and 720 to two
    matches. Sampling wider costs only resolution along the line, and a line is
    fitted from its samples rather than measured at their endpoints.
    """
    ys = np.linspace(y_span[0], y_span[1], _N_SAMPLES)
    t = -rot @ centre
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    pts = np.column_stack([np.full_like(ys, float(world_x)), ys, np.zeros_like(ys)])
    q = (k @ (rot @ pts.T + t[:, None])).T
    ok = q[:, 2] > 1e-6
    uv = np.zeros((_N_SAMPLES, 2))
    uv[ok] = q[ok, :2] / q[ok, 2:3]
    vis = ok & (uv[:, 0] > 0) & (uv[:, 0] < width) \
        & (uv[:, 1] > 0) & (uv[:, 1] < height)
    return uv, vis


def associate(focal, rot, centre, world_xs, dets, *, cx, cy, tol_px=70.0,
              require_unambiguous=True, ratio=2.5, margin=10.0):
    """Match each world yard line to the detected line nearest the guess.

    With ``require_unambiguous``, a match is kept only when the runner-up is
    clearly further away. Yard lines repeat, so an ambiguous match is not a
    weak constraint -- it is a confident pull toward the WRONG line, and it
    survives every downstream check because the result stays self-consistent.
    """
    pairs = []
    for world_x in world_xs:
        uv, vis = project_line(focal, rot, centre, world_x, cx=cx, cy=cy)
        if vis.sum() < 8:
            continue
        homo = np.column_stack([uv[vis], np.ones(int(vis.sum()))])
        scored = sorted((float(np.median(np.abs(homo @ ln))), i)
                        for i, ln in enumerate(dets))
        if not scored or scored[0][0] >= tol_px:
            continue
        if require_unambiguous and len(scored) > 1 \
                and scored[1][0] <= ratio * scored[0][0] + margin:
            continue
        pairs.append((float(world_x), dets[scored[0][1]]))
    return pairs


def refine_frame(focal, rot, centre, pairs, *, cx, cy, max_shift_px=None,
                 anchor=None):
    """Re-solve one frame's focal and rotation against its own detected lines.

    ``max_shift_px`` rejects a result that moved further than that from
    ``anchor`` (a globally anchored camera). The two error scales separate
    cleanly: propagated drift measured 26-55 px, while adjacent yard lines sit
    ~109 px apart in this view, so a larger move is a line swap rather than a
    correction. Returns ``None`` when the solve should not be trusted.
    """
    from scipy.optimize import least_squares

    if len(pairs) < 3:
        return None
    rvec, _ = cv2.Rodrigues(rot)
    p0 = np.concatenate([[float(focal)], rvec.ravel()])

    def residual(p):
        rot_p, _ = cv2.Rodrigues(p[1:4])
        out = []
        for world_x, ln in pairs:
            uv, vis = project_line(p[0], rot_p, centre, world_x, cx=cx, cy=cy)
            d = np.column_stack([uv, np.ones(_N_SAMPLES)]) @ ln
            out.append(np.where(vis, d, _OFF_SENSOR_COST))
        return np.concatenate(out)

    before = float(np.median(np.abs(residual(p0))))
    sol = least_squares(residual, p0, loss="soft_l1", f_scale=2.0, max_nfev=150)
    if not np.isfinite(sol.x).all() or sol.x[0] <= 100.0:
        return None
    after = float(np.median(np.abs(residual(sol.x))))
    if after > before:
        return None
    rot_new, _ = cv2.Rodrigues(sol.x[1:4])
    focal_new = float(sol.x[0])
    if max_shift_px is not None and anchor is not None:
        if projected_shift(focal_new, rot_new, anchor[0], anchor[1], centre,
                           [w for w, _ in pairs], cx=cx, cy=cy) > max_shift_px:
            return None
    return focal_new, rot_new


def projected_shift(f_a, rot_a, f_b, rot_b, centre, world_xs, *, cx, cy):
    """Median on-sensor displacement between two cameras, in px.

    Only samples that land near the sensor count. A line projecting far outside
    moves enormously for a tiny camera change, and including those inflated this
    measure so much that a slip guard built on it rejected 926 of 1004 honest
    refinements.
    """
    out = []
    for world_x in world_xs:
        ua, va = project_line(f_a, rot_a, centre, world_x, cx=cx, cy=cy)
        ub, vb = project_line(f_b, rot_b, centre, world_x, cx=cx, cy=cy)
        m = va & vb
        if m.sum() >= 8:
            out.append(float(np.median(np.linalg.norm(ua[m] - ub[m], axis=1))))
    return float(np.median(out)) if out else float("inf")


def verify_frame(focal, rot, centre, dets, world_xs, *, cx, cy,
                 max_offset_px=6.0, min_ratio=3.0):
    """Score a camera against a frame's own lines, WITH a control.

    Returns ``(offset, control, ok)``. ``control`` is the same measurement with
    the field model shifted by one yard; if it is not at least ``min_ratio``
    times worse, this frame cannot distinguish a correct camera from a wrong
    one and ``ok`` is False regardless of how small ``offset`` looks.

    That guard is the whole point. Four per-frame metrics used during bring-up
    reported ~20 px on a camera independently known to be correct to 2.7 px --
    they were saturated, and only a control exposed it.
    """
    def score(dx):
        vals = []
        for world_x in world_xs:
            uv, vis = project_line(focal, rot, centre, world_x + dx,
                                   cx=cx, cy=cy)
            if vis.sum() < 8:
                continue
            homo = np.column_stack([uv[vis], np.ones(int(vis.sum()))])
            d = [float(np.median(np.abs(homo @ ln))) for ln in dets]
            if d and min(d) < 80.0:
                vals.append(min(d))
        return float(np.median(vals)) if len(vals) >= 3 else None

    shift_m = YARD_LINE_SPACING_M / 5.0             # one yard
    offset = score(0.0)
    control = score(shift_m)
    if offset is None or control is None:
        return offset, control, False

    # Judge the control against how far the model ACTUALLY MOVED on this frame,
    # not against a fixed ratio. A fixed ratio is not scale-aware: a camera off
    # by e, with the control shifted s px, reads about s - e, so demanding
    # control >= 3*offset silently also demands e <= s/4 -- on the endzone, where
    # a yard is only ~21 px, that quietly contradicted the 6 px accuracy gate and
    # rejected frames for being 5.45 px off. What actually indicates a saturated
    # metric is the control failing to move when the model does.
    moved = []
    for world_x in world_xs:
        ua, va = project_line(focal, rot, centre, world_x, cx=cx, cy=cy)
        ub, vb = project_line(focal, rot, centre, world_x + shift_m,
                              cx=cx, cy=cy)
        m = va & vb
        if m.sum() >= 8:
            moved.append(float(np.median(np.linalg.norm(ua[m] - ub[m], axis=1))))
    if not moved:
        return offset, control, False
    moved_px = float(np.median(moved))
    discriminates = (control - offset) >= 0.4 * moved_px
    return offset, control, (offset <= max_offset_px and discriminates)


def bundle_adjust(pair_homographies, anchors, centre, initial, *, cx, cy,
                  chain_weight=1.0, anchor_weight=3.0, max_nfev=400):
    """Solve every node's focal and rotation at once.

    ``pair_homographies`` maps ``(node_a, node_b)`` to the measured homography
    carrying node_a's pixels into node_b. ``anchors`` maps a node to its
    ``(world_x, image_line)`` pairs. ``initial`` maps a node to ``(focal, rot)``.
    Returns the same mapping, optimised.

    Two residual families, and BOTH are needed:

    * chain -- the tripod identity ``H = K_b R_b R_a^T K_a^-1`` holds exactly and
      independently of the scene, so each measured homography ties a pair of
      nodes. Supply CONSECUTIVE pairs: adjacent frames overlap heavily (measured
      median 304 inliers against a floor of 25), while the long-range fits onto
      a distant reference are precisely what drifts.
    * anchor -- without it the solution is only self-consistent. The whole chain
      could rotate or zoom as one rigid thing and satisfy every chain residual
      perfectly while sitting a yard line out, which a sweep did in practice
      while reporting a flattering 1.59 px.

    Error is then distributed across the network instead of accumulating along a
    path -- the same reason a stitched panorama bundles rather than chaining.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    nodes = sorted(initial)
    index = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    if n_nodes < 2:
        raise CalibrationError(
            "endzone bundle: need at least 2 nodes, got "
            f"{n_nodes} -- nothing to tie together.")
    if not anchors:
        raise CalibrationError(
            "endzone bundle: no anchor nodes. Chain residuals alone are "
            "satisfied by a solution that is rigidly wrong, so the result "
            "would be self-consistent and unusable.")

    grid = np.array([[u, v, 1.0] for u in (200.0, 700.0, 1200.0, 1700.0)
                     for v in (150.0, 500.0, 900.0)])
    pairs = sorted(pair_homographies)

    p0 = np.zeros(4 * n_nodes)
    for node in nodes:
        focal, rot = initial[node]
        rvec, _ = cv2.Rodrigues(rot)
        i = 4 * index[node]
        p0[i] = float(focal)
        p0[i + 1:i + 4] = rvec.ravel()

    def unpack(p, node):
        i = 4 * index[node]
        rot, _ = cv2.Rodrigues(p[i + 1:i + 4])
        return p[i], rot

    def residual(p):
        out = []
        for a, b in pairs:
            f_a, rot_a = unpack(p, a)
            f_b, rot_b = unpack(p, b)
            k_a = np.array([[f_a, 0.0, cx], [0.0, f_a, cy], [0.0, 0.0, 1.0]])
            k_b = np.array([[f_b, 0.0, cx], [0.0, f_b, cy], [0.0, 0.0, 1.0]])
            implied = k_b @ rot_b @ rot_a.T @ np.linalg.inv(k_a)
            q1 = (implied @ grid.T).T
            q2 = (pair_homographies[(a, b)] @ grid.T).T
            w1 = np.where(np.abs(q1[:, 2:3]) < 1e-9, 1e-9, q1[:, 2:3])
            w2 = np.where(np.abs(q2[:, 2:3]) < 1e-9, 1e-9, q2[:, 2:3])
            out.append(chain_weight * np.clip(
                q1[:, :2] / w1 - q2[:, :2] / w2, -500.0, 500.0).ravel())
        for node, got in anchors.items():
            focal, rot = unpack(p, node)
            for world_x, ln in got:
                uv, vis = project_line(focal, rot, centre, world_x, cx=cx, cy=cy)
                d = np.column_stack([uv, np.ones(_N_SAMPLES)]) @ ln
                out.append(anchor_weight * np.where(
                    vis, np.clip(d, -200.0, 200.0), 0.0))
        return np.concatenate(out)

    # Sparsity is mandatory, not an optimisation: each residual touches at most
    # two nodes, and without the pattern the solver probes a dense Jacobian with
    # 4 columns per node every iteration and never finishes.
    rows_chain = 2 * len(grid)
    rows_anchor = {n: _N_SAMPLES * len(g) for n, g in anchors.items()}
    total = rows_chain * len(pairs) + sum(rows_anchor.values())
    sparsity = lil_matrix((total, 4 * n_nodes), dtype=int)
    row = 0
    for a, b in pairs:
        for node in (a, b):
            sparsity[row:row + rows_chain,
                     4 * index[node]:4 * index[node] + 4] = 1
        row += rows_chain
    for node in anchors:
        sparsity[row:row + rows_anchor[node],
                 4 * index[node]:4 * index[node] + 4] = 1
        row += rows_anchor[node]

    # Per-parameter scaling is required: focal is ~1e4 while rotation components
    # are ~1, and without it the trust region is meaningless for one block or the
    # other -- the solve stalled on xtol after cutting the cost 18%, and drove one
    # node's focal to 1328 against a median of 17791. Bounds keep focal physical.
    x_scale = np.tile([1.0e4, 1.0e-1, 1.0e-1, 1.0e-1], n_nodes)
    lo = np.tile([4.0e3, -np.inf, -np.inf, -np.inf], n_nodes)
    hi = np.tile([3.0e4, np.inf, np.inf, np.inf], n_nodes)

    before = float(np.median(np.abs(residual(p0))))
    sol = least_squares(residual, np.clip(p0, lo, hi), jac_sparsity=sparsity,
                        loss="soft_l1", f_scale=3.0, x_scale=x_scale,
                        bounds=(lo, hi), max_nfev=max_nfev,
                        xtol=1e-10, ftol=1e-10)
    after = float(np.median(np.abs(residual(sol.x))))
    _LOG.info("endzone bundle: %d nodes, %d chain pairs, %d anchors; "
              "median |residual| %.2f -> %.2f px",
              n_nodes, len(pairs), len(anchors), before, after)
    return {node: (float(unpack(sol.x, node)[0]), unpack(sol.x, node)[1])
            for node in nodes}
