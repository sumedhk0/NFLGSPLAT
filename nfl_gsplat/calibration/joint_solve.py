"""Fixed-center joint solve (pretrained calibration, phase 3).

The camera is on a tripod: ONE shared center C for the whole play, per-frame
rotation r_t (Rodrigues) and focal f_t (fx=fy, principal point = image
center). Fit jointly to every fused correspondence with a robust loss —
per-frame PnP on planar telephoto views is multimodal (measured on real
footage: the same frame solves blind but fails with a neighbor's prior), and
a consistently mislabeled yard line is invisible per frame but irreconcilable
with the shared center.

Solve orchestration (design addendum 2026-07-06, validated on real footage):

1. Multi-start init — the per-frame sweep produced only physically-impossible
   anchors, so we do NOT trust it to seed the solve. Candidate camera centers
   come from a coarse plausibility grid plus (if the sweep left >= 3 plausible
   anchors) the median of those anchors. Each candidate gets per-frame look-at
   rotation + span-derived focal.
2. Reflection auto-resolve — the fused left/right hash convention is
   camera-side dependent; a mirrored labeling keeps every homography perfect
   but is a reflection no rigid camera fits. Both labelings (as-is / world-Y
   negated) are scored on a frame subsample; the lower-cost one wins. A
   mirrored win negates world Y for the rest of the solve (the field is
   Y-symmetric, so results stay in the TRUE world frame) and is reported.
3. Staged solve — trf needs x_scale and a subsample-deep -> full warm start to
   converge at this parameter scale.
4. Fixed-C per-frame rescue refit — freeze C, refit every frame's (r, f)
   independently with a small LM, keep frames whose median reprojection is
   within tolerance. Replaces the cascade audit, which dropped good frames
   against half-converged intermediates.

Parameter vector layout: [C (3)] + per usable frame [r_t (3), f_t (1)],
frames in ascending index order.
"""
from __future__ import annotations

import numpy as np

from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
from nfl_gsplat.errors import CalibrationError

LAMBDA_F = 0.01          # px of residual per px of frame-to-frame focal change
LAMBDA_R = 50.0          # px of residual per radian of frame-to-frame rotation change
SMOOTH_MAX_GAP = 3       # only penalize usable-frame pairs this close in index
F_BOUNDS = (200.0, 40000.0)
AUDIT_DROP_PX = 6.0
F_SCALE_PX = 3.0

# Multi-start grid of plausible camera centers (world meters). The grid does
# NOT cover the synthetic-test geometry near (-19, 1, 95); the plausible-anchor
# candidate keeps that reachable (and short-circuits the grid when present).
_GRID_X = (-30.0, 0.0, 30.0)
_GRID_Y = (-130.0, -80.0, -45.0, 45.0, 80.0, 130.0)
_GRID_Z = (15.0, 35.0, 65.0)

# Behind-endzone grid for cameras positioned behind an endzone (|X| 60-120, Y ~ 0)
_GRID_EZ_X = (-120.0, -90.0, -60.0, 60.0, 90.0, 120.0)
_GRID_EZ_Y = (-20.0, 0.0, 20.0)
_GRID_EZ_Z = (15.0, 35.0, 65.0)

# x_scale for trf: C in ~tens of meters, r in ~hundredths of a radian, f in
# ~thousands of pixels. Solving all three at unit scale stalls.
_XS_C = 10.0
_XS_R = 0.01
_XS_F = 1000.0

_SHORT_NFEV = 12         # per-candidate scoring solve (cost discrimination, not convergence)
_ANCHOR_NFEV = 50        # the plausible-anchor candidate gets a real convergence budget:
                         # it is a trusted physical prior and, when it early-accepts, the
                         # grid is never scored
_STAGE_A_NFEV = 1000     # subsample deep solve
_STAGE_B_NFEV = 600      # full warm-started solve
_RESCUE_NFEV = 300       # per-frame fixed-C LM refit
_EARLY_ACCEPT_PX = 2.0   # a candidate this good on the subsample ends the search

FrameData = tuple[np.ndarray, np.ndarray]     # (world (N,3), uv (N,2))


def build_frame_data(corrs_by_frame, min_corrs: int = 4) -> dict[int, FrameData]:
    """Resolve landmark names to world points; keep frames with >= min_corrs."""
    out: dict[int, FrameData] = {}
    for fidx, corrs in corrs_by_frame.items():
        if not corrs or len(corrs) < min_corrs:
            continue
        world_pts = []
        for n, _uv in corrs:
            if n not in NFL_LANDMARKS:
                raise CalibrationError(f"unknown landmark {n!r} in correspondences.")
            world_pts.append(NFL_LANDMARKS[n])
        world = np.stack(world_pts).astype(np.float64)
        uv = np.array([p for (_n, p) in corrs], dtype=np.float64)
        out[int(fidx)] = (world, uv)
    return out


def pack_params(C, r_by_frame, f_by_frame, frame_ids) -> np.ndarray:
    parts = [np.asarray(C, np.float64).ravel()]
    for i in frame_ids:
        parts.append(np.asarray(r_by_frame[i], np.float64).ravel())
        parts.append(np.array([float(f_by_frame[i])]))
    return np.concatenate(parts)


def unpack_params(x, frame_ids):
    C = x[:3]
    r_by_frame, f_by_frame = {}, {}
    for k, i in enumerate(frame_ids):
        o = 3 + 4 * k
        r_by_frame[i] = x[o:o + 3]
        f_by_frame[i] = float(x[o + 3])
    return C, r_by_frame, f_by_frame


def _project(world, r, f, C, image_size):
    """Project (N,3) world points; camera center C, Rodrigues r, focal f."""
    import cv2
    R = cv2.Rodrigues(np.asarray(r, np.float64))[0]
    t = -R @ np.asarray(C, np.float64)
    x_cam = world @ R.T + t.reshape(1, 3)
    z = np.maximum(x_cam[:, 2], 1e-9)          # smooth guard: no NaN in residuals
    u = f * x_cam[:, 0] / z + image_size[0] / 2.0
    v = f * x_cam[:, 1] / z + image_size[1] / 2.0
    return np.stack([u, v], axis=1)


def residuals(x, frame_ids, frame_data, image_size, *,
              lambda_f: float = LAMBDA_F, lambda_r: float = LAMBDA_R,
              smooth_max_gap: int = SMOOTH_MAX_GAP) -> np.ndarray:
    C, r_by, f_by = unpack_params(x, frame_ids)
    parts = []
    for i in frame_ids:
        world, uv = frame_data[i]
        parts.append((_project(world, r_by[i], f_by[i], C, image_size) - uv).ravel())
    for a, b in zip(frame_ids, frame_ids[1:]):
        if b - a > smooth_max_gap:
            continue
        parts.append(np.array([lambda_f * (f_by[b] - f_by[a])]))
        parts.append(lambda_r * (r_by[b] - r_by[a]))
    return np.concatenate(parts)


def jac_sparsity(frame_ids, frame_data, *, smooth_max_gap: int = SMOOTH_MAX_GAP):
    """Sparsity pattern aligned with `residuals`: each reprojection row touches
    C (cols 0-2) and its frame's (r, f) block; smoothness rows touch two blocks."""
    from scipy.sparse import lil_matrix
    col = {i: 3 + 4 * k for k, i in enumerate(frame_ids)}
    n_rows = sum(2 * len(frame_data[i][1]) for i in frame_ids)
    pairs = [(a, b) for a, b in zip(frame_ids, frame_ids[1:])
             if b - a <= smooth_max_gap]
    n_rows += sum(1 + 3 for _ in pairs)
    S = lil_matrix((n_rows, 3 + 4 * len(frame_ids)), dtype=np.uint8)
    row = 0
    for i in frame_ids:
        n = 2 * len(frame_data[i][1])
        S[row:row + n, 0:3] = 1
        S[row:row + n, col[i]:col[i] + 4] = 1
        row += n
    for a, b in pairs:
        S[row, col[a] + 3] = 1; S[row, col[b] + 3] = 1        # focal pair
        row += 1
        for d in range(3):                                     # rotation pair
            S[row, col[a] + d] = 1; S[row, col[b] + d] = 1
            row += 1
    return S


def init_from_results(init_results) -> np.ndarray | None:
    """Median center of the sweep's PLAUSIBLE anchors, or None.

    The per-frame sweep on real footage produces physically-impossible anchors
    (focals to 1e13 px, centers scattered 1e8 m); those are gated out. If at
    least three anchors survive the plausibility gate their median center is a
    trustworthy extra multi-start candidate (it keeps the synthetic geometry
    near (-19, 1, 95) reachable). Fewer than three plausible anchors -> None;
    the grid multi-start covers that case, so this no longer fails loud."""
    plausible = []
    for r in init_results:
        if r is None:
            continue
        C = np.asarray(r.pose.center_world(), np.float64)
        f = float(r.intrinsics.fx)
        if (1000.0 <= f <= 25000.0 and 5.0 <= C[2] <= 250.0
                and abs(C[0]) <= 300.0 and abs(C[1]) <= 400.0):
            plausible.append(C)
    if len(plausible) < 3:
        return None
    return np.median(np.stack(plausible), axis=0)


def _look_at_R(C, target):
    """World->camera rotation for a camera at C looking at target, +Z world up.
    Camera axes: z forward, x right, y down. None if the geometry is degenerate."""
    fwd = np.asarray(target, np.float64) - np.asarray(C, np.float64)
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return None
    fwd = fwd / n
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        return None
    right = right / rn
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=0)


def _init_frame(C, world, uv, *, view_deg: int = 0):
    """Per-frame look-at rotation + span-derived focal seed for a camera at C.

    The minimal-roll look-at (right = fwd x world-Z) is correct for an
    unrotated view. A working image that was itself rotated 90/180/270 deg
    upstream (endzone pipeline, see view_rotation.py) needs that same extra
    in-plane roll composed in -- something look-at cannot see from C and
    target alone (it never looks at uv).

    ``view_deg`` is the KNOWN working-view rotation (0 for sideline, 90/180/270
    for endzone), passed in by the caller -- it is not inferred here. When
    ``view_deg == 0`` this seeds with the plain roll=0 look-at only (the
    pre-roll-search behavior): an error-threshold heuristic that instead tried
    to detect rotation from reprojection error cannot tell "wrong candidate
    center" from "rotated view" apart (both produce high roll=0 error), and at
    a WRONG multi-start candidate the 4-roll scan finds spuriously low-cost
    non-zero rolls that let a wrong camera win the multi-start search
    (measured on real sideline footage: winner cost 63041 at a wrong C vs the
    true 84397 at the correct C). When ``view_deg != 0`` scores all 4
    candidate rolls against the actual observations (shape only, both point
    clouds re-centered on their own mean so no principal-point assumption is
    needed) and seeds LM from whichever is closest."""
    import cv2
    from nfl_gsplat.calibration.view_rotation import _rz
    target = world.mean(axis=0)
    R = _look_at_R(C, target)
    if R is None:
        return None, None
    dist = np.linalg.norm(target - np.asarray(C, np.float64))
    wspan = np.linalg.norm(world.max(0) - world.min(0))
    pspan = np.linalg.norm(uv.max(0) - uv.min(0))
    f = float(np.clip(pspan * dist / max(wspan, 1e-6), 500.0, 39000.0))

    if view_deg == 0:
        return cv2.Rodrigues(R)[0].ravel(), f

    C_np = np.asarray(C, np.float64)
    x_world = world - C_np
    uv_c = uv - uv.mean(axis=0)
    best_deg, best_err = 0, np.inf
    for deg in (0, 90, 180, 270):
        Rr = R if deg == 0 else _rz(deg) @ R
        x_cam = x_world @ Rr.T
        z = np.maximum(x_cam[:, 2], 1e-9)
        proj = f * x_cam[:, :2] / z[:, None]
        err = float(np.median(np.linalg.norm(proj - proj.mean(axis=0) - uv_c, axis=1)))
        if err < best_err:
            best_deg, best_err = deg, err
    if best_deg != 0:
        R = _rz(best_deg) @ R
    return cv2.Rodrigues(R)[0].ravel(), f


def _init_pose_for_C(C, ids, frame_data, *, view_deg: int = 0):
    """look-at/span init for every frame in ids, or (None, None) if degenerate."""
    r0, f0 = {}, {}
    for i in ids:
        world, uv = frame_data[i]
        r, f = _init_frame(C, world, uv, view_deg=view_deg)
        if r is None:
            return None, None
        r0[i], f0[i] = r, f
    return r0, f0


def _negate_y(frame_data):
    """Left/right relabel: negate world Y (the field is Y-symmetric)."""
    flip = np.array([1.0, -1.0, 1.0])
    return {i: (world * flip, uv) for i, (world, uv) in frame_data.items()}


def _median_reproj(ids, frame_data, C, r_by, f_by, image_size) -> float:
    """Median over frames of each frame's median reprojection residual (px)."""
    meds = []
    for i in ids:
        world, uv = frame_data[i]
        proj = _project(world, r_by[i], f_by[i], C, image_size)
        meds.append(float(np.median(np.linalg.norm(proj - uv, axis=1))))
    return float(np.median(meds)) if meds else float("inf")


def _robust_cost(res: np.ndarray) -> float:
    r = res / F_SCALE_PX
    return float(np.sum(2.0 * (np.sqrt(1.0 + r ** 2) - 1.0)))


def _solve_once(frame_ids, frame_data, image_size, C0, r0, f0, *, max_nfev=200, jac=None):
    """One trf round with x_scale. Returns (C, r_by, f_by, cost_before, cost_after).

    ``jac`` optionally supplies a precomputed sparsity pattern (the pattern only
    depends on frame_ids and per-frame point counts, so the multi-start scoring
    loop reuses one instance across every candidate)."""
    from scipy.optimize import least_squares
    x0 = pack_params(C0, r0, f0, frame_ids)
    lo = np.full_like(x0, -np.inf)
    hi = np.full_like(x0, np.inf)
    xs = np.ones_like(x0)
    xs[:3] = _XS_C
    for k in range(len(frame_ids)):
        o = 3 + 4 * k
        lo[o + 3] = F_BOUNDS[0]
        hi[o + 3] = F_BOUNDS[1]
        xs[o:o + 3] = _XS_R
        xs[o + 3] = _XS_F
    x0 = np.clip(x0, lo, hi)

    def fn(x):
        return residuals(x, frame_ids, frame_data, image_size)
    sparsity = jac if jac is not None else jac_sparsity(frame_ids, frame_data)
    # 1e-4 tolerances: the finite-difference Jacobian makes each iteration
    # costly, and the shared center is nailed to ~0.1 m well before the default
    # 1e-8 tolerances stop it (per-frame focal is then refined tightly by the
    # fixed-C rescue LM anyway).
    sol = least_squares(fn, x0, method="trf", loss="soft_l1", f_scale=F_SCALE_PX,
                        jac_sparsity=sparsity, bounds=(lo, hi), x_scale=xs, max_nfev=max_nfev,
                        ftol=1e-4, xtol=1e-4, gtol=1e-4)
    C, r_by, f_by = unpack_params(sol.x, frame_ids)
    return C, r_by, f_by, _robust_cost(fn(x0)), _robust_cost(fn(sol.x))


def _frame_result(i, frame_data, C, r, f, image_size):
    """Build a CalibrationResult for frame i from its per-frame (r, f) and shared C."""
    import cv2
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
    world, uv = frame_data[i]
    proj = _project(world, r, f, C, image_size)
    rms = float(np.sqrt(np.mean(np.sum((proj - uv) ** 2, axis=1))))
    R = cv2.Rodrigues(np.asarray(r, np.float64))[0]
    return CalibrationResult(
        intrinsics=CameraIntrinsics(f, f, image_size[0] / 2.0,
                                    image_size[1] / 2.0, image_size[0], image_size[1]),
        pose=CameraPose(R=R, t=-R @ np.asarray(C, np.float64)),
        rms_px=rms, num_correspondences=len(uv), refined_with_ba=True)


def _candidate_centers(init_results, seed_centers=None) -> list[np.ndarray]:
    """Seeds first, then the plausible-anchor median, then the grid.

    ``seed_centers`` are centres known from OUTSIDE this solve. The strongest
    such prior is another play from the SAME GAME: a broadcast camera sits on a
    fixed mount and does not move between snaps, so a centre solved once is a
    near-exact starting point for every other play of that game. The grid alone
    is coarse -- 3x6x3 points tens of metres apart -- and a camera that falls
    between its points can fail to be found at all.
    """
    cands: list[np.ndarray] = []
    for c in (seed_centers or ()):
        cands.append(np.asarray(c, dtype=np.float64).reshape(3))
    anchor = init_from_results(init_results)
    if anchor is not None:
        cands.append(anchor)
    for X in _GRID_X:
        for Y in _GRID_Y:
            for Z in _GRID_Z:
                cands.append(np.array([X, Y, Z]))
    for X in _GRID_EZ_X:                     # behind-endzone cameras
        for Y in _GRID_EZ_Y:
            for Z in _GRID_EZ_Z:
                cands.append(np.array([X, Y, Z]))
    return cands


def _filter_centers(cands, center_bounds):
    """Keep only candidate centers inside center_bounds
    ((xmin,xmax),(ymin,ymax),(zmin,zmax)); None -> unchanged. Always keeps at
    least the first candidate (the anchor) so the solve never starves."""
    if center_bounds is None:
        return cands
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = center_bounds
    kept = [c for c in cands
            if xlo <= c[0] <= xhi and ylo <= c[1] <= yhi and zlo <= c[2] <= zhi]
    return kept if kept else cands[:1]


def _subsample(frame_ids) -> list[int]:
    """Every 7th usable frame, at least 8 (densify if too sparse); all if < 8."""
    if len(frame_ids) <= 8:
        return list(frame_ids)
    sub = frame_ids[::7]
    if len(sub) < 8:
        idx = np.linspace(0, len(frame_ids) - 1, 8).round().astype(int)
        sub = [frame_ids[j] for j in sorted(set(idx.tolist()))]
    return sub


def _resolve_reflection(frame_ids, frame_data, image_size, candidates, n_anchor: int = 0,
                        *, view_deg: int = 0):
    """Score both labelings x every candidate on a subsample; return the winner.

    ``n_anchor`` is the count of leading entries in ``candidates`` that are
    plausible-anchor candidates (see ``init_from_results``) rather than grid
    points. Those get ``_ANCHOR_NFEV`` iterations — enough to actually converge,
    since the anchor is a trusted physical prior and an early accept skips
    scoring the grid entirely. The remaining (grid) candidates only need
    ``_SHORT_NFEV`` iterations to be distinguished from each other, not to
    converge. On real footage ``init_from_results`` returns None (sweep anchors
    are implausible), so ``n_anchor`` is 0 and nothing costs more than before
    this split existed.

    Returns (mirrored, C_refined, cost). Early-accepts the first candidate whose
    short-solve reaches <= _EARLY_ACCEPT_PX median reprojection on the subsample
    (its labeling is the true one — the reflection can never reach that)."""
    sub = _subsample(frame_ids)
    sparsity = jac_sparsity(sub, {i: frame_data[i] for i in sub})   # identical every candidate
    best = None                                   # (cost, mirrored, C)
    for mirrored, fd in ((False, frame_data), (True, _negate_y(frame_data))):
        sub_fd = {i: fd[i] for i in sub}
        for idx, C0 in enumerate(candidates):
            r0, f0 = _init_pose_for_C(C0, sub, fd, view_deg=view_deg)
            if r0 is None:
                continue
            nfev = _ANCHOR_NFEV if idx < n_anchor else _SHORT_NFEV
            C, r_by, f_by, _before, after = _solve_once(
                sub, sub_fd, image_size, C0, r0, f0, max_nfev=nfev, jac=sparsity)
            if not np.isfinite(after):
                continue
            if best is None or after < best[0]:
                best = (after, mirrored, C)
            if _median_reproj(sub, fd, C, r_by, f_by, image_size) <= _EARLY_ACCEPT_PX:
                return mirrored, C, after
    if best is None or not np.isfinite(best[0]):
        raise CalibrationError(
            "multi-start joint solve found no finite-cost camera candidate — "
            "fusion output unusable (inspect scripts/diag_pretrained.py).")
    return best[1], best[2], best[0]


def _staged_solve(frame_ids, frame_data, image_size, C_win, *, view_deg: int = 0):
    """Stage A (every 3rd frame, deep) -> Stage B (all frames, warm-started)."""
    subA = frame_ids[::3]
    rA0, fA0 = _init_pose_for_C(C_win, subA, frame_data, view_deg=view_deg)
    if rA0 is None:
        raise CalibrationError("joint solve: winning camera center is degenerate "
                               "for the subsample (look-at undefined).")
    CA, rA, fA, beforeA, afterA = _solve_once(
        subA, {i: frame_data[i] for i in subA}, image_size, C_win, rA0, fA0,
        max_nfev=_STAGE_A_NFEV)
    if afterA > beforeA:
        raise CalibrationError(
            f"fixed-center joint solve diverged in stage A "
            f"(robust cost {beforeA:.1f} -> {afterA:.1f}); not returning garbage.")

    idsA = np.array(subA, dtype=float)
    rstack = np.stack([rA[i] for i in subA])
    fs = np.array([fA[i] for i in subA])
    r0F = {i: np.array([np.interp(i, idsA, rstack[:, k]) for k in range(3)])
           for i in frame_ids}
    f0F = {i: float(np.interp(i, idsA, fs)) for i in frame_ids}
    CB, rB, fB, beforeB, afterB = _solve_once(
        frame_ids, frame_data, image_size, CA, r0F, f0F, max_nfev=_STAGE_B_NFEV)
    if afterB > beforeB:
        raise CalibrationError(
            f"fixed-center joint solve diverged in stage B "
            f"(robust cost {beforeB:.1f} -> {afterB:.1f}); not returning garbage.")
    return CB, rB, fB


def _rescue_refit(frame_ids, frame_data, image_size, C, *, view_deg: int = 0):
    """Fixed-C per-frame LM refit of (r, f). Returns {i: (r, f, median_px)}."""
    from scipy.optimize import least_squares
    out = {}
    for i in frame_ids:
        world, uv = frame_data[i]
        r0, f0 = _init_frame(C, world, uv, view_deg=view_deg)
        if r0 is None:
            continue
        x0 = np.concatenate([r0, [f0]])

        def fn(x, w=world, u=uv):
            return (_project(w, x[:3], x[3], C, image_size) - u).ravel()
        sol = least_squares(fn, x0, method="lm", max_nfev=_RESCUE_NFEV)
        proj = _project(world, sol.x[:3], sol.x[3], C, image_size)
        med = float(np.median(np.linalg.norm(proj - uv, axis=1)))
        out[i] = (sol.x[:3].copy(), float(sol.x[3]), med)
    return out


def solve_fixed_center(corrs_by_frame, image_size, *, init_results,
                       max_rounds: int = 2, _frame_data_override=None,
                       view_deg: int = 0, center_bounds=None, audit_drop_px=None,
                       seed_centers=None, fixed_center=None):
    """Joint solve over all usable frames -> (results, mirrored).

    ``results`` is a list aligned to ``init_results``' length with a
    ``CalibrationResult`` per rescued frame and ``None`` elsewhere; ``mirrored``
    is True when the fused left/right hash convention was flipped (world Y
    negated) to fit a rigid camera. The returned results are always in the TRUE
    world frame. ``max_rounds`` is accepted for backward compatibility.

    ``view_deg`` is the KNOWN working-view rotation (0 for sideline, 90 for
    endzone, etc. -- see view_rotation.py). It gates the 4-roll init search in
    ``_init_frame``: that search is needed to seed a rotated view correctly
    but is HARMFUL for an unrotated view, where it lets wrong multi-start
    candidates reach a spuriously low-cost minimum (measured on real sideline
    footage, see joint_solve.py module docstring / _init_frame). Default 0
    preserves prior behavior for existing callers.

    ``center_bounds`` restricts the multi-start candidate centers to a box
    ((xmin,xmax),(ymin,ymax),(zmin,zmax)); None (default) leaves the full
    candidate set untouched -- see ``_filter_centers``.

    ``audit_drop_px`` overrides the module-level ``AUDIT_DROP_PX`` gate for
    the frame-consistency audits in this solve only; None (default) falls
    back to ``AUDIT_DROP_PX``.

    ``seed_centers`` are extra starting centres tried BEFORE the grid -- see
    ``_candidate_centers``. Use it to carry a camera from one play of a game to
    the next, since the mount does not move between snaps.
    ``fixed_center`` HOLDS the centre there instead: no multi-start, no staged
    solve; the reflection is resolved against that centre and every frame's
    (r, f) is refit with C frozen (the rescue refit). Measured 2026-09-05 on
    play 3: seeding alone let the solve wander to a 41-degree lens 130 m out
    with the best paint residual of all (4.2 px) -- the paint's minimum is
    not at the mount, so on a play whose paint admits the wrong lens the
    centre must be held, and the paint then sets the focal per frame."""
    del max_rounds
    drop_px = AUDIT_DROP_PX if audit_drop_px is None else audit_drop_px
    frame_data = (_frame_data_override if _frame_data_override is not None
                  else build_frame_data(corrs_by_frame))
    frame_ids = sorted(frame_data)
    T = len(init_results)
    if not frame_ids:
        raise CalibrationError("no usable frames (>=4 correspondences) for the joint solve.")

    if fixed_center is not None:
        C = np.asarray(fixed_center, float).reshape(3)
        mirrored, C_win, cost = _resolve_reflection(
            frame_ids, frame_data, image_size, [C], n_anchor=1, view_deg=view_deg)
        if not np.isfinite(cost):
            raise CalibrationError("fixed-centre solve: cost not finite at the given centre.")
        if mirrored:
            frame_data = _negate_y(frame_data)
        refit = _rescue_refit(frame_ids, frame_data, image_size, C, view_deg=view_deg)
        results = [None] * T
        for i, (r_i, f_i, med) in refit.items():
            if med <= drop_px:
                results[i] = _frame_result(i, frame_data, C, r_i, f_i, image_size)
        if all(r is None for r in results):
            raise CalibrationError(
                f"fixed-centre solve: every frame's median residual > {drop_px} px at "
                f"C={np.round(C, 1)} -- the paint does not admit a camera on that mount.")
        return results, mirrored
    candidates = _candidate_centers(init_results, seed_centers)
    candidates = _filter_centers(candidates, center_bounds)
    n_anchor = len(seed_centers or ()) + (
        1 if init_from_results(init_results) is not None else 0)
    mirrored, C_win, cost = _resolve_reflection(
        frame_ids, frame_data, image_size, candidates, n_anchor=n_anchor,
        view_deg=view_deg)
    if not np.isfinite(cost):
        raise CalibrationError("multi-start joint solve: best candidate cost not finite.")
    if mirrored:
        frame_data = _negate_y(frame_data)

    # Rescue-first: refit every frame against the multi-start winner and run the
    # staged solve on CLEAN frames only. Solving on all frames first lets the
    # poisoned minority (residual identity/geometry errors) drag the shared
    # center away (measured on real footage: C wandered onto the field and the
    # final rescue rejected everything).
    refit0 = _rescue_refit(frame_ids, frame_data, image_size, C_win, view_deg=view_deg)
    kept = sorted(i for i, (_r, _f, m) in refit0.items() if m <= drop_px)
    if len(kept) < 10:
        raise CalibrationError(
            f"only {len(kept)} frames consistent with the multi-start camera "
            f"(need >= 10) — fusion output unusable for a fixed-center solve.")
    C, _r_by, _f_by = _staged_solve(kept, {i: frame_data[i] for i in kept},
                                    image_size, C_win, view_deg=view_deg)

    refit = _rescue_refit(frame_ids, frame_data, image_size, C, view_deg=view_deg)
    results = [None] * T
    for i, (r_i, f_i, med) in refit.items():
        if med <= drop_px:
            results[i] = _frame_result(i, frame_data, C, r_i, f_i, image_size)
    if all(r is None for r in results):
        raise CalibrationError(
            f"self-audit rejected every frame (all median residuals > {drop_px} px) — "
            "fusion output inconsistent with a fixed-center camera.")
    return results, mirrored
