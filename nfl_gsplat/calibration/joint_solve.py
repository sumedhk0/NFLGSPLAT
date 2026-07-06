"""Fixed-center joint solve (pretrained calibration, phase 3).

The camera is on a tripod: ONE shared center C for the whole play, per-frame
rotation r_t (Rodrigues) and focal f_t (fx=fy, principal point = image
center). Fit jointly to every fused correspondence with a robust loss —
per-frame PnP on planar telephoto views is multimodal (measured on real
footage: the same frame solves blind but fails with a neighbor's prior), and
a consistently mislabeled yard line is invisible per frame but irreconcilable
with the shared center.

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
MIN_INIT_FRAMES = 20
AUDIT_DROP_PX = 6.0
F_SCALE_PX = 3.0

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


def init_from_results(init_results, frame_ids):
    """Initialize (C0, r0, f0) from the per-frame sweep's successes.

    C0 = median center of successes (center is the well-agreed quantity on
    real footage); per-frame r/f from successes, linearly interpolated (and
    boundary-clamped) across frames the sweep could not solve. Fails loud
    when the sweep produced too few successes to trust."""
    import cv2
    ok = [(i, r) for i, r in enumerate(init_results) if r is not None]
    if len(ok) < MIN_INIT_FRAMES:
        raise CalibrationError(
            f"only {len(ok)} per-frame initializer successes (< {MIN_INIT_FRAMES}) — "
            "not enough to seed the fixed-center joint solve. Check the fusion "
            "diagnostics (scripts/diag_pretrained.py)."
        )
    C0 = np.median(np.stack([r.pose.center_world() for _i, r in ok]), axis=0)
    ok_ids = np.array([i for i, _r in ok], dtype=float)
    rvecs = np.stack([cv2.Rodrigues(r.pose.R)[0].ravel() for _i, r in ok])
    fs = np.array([r.intrinsics.fx for _i, r in ok])
    r0, f0 = {}, {}
    for i in frame_ids:
        f0[i] = float(np.interp(i, ok_ids, fs))
        r0[i] = np.array([np.interp(i, ok_ids, rvecs[:, d]) for d in range(3)])
    return C0, r0, f0


def _solve_once(frame_ids, frame_data, image_size, C0, r0, f0):
    """One trf round. Returns (C, r_by, f_by, robust_cost_before, robust_cost_after)."""
    from scipy.optimize import least_squares
    x0 = pack_params(C0, r0, f0, frame_ids)
    lo = np.full_like(x0, -np.inf)
    hi = np.full_like(x0, np.inf)
    for k in range(len(frame_ids)):
        lo[3 + 4 * k + 3] = F_BOUNDS[0]
        hi[3 + 4 * k + 3] = F_BOUNDS[1]
    x0 = np.clip(x0, lo, hi)

    def fn(x):
        return residuals(x, frame_ids, frame_data, image_size)
    sol = least_squares(fn, x0, method="trf", loss="soft_l1", f_scale=F_SCALE_PX,
                        jac_sparsity=jac_sparsity(frame_ids, frame_data),
                        bounds=(lo, hi), max_nfev=200)

    def robust_cost(x):
        r = fn(x) / F_SCALE_PX
        return float(np.sum(2.0 * (np.sqrt(1.0 + r ** 2) - 1.0)))
    C, r_by, f_by = unpack_params(sol.x, frame_ids)
    return C, r_by, f_by, robust_cost(x0), robust_cost(sol.x)


def _frame_result(i, frame_data, C, r_by, f_by, image_size):
    import cv2
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose
    world, uv = frame_data[i]
    proj = _project(world, r_by[i], f_by[i], C, image_size)
    rms = float(np.sqrt(np.mean(np.sum((proj - uv) ** 2, axis=1))))
    R = cv2.Rodrigues(np.asarray(r_by[i], np.float64))[0]
    return CalibrationResult(
        intrinsics=CameraIntrinsics(f_by[i], f_by[i], image_size[0] / 2.0,
                                    image_size[1] / 2.0, image_size[0], image_size[1]),
        pose=CameraPose(R=R, t=-R @ C),
        rms_px=rms, num_correspondences=len(uv), refined_with_ba=True)


def solve_fixed_center(corrs_by_frame, image_size, *, init_results,
                       max_rounds: int = 2, _frame_data_override=None):
    """Entry point: joint solve over all usable frames -> [CalibrationResult|None]
    aligned to init_results' length. Self-audit (Task 3) drops irreconcilable
    frames between rounds."""
    frame_data = (_frame_data_override if _frame_data_override is not None
                  else build_frame_data(corrs_by_frame))
    frame_ids = sorted(frame_data)
    T = len(init_results)
    if not frame_ids:
        raise CalibrationError("no usable frames (>=4 correspondences) for the joint solve.")
    C0, r0, f0 = init_from_results(init_results, frame_ids)
    C, r_by, f_by, before, after = _solve_once(frame_ids, frame_data, image_size,
                                               C0, r0, f0)
    if after >= before:
        raise CalibrationError(
            f"fixed-center joint solve diverged (robust cost {before:.1f} -> {after:.1f}); "
            "not returning the initializer. Inspect fusion output.")

    results = [None] * T
    kept = list(frame_ids)
    for round_no in range(max_rounds):
        med = {}
        for i in kept:
            world, uv = frame_data[i]
            proj = _project(world, r_by[i], f_by[i], C, image_size)
            med[i] = float(np.median(np.linalg.norm(proj - uv, axis=1)))
        drop = [i for i in kept if med[i] > AUDIT_DROP_PX]
        if not drop or round_no == max_rounds - 1:
            break
        kept = [i for i in kept if i not in drop]
        if not kept:
            raise CalibrationError("self-audit dropped every frame — fusion output unusable.")
        C, r_by, f_by, before, after = _solve_once(
            kept, {i: frame_data[i] for i in kept}, image_size,
            C, {i: r_by[i] for i in kept}, {i: f_by[i] for i in kept})
        if after >= before:
            raise CalibrationError(
                f"joint re-solve after audit diverged ({before:.1f} -> {after:.1f}).")
    for i in kept:
        if med[i] <= AUDIT_DROP_PX:
            results[i] = _frame_result(i, frame_data, C, r_by, f_by, image_size)
    if all(r is None for r in results):
        raise CalibrationError(
            f"self-audit rejected every frame (all median residuals > {AUDIT_DROP_PX} px) — "
            "fusion output inconsistent with a fixed-center camera.")
    return results
