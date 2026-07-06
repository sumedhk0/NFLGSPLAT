# Fixed-Center Joint Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragile per-frame PnP in pretrained calibration with one joint optimization: a single shared camera center + smooth per-frame rotation/focal fit to ALL fused correspondences.

**Architecture:** New module `nfl_gsplat/calibration/joint_solve.py` with a parameter vector `[C(3), then per usable frame r_t(3), f_t(1)]`, sparse-Jacobian `scipy.optimize.least_squares(method="trf", loss="soft_l1")` over reprojection residuals + small smoothness penalties, initialized from the existing per-frame sweep, followed by a self-audit round that drops irreconcilable frames. Wired into `build_autocalib_npz_pretrained` as phase 3; output feeds the unchanged `assemble_track_from_results` → `cameras.npz`.

**Tech Stack:** numpy, scipy (`least_squares` — already a project dependency, see `solve_pnp.py:22`), cv2 (`Rodrigues`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-fixed-center-joint-solve-design.md`

## Global Constraints

- Camera model: `fx = fy = f_t`, principal point = image center `(W/2, H/2)`, zero skew/distortion.
- Convention (from `nfl_gsplat/utils/geometry.py`): `CameraPose.R/t` are world→camera, `x_cam = R X + t`, camera center `C = -Rᵀ t`; therefore per-frame `t_t = -R_t @ C`.
- Pinned constants: smoothness weights `LAMBDA_F = 0.01` (px residual per px of Δf), `LAMBDA_R = 50.0` (px residual per radian of Δr), applied only between usable frames ≤3 indices apart; robust loss `soft_l1`, `f_scale = 3.0`; self-audit drop threshold `6.0` px median per frame; max 2 solve rounds; minimum initializer successes `MIN_INIT_FRAMES = 20`; usable frame = ≥4 correspondences; focal bounds `[200, 40000]` px.
- Fail loud: `CalibrationError` (from `nfl_gsplat.errors`) on too-few init frames or optimizer divergence (final robust cost ≥ initial). NEVER silently return init or per-frame results.
- No GPU/torch. Synthetic tests must run in seconds. `pytest -m "not gpu and not slow"` stays green.
- NEVER commit real NFL video/frames; tests are synthetic numpy only.
- Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Parameterization + residual core (no optimizer yet)

**Files:**
- Create: `nfl_gsplat/calibration/joint_solve.py`
- Test: `tests/test_joint_solve.py` (new)

**Interfaces:**
- Consumes: `CameraIntrinsics`, `CameraPose`, `project_points` (`nfl_gsplat/utils/geometry.py`), `NFL_LANDMARKS` (`field_landmarks.py`), `cv2.Rodrigues`.
- Produces (Task 2 relies on these exact names):
  - `FrameData = tuple[np.ndarray, np.ndarray]` — per-frame `(world_xyz (N,3), uv (N,2))` float64 arrays.
  - `build_frame_data(corrs_by_frame: dict[int, list], min_corrs: int = 4) -> dict[int, FrameData]` — resolves landmark names to world points, keeps frames with ≥ `min_corrs`.
  - `pack_params(C, r_by_frame, f_by_frame, frame_ids) -> np.ndarray` and `unpack_params(x, frame_ids) -> (C, {fidx: r}, {fidx: f})`.
  - `residuals(x, frame_ids, frame_data, image_size, *, lambda_f=0.01, lambda_r=50.0, smooth_max_gap=3) -> np.ndarray` — reprojection residuals (2 per point, px) followed by smoothness residuals.
  - `jac_sparsity(frame_ids, frame_data, *, smooth_max_gap=3) -> scipy.sparse.lil_matrix` — rows aligned with `residuals` output.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_joint_solve.py
"""Fixed-center joint solve. Synthetic camera: fixed center, panning/zooming,
looking at the field plane — all geometry self-checked via project_points."""
import numpy as np
import pytest

from nfl_gsplat.utils.geometry import CameraIntrinsics, CameraPose, project_points

W, H = 1920, 1080
C_TRUE = np.array([-19.0, 1.0, 95.0])          # measured real-camera ballpark


def _look_at_R(C, target):
    """World->camera rotation for a camera at C looking at target, +Z world up.
    Camera axes: z forward, x right, y down (standard CV)."""
    fwd = target - C
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=0)


def _synthetic_frames(n_frames=50, noise_px=0.5, seed=0):
    """Fixed center, pan target sweeping x in [-15, 15], focal ramp 7000->8500.
    World points: 4 yard lines x hash rows + number rows (well-spread, on Z=0).
    Returns (frame_ids, frame_data {fidx: (world, uv)}, f_true, R_true)."""
    from nfl_gsplat.calibration.field_landmarks import (
        HASH_OFFSET_M, NUMBER_CENTER_Y_M, YARD_LINE_SPACING_M,
    )
    rng = np.random.default_rng(seed)
    Xs = np.array([-2, -1, 0, 1, 2]) * YARD_LINE_SPACING_M * 2   # 5 lines, 10yd apart
    Ys = np.array([+NUMBER_CENTER_Y_M, +HASH_OFFSET_M, -HASH_OFFSET_M, -NUMBER_CENTER_Y_M])
    world = np.array([[x, y, 0.0] for x in Xs for y in Ys])      # 20 points
    frame_data, f_true, R_true = {}, {}, {}
    for i in range(n_frames):
        f = 7000.0 + 1500.0 * i / max(1, n_frames - 1)
        tx = -15.0 + 30.0 * i / max(1, n_frames - 1)
        R = _look_at_R(C_TRUE, np.array([tx, 0.0, 0.0]))
        t = -R @ C_TRUE
        K = CameraIntrinsics(f, f, W / 2, H / 2, W, H).K()
        uv = project_points(world, K, R, t)
        ok = np.isfinite(uv).all(axis=1)
        assert ok.sum() >= 8, "synthetic geometry broken — points behind camera"
        uv_n = uv[ok] + rng.normal(0, noise_px, uv[ok].shape)
        frame_data[i] = (world[ok].copy(), uv_n)
        f_true[i], R_true[i] = f, R
    return sorted(frame_data), frame_data, f_true, R_true


def test_pack_unpack_round_trip():
    from nfl_gsplat.calibration.joint_solve import pack_params, unpack_params
    ids = [0, 3, 7]
    C = np.array([1.0, 2.0, 3.0])
    r = {0: np.array([0.1, 0.0, 0.0]), 3: np.array([0.0, 0.2, 0.0]),
         7: np.array([0.0, 0.0, 0.3])}
    f = {0: 7000.0, 3: 7100.0, 7: 7200.0}
    x = pack_params(C, r, f, ids)
    assert x.shape == (3 + 4 * 3,)
    C2, r2, f2 = unpack_params(x, ids)
    assert np.allclose(C2, C)
    for i in ids:
        assert np.allclose(r2[i], r[i]) and f2[i] == pytest.approx(f[i])


def test_residuals_zero_at_ground_truth_no_noise():
    import cv2
    from nfl_gsplat.calibration.joint_solve import pack_params, residuals
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=5, noise_px=0.0)
    r = {i: cv2.Rodrigues(R_true[i])[0].ravel() for i in ids}
    x = pack_params(C_TRUE, r, f_true, ids)
    res = residuals(x, ids, fd, (W, H))
    n_pts = sum(len(fd[i][1]) for i in ids)
    assert res.shape[0] >= 2 * n_pts            # reprojection + smoothness rows
    assert np.abs(res[:2 * n_pts]).max() < 1e-6  # exact at ground truth
    # smoothness rows small but the focal ramp is nonzero
    assert np.abs(res[2 * n_pts:]).max() < 25.0


def test_jac_sparsity_shape_and_locality():
    import cv2
    from nfl_gsplat.calibration.joint_solve import (
        jac_sparsity, pack_params, residuals,
    )
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=4, noise_px=0.0)
    r = {i: cv2.Rodrigues(R_true[i])[0].ravel() for i in ids}
    x = pack_params(C_TRUE, r, f_true, ids)
    S = jac_sparsity(ids, fd)
    assert S.shape == (residuals(x, ids, fd, (W, H)).shape[0], x.shape[0])
    S = S.toarray()
    # first frame's first residual touches C (cols 0-2) + frame 0's block (3-6) only
    row = S[0]
    assert row[:7].all() and not row[7:].any()


def test_build_frame_data_resolves_names_and_filters():
    from nfl_gsplat.calibration.field_landmarks import NFL_LANDMARKS
    from nfl_gsplat.calibration.joint_solve import build_frame_data
    corrs = {
        0: [("away_30_left_hash", (100.0, 200.0)), ("away_30_right_hash", (110.0, 500.0)),
            ("away_20_left_hash", (700.0, 210.0)), ("away_20_right_hash", (720.0, 520.0))],
        1: [("away_30_left_hash", (1.0, 2.0))],          # <4 -> dropped
    }
    fd = build_frame_data(corrs)
    assert set(fd) == {0}
    world, uv = fd[0]
    assert world.shape == (4, 3) and uv.shape == (4, 2)
    assert np.allclose(world[0], NFL_LANDMARKS["away_30_left_hash"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_joint_solve.py -v`
Expected: FAIL — `ModuleNotFoundError: nfl_gsplat.calibration.joint_solve`

- [ ] **Step 3: Implement**

```python
# nfl_gsplat/calibration/joint_solve.py
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
        world = np.stack([NFL_LANDMARKS[n] for (n, _uv) in corrs]).astype(np.float64)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_joint_solve.py -v`
Expected: 4 PASS

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/joint_solve.py tests/test_joint_solve.py
git commit -m "feat(calibration): joint-solve parameterization and residual core

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Optimizer + initialization + divergence guard

**Files:**
- Modify: `nfl_gsplat/calibration/joint_solve.py` (append)
- Test: `tests/test_joint_solve.py` (append)

**Interfaces:**
- Consumes: Task 1's `build_frame_data`, `pack_params`, `unpack_params`, `residuals`, `jac_sparsity`; `CalibrationResult` (`solve_pnp.py`: fields `intrinsics, pose, rms_px, num_correspondences, refined_with_ba`); `CameraPose.center_world()`.
- Produces (Tasks 3-4 rely on):
  - `init_from_results(init_results: list, frame_ids: list[int]) -> (C0, r0: dict, f0: dict)` — raises `CalibrationError` if `< MIN_INIT_FRAMES` non-None entries.
  - `_solve_once(frame_ids, frame_data, image_size, C0, r0, f0) -> (C, r_by, f_by, cost_before, cost_after)`.
  - `solve_fixed_center(corrs_by_frame, image_size, *, init_results, max_rounds: int = 2) -> list` — full entry point (self-audit added in Task 3; this task returns results for all usable frames, single round).

- [ ] **Step 1: Write the failing tests** (append)

```python
def _init_results_from_truth(ids, f_true, R_true, n_frames, jitter=0.0, seed=1,
                             keep_every=1):
    """Fake per-frame sweep output: CalibrationResult for kept frames, None else."""
    import cv2
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    rng = np.random.default_rng(seed)
    out = [None] * n_frames
    for i in ids:
        if i % keep_every != 0:
            continue
        R = R_true[i]
        C = C_TRUE + rng.normal(0, jitter, 3)
        t = -R @ C
        out[i] = CalibrationResult(
            intrinsics=CameraIntrinsics(f_true[i] * (1 + rng.normal(0, jitter / 50)),
                                        f_true[i], W / 2, H / 2, W, H),
            pose=CameraPose(R=R, t=t), rms_px=0.5, num_correspondences=8,
            refined_with_ba=True)
    return out


def test_init_from_results_median_center_and_interpolation():
    from nfl_gsplat.calibration.joint_solve import init_from_results
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=30)
    init = _init_results_from_truth(ids, f_true, R_true, 30, jitter=0.5, keep_every=1)
    init[5] = init[6] = None                    # gap: must be interpolated
    C0, r0, f0 = init_from_results(init, ids)
    assert np.linalg.norm(C0 - C_TRUE) < 2.0
    assert 5 in r0 and 5 in f0                  # gap frames got initialized
    assert f0[5] == pytest.approx(f_true[5], rel=0.05)


def test_init_from_results_fails_loud_when_sparse():
    from nfl_gsplat.calibration.joint_solve import init_from_results
    from nfl_gsplat.errors import CalibrationError
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=30)
    init = _init_results_from_truth(ids, f_true, R_true, 30, keep_every=10)  # 3 frames
    with pytest.raises(CalibrationError, match="initializer"):
        init_from_results(init, ids)


def test_solve_fixed_center_recovers_ground_truth():
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=50, noise_px=0.5)
    # _frame_data_override bypasses landmark-name resolution (synthetic points
    # are arbitrary field-plane locations, not named NFL landmarks)
    results = solve_fixed_center(
        corrs_by_frame=None, image_size=(W, H),
        init_results=_init_results_from_truth(ids, f_true, R_true, 50, jitter=1.0),
        _frame_data_override=fd)
    solved = [r for r in results if r is not None]
    assert len(solved) >= 45
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_TRUE) < 0.5
    for i, r in enumerate(results):
        if r is None:
            continue
        assert r.intrinsics.fx == pytest.approx(f_true[i], rel=0.02)
        assert np.allclose(r.pose.center_world(), C_rec)        # ONE center, all frames


def test_solve_fixed_center_diverging_fails_loud(monkeypatch):
    from nfl_gsplat.calibration import joint_solve as js
    from nfl_gsplat.errors import CalibrationError
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=25, noise_px=0.5)

    def fake_solve_once(frame_ids, frame_data, image_size, C0, r0, f0):
        return C0, r0, f0, 100.0, 500.0          # cost went UP
    monkeypatch.setattr(js, "_solve_once", fake_solve_once)
    with pytest.raises(CalibrationError, match="diverged"):
        js.solve_fixed_center(corrs_by_frame=None, image_size=(W, H),
                              init_results=_init_results_from_truth(ids, f_true, R_true, 25),
                              _frame_data_override=fd)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_joint_solve.py -v -k "init_from or solve_fixed"`
Expected: FAIL — `ImportError: cannot import name 'init_from_results'`

- [ ] **Step 3: Implement** (append to joint_solve.py)

```python
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
    fn = lambda x: residuals(x, frame_ids, frame_data, image_size)
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
    for i in frame_ids:
        results[i] = _frame_result(i, frame_data, C, r_by, f_by, image_size)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_joint_solve.py -v`
Expected: all PASS. Recovery test must finish in seconds; if slow, check the
sparsity matrix is actually passed (dense Jacobian on 200+ params is the
usual culprit).

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/joint_solve.py tests/test_joint_solve.py
git commit -m "feat(calibration): fixed-center joint optimizer with sweep init

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Self-audit round (drop irreconcilable frames, re-solve)

**Files:**
- Modify: `nfl_gsplat/calibration/joint_solve.py` (`solve_fixed_center`)
- Test: `tests/test_joint_solve.py` (append)

**Interfaces:**
- Consumes: Task 2's `solve_fixed_center`, `_solve_once`, `_frame_result`, `AUDIT_DROP_PX`.
- Produces: same `solve_fixed_center` signature; audited-out frames return `None` in the results list.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_self_audit_drops_identity_shifted_frame():
    # one frame's world points shifted a full yard-line spacing (+4.572 m in X)
    # = the consistent-mislabel failure mode: PERFECT per-frame residual,
    # irreconcilable with the shared center -> must be dropped, others kept
    from nfl_gsplat.calibration.field_landmarks import YARD_LINE_SPACING_M
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=40, noise_px=0.3)
    bad = 17
    world, uv = fd[bad]
    world = world.copy(); world[:, 0] += YARD_LINE_SPACING_M
    fd[bad] = (world, uv)
    results = solve_fixed_center(
        corrs_by_frame=None, image_size=(W, H),
        init_results=_init_results_from_truth(ids, f_true, R_true, 40, jitter=1.0),
        _frame_data_override=fd)
    assert results[bad] is None                          # audited out
    solved = [r for r in results if r is not None]
    assert len(solved) >= 35
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_TRUE) < 0.5          # unpoisoned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_joint_solve.py::test_self_audit_drops_identity_shifted_frame -v`
Expected: FAIL (frame 17 currently kept; either non-None result or poisoned center).

- [ ] **Step 3: Implement** — replace the tail of `solve_fixed_center` (after the divergence check) with the audit loop:

```python
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
    return results
```

- [ ] **Step 4: Run all joint-solve tests**

Run: `python -m pytest tests/test_joint_solve.py -v`
Expected: all PASS (including Task 2's — audit is a no-op on clean data).

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

```bash
git add nfl_gsplat/calibration/joint_solve.py tests/test_joint_solve.py
git commit -m "feat(calibration): joint-solve self-audit drops irreconcilable frames

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire into pretrained pipeline + acceptance

**Files:**
- Modify: `nfl_gsplat/calibration/run_autocalib.py` (`build_autocalib_npz_pretrained`)
- Test: `tests/test_run_autocalib.py` (append)

**Interfaces:**
- Consumes: `solve_fixed_center(corrs_by_frame, image_size, *, init_results, ...)` (Tasks 2-3); existing `_solve_sweep`, `assemble_track_from_results`, `write_camera_track`.
- Produces: pretrained mode pipeline = phase 1 (detect+fuse, unchanged) → phase 2 `_solve_sweep` (unchanged, initializer) → phase 3 `solve_fixed_center` → assembly. Import `solve_fixed_center` at module level in run_autocalib.py (monkeypatchable as `ra.solve_fixed_center`). `_register_sequence_pretrained` (test seam) stays sweep-only — do not change it.

- [ ] **Step 1: Write the failing test** (append to tests/test_run_autocalib.py)

```python
def test_build_pretrained_uses_joint_solve(monkeypatch):
    # phase 3 gate: build_autocalib_npz_pretrained must pass the sweep output
    # into solve_fixed_center and assemble ITS results, not the sweep's
    from nfl_gsplat.calibration import run_autocalib as ra

    captured = {}

    def fake_joint(corrs_by_frame, image_size, *, init_results, **kw):
        captured["init"] = init_results
        return ["JOINT0", "JOINT1"]
    monkeypatch.setattr(ra, "solve_fixed_center", fake_joint)

    assembled = {}
    def fake_assemble(results, *, width, height, **kw):
        assembled["results"] = results
        return "TRACK"
    monkeypatch.setattr(ra, "assemble_track_from_results", fake_assemble)
    monkeypatch.setattr(ra, "write_camera_track", lambda p, tr, fps: p)

    class _Meta:
        num_frames, width, height = 2, 1920, 1080
    monkeypatch.setattr("nfl_gsplat.utils.video.ffprobe_meta", lambda v: _Meta())
    monkeypatch.setattr("nfl_gsplat.utils.video.iter_frames",
                        lambda v, start_frame=0: iter([]))
    monkeypatch.setattr(ra, "load_kps_json", lambda p, expect_num_frames=None: {},
                        raising=False)

    ra.build_autocalib_npz_pretrained(
        play_dir=".", videos={"sideline": "v.mp4"}, fps=30.0,
        kps_json="kps.json", territory="away")
    assert assembled["results"] == ["JOINT0", "JOINT1"]     # joint output assembled
    assert "init" in captured                               # sweep fed the init
```

(Adapt monkeypatch targets to how run_autocalib actually imports ffprobe_meta /
iter_frames / load_kps_json — they are imported inside the function body from
`nfl_gsplat.utils.video` and `nfl_gsplat.calibration.roboflow_kps`; patch at the
SOURCE module path as shown, and verify the empty-frames case flows through. If
`load_kps_json` is imported locally, patch `nfl_gsplat.calibration.roboflow_kps.load_kps_json`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_autocalib.py::test_build_pretrained_uses_joint_solve -v`
Expected: FAIL — `AttributeError: ... no attribute 'solve_fixed_center'`

- [ ] **Step 3: Implement**

In `run_autocalib.py`: add module-level import
`from nfl_gsplat.calibration.joint_solve import solve_fixed_center`.
In `build_autocalib_npz_pretrained`, after the existing sweep produces
`results` (the per-frame initializer), replace the direct assembly with:

```python
        # Phase 3: fixed-center joint solve — per-frame PnP is multimodal on
        # planar telephoto views (see spec 2026-07-06); the sweep output only
        # initializes the joint problem.
        joint = solve_fixed_center(corrs_by_frame, (meta.width, meta.height),
                                   init_results=results)
        tracks[cam] = assemble_track_from_results(joint, width=meta.width,
                                                  height=meta.height)
```

(Keep the sweep call and `corrs_by_frame` exactly as built today; the only
change is what gets assembled.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_run_autocalib.py tests/test_joint_solve.py -v`
Expected: all PASS.
Run: `python -m pytest -m "not gpu and not slow" -q` — all pass.

- [ ] **Step 5: Commit**

```bash
git add nfl_gsplat/calibration/run_autocalib.py tests/test_run_autocalib.py
git commit -m "feat(calibration): pretrained mode assembles the fixed-center joint solve

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Real-footage acceptance (human gate, run on the user's Windows machine)**

```
python scripts/02_autocalibrate.py --play-dir data\2025\week_01\SEA_at_AZ\play_001 --mode pretrained --roboflow-kps "data\2025\week_01\SEA_at_AZ\play_001\roboflow_kps.json" --territory away --cameras sideline
```
Expected: `wrote automatic per-frame calibration → ...cameras.npz` (first full-play
success). Then verify: recovered center within ~2 m of (−19, 1, 95) — print via
`python -c "import numpy as np; d=np.load(r'data\2025\week_01\SEA_at_AZ\play_001\cameras.npz'); K=d['sideline_K']; R=d['sideline_R']; t=d['sideline_t']; C=-np.einsum('nij,nj->ni', R.transpose(0,2,1), t); print(C[d['sideline_conf']>0].mean(0), K[:,0,0].min(), K[:,0,0].max())"`
(adapt key names to cameras_io's actual npz layout — check
`nfl_gsplat/calibration/cameras_io.py` first). Grid-overlay spot check via the
existing diag on sampled frames. THIS IS THE ACCEPTANCE GATE.
