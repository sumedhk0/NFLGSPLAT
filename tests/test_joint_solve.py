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


def test_build_frame_data_unknown_name_fails_loud():
    from nfl_gsplat.calibration.joint_solve import build_frame_data
    from nfl_gsplat.errors import CalibrationError
    corrs = {0: [("not_a_landmark", (1.0, 2.0))] * 4}
    with pytest.raises(CalibrationError, match="unknown landmark"):
        build_frame_data(corrs)


def _init_results_from_truth(ids, f_true, R_true, n_frames, jitter=0.0, seed=1,
                             keep_every=1):
    """Fake per-frame sweep output: CalibrationResult for kept frames, None else."""
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


def test_init_from_results_plausible_anchor_median():
    # new contract: median center of the PLAUSIBLE anchors, else None (no raise)
    from nfl_gsplat.calibration.joint_solve import init_from_results
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=30)
    init = _init_results_from_truth(ids, f_true, R_true, 30, jitter=0.5, keep_every=1)
    C0 = init_from_results(init)
    assert C0 is not None
    assert np.linalg.norm(C0 - C_TRUE) < 2.0


def test_init_from_results_none_when_too_few_plausible():
    from nfl_gsplat.calibration.joint_solve import init_from_results
    # keep_every=20 over 30 frames leaves anchors at 0 and 20 -> only 2 -> None
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=30)
    init = _init_results_from_truth(ids, f_true, R_true, 30, keep_every=20)
    assert init_from_results(init) is None


def test_init_from_results_none_when_implausible_focals():
    # anchors present but physically impossible (sweep failure mode) -> gated out
    from nfl_gsplat.calibration.joint_solve import init_from_results
    from nfl_gsplat.calibration.solve_pnp import CalibrationResult
    bad = []
    for _ in range(10):
        R = _look_at_R(C_TRUE, np.array([0.0, 0.0, 0.0]))
        bad.append(CalibrationResult(
            intrinsics=CameraIntrinsics(5e13, 5e13, W / 2, H / 2, W, H),
            pose=CameraPose(R=R, t=-R @ C_TRUE), rms_px=0.5,
            num_correspondences=8, refined_with_ba=True))
    assert init_from_results(bad) is None


def test_solve_fixed_center_recovers_ground_truth():
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=50, noise_px=0.5)
    # _frame_data_override bypasses landmark-name resolution (synthetic points
    # are arbitrary field-plane locations, not named NFL landmarks)
    results, mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=(W, H),
        init_results=_init_results_from_truth(ids, f_true, R_true, 50, jitter=1.0),
        _frame_data_override=fd)
    assert mirrored is False
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

    def fake_solve_once(frame_ids, frame_data, image_size, C0, r0, f0, **kw):
        return C0, r0, f0, 100.0, 500.0          # cost went UP -> divergence
    monkeypatch.setattr(js, "_solve_once", fake_solve_once)
    with pytest.raises(CalibrationError, match="diverged"):
        js.solve_fixed_center(corrs_by_frame=None, image_size=(W, H),
                              init_results=_init_results_from_truth(ids, f_true, R_true, 25),
                              _frame_data_override=fd)


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
    results, mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=(W, H),
        init_results=_init_results_from_truth(ids, f_true, R_true, 40, jitter=1.0),
        _frame_data_override=fd)
    assert mirrored is False
    assert results[bad] is None                          # rescue refit rejects it
    solved = [r for r in results if r is not None]
    assert len(solved) >= 35
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_TRUE) < 0.5          # unpoisoned


def test_self_audit_all_frames_bad_fails_loud(monkeypatch):
    from nfl_gsplat.calibration import joint_solve as js
    from nfl_gsplat.errors import CalibrationError
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=25, noise_px=0.3)

    def fake_solve_once(frame_ids, frame_data, image_size, C0, r0, f0, **kw):
        # Every solve "converges" (cost drops, no divergence) but to a camera
        # 60 m off in X. The rescue-first gate freezes that wrong C and cannot
        # fit any frame's (r, f) within tolerance -> fail loud (either at the
        # rescue-first consistency gate or the final all-rejected guard).
        return C0 + np.array([60.0, 0.0, 0.0]), r0, f0, 100.0, 50.0
    monkeypatch.setattr(js, "_solve_once", fake_solve_once)
    with pytest.raises(CalibrationError,
                       match="rejected every frame|frames consistent with the multi-start"):
        js.solve_fixed_center(
            corrs_by_frame=None, image_size=(W, H),
            init_results=_init_results_from_truth(ids, f_true, R_true, 25),
            _frame_data_override=fd)


def test_solve_fixed_center_resolves_mirrored_labels():
    # The fused left/right hash convention can be flipped for a camera side:
    # negating world Y keeps every homography perfect but is a reflection no
    # rigid camera fits. solve_fixed_center must detect it, relabel, and return
    # results in the TRUE world frame (same camera, since the field is
    # Y-symmetric).
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=50, noise_px=0.5)
    flip = np.array([1.0, -1.0, 1.0])
    fd_mir = {i: (w * flip, uv) for i, (w, uv) in fd.items()}   # wrong labeling
    results, mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=(W, H),
        init_results=_init_results_from_truth(ids, f_true, R_true, 50, jitter=1.0),
        _frame_data_override=fd_mir)
    assert mirrored is True
    solved = [r for r in results if r is not None]
    assert len(solved) >= 45
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_TRUE) < 0.5


def test_rotated_view_solves_to_same_camera():
    # Simulate an endzone-style rotated view: rotate every synthetic uv
    # observation 90 deg (as the pipeline does to endzone frames), solve in
    # rotated coordinates, de-rotate — the recovered camera must match the
    # unrotated ground truth. End-to-end check of rotate_uv + solve +
    # derotate_result composing correctly.
    #
    # Anchor-seeded (not the no-anchor grid-only variant): C_TRUE = (-19, 1, 95)
    # is far from every sideline/endzone grid point (nearest grid points are
    # ~50-55 m away — see _GRID_X/_GRID_Y/_GRID_Z and _GRID_EZ_* in
    # joint_solve.py), so a grid-only multi-start has no nearby candidate to
    # converge from. Anchors are rotation-agnostic (they carry only C; the
    # per-frame pose init is recomputed by look-at inside the solver) — but
    # look-at alone still assumes near-zero roll, which a 90-deg-rotated
    # working image never has (measured: initial per-frame reprojection error
    # in the thousands of px, ~80-100 deg of rotation from the true pose, too
    # far for LM to recover under the robust loss no matter the iteration
    # budget). _init_frame (joint_solve.py) scores 4 in-plane roll hypotheses
    # per frame against the actual observations and seeds LM from the best
    # one, but ONLY when told the view is rotated (view_deg=90 below) — an
    # error-threshold heuristic can't safely auto-detect this (a wrong
    # multi-start candidate on an UNROTATED view also produces high roll=0
    # error, and scoring rolls there let a wrong camera win; see
    # joint_solve.py's _init_frame docstring). Passing view_deg=90 here
    # mirrors what the pretrained pipeline does: it knows the working-view
    # rotation up front and threads it through explicitly.
    from nfl_gsplat.calibration.joint_solve import solve_fixed_center
    from nfl_gsplat.calibration.view_rotation import (
        derotate_result, rotate_uv, rotated_wh,
    )
    ids, fd, f_true, R_true = _synthetic_frames(n_frames=30, noise_px=0.3)
    rw_h = rotated_wh(90, (W, H))
    fd_rot = {}
    for i in ids:
        world, uv = fd[i]
        uv_r = np.array([rotate_uv(u, v, 90, (W, H)) for (u, v) in uv])
        fd_rot[i] = (world, uv_r)
    results, mirrored = solve_fixed_center(
        corrs_by_frame=None, image_size=rw_h,
        init_results=_init_results_from_truth(ids, f_true, R_true, 30, jitter=1.0),
        _frame_data_override=fd_rot, view_deg=90)
    # mirrored may legitimately be True here: a 90 deg rotation can flip the
    # handedness the reflection-resolve sees (the field is Y-symmetric), and
    # solve_fixed_center already returns results in the TRUE world frame
    # regardless (see test_solve_fixed_center_resolves_mirrored_labels).
    # Assert on the recovered camera, not on `mirrored`.
    solved = [(i, r) for i, r in enumerate(results) if r is not None]
    assert len(solved) >= 25
    deros = [derotate_result(r, 90, (W, H)) for _i, r in solved]
    C_rec = deros[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_TRUE) < 0.5
    # C alone cannot catch a wrong _rz composition: center_world() = -R^T t is
    # provably invariant to ANY orthogonal rz (see view_rotation.py's own
    # docstring), correct angle or not — verified empirically by temporarily
    # flipping _rz's sign for deg=90: C_rec still landed within 1 cm of
    # C_TRUE. Check the recovered ORIENTATION too, which the same broken sign
    # threw ~180 deg off (vs. <0.02 deg with the correct sign).
    for (i, _r), d in zip(solved, deros):
        assert d.intrinsics.fx == pytest.approx(f_true[i], rel=0.02)
        cos_ang = (np.trace(d.pose.R @ R_true[i].T) - 1) / 2
        ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        assert ang_deg < 1.0


def test_candidate_centers_include_endzone_positions():
    # endzone cameras sit behind an endzone (|X| 60-120, Y ~ 0) — the
    # sideline-only grid physically could not reach them (2026-07-07 failure)
    from nfl_gsplat.calibration.joint_solve import _candidate_centers
    cands = np.stack(_candidate_centers([None] * 5))
    endzoneish = (np.abs(cands[:, 0]) >= 60) & (np.abs(cands[:, 1]) <= 20)
    assert endzoneish.sum() >= 18            # both endzones x several Z/Y
    sideline = (np.abs(cands[:, 0]) <= 30) & (np.abs(cands[:, 1]) >= 45)
    assert sideline.sum() >= 54              # original grid retained
