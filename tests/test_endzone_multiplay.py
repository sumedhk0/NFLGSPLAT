import numpy as np
from nfl_gsplat.utils.geometry import CameraIntrinsics, project_points


def _look_at(C, target):
    fwd = np.asarray(target, float) - np.asarray(C, float); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _play_corrs(C_true, rng, n_frames=25, n_players=8, wh=(1920, 1080)):
    corrs = {}
    for i in range(n_frames):
        tx = -20.0 + 30.0 * i / (n_frames - 1)
        xs = tx + rng.uniform(-8, 8, n_players); ys = rng.uniform(-22, 22, n_players)
        world = np.column_stack([xs, ys, np.zeros(n_players)])
        R = _look_at(C_true, [tx, 0.0, 0.0]); t = -R @ C_true
        f = 2500.0 + 300.0 * i / (n_frames - 1)
        K = CameraIntrinsics(f, f, wh[0] / 2, wh[1] / 2, wh[0], wh[1]).K()
        uv = project_points(world, K, R, t)
        ok = np.isfinite(uv).all(axis=1)
        corrs[i] = (world[ok], uv[ok] + rng.normal(0, 0.5, uv[ok].shape))
    return corrs


def test_solve_endzone_identity_multiplay_recovers():
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior, solve_endzone_identity
    rng = np.random.default_rng(0)
    C_true = np.array([-112.0, 0.0, 24.0])
    plays = [_play_corrs(C_true, rng) for _ in range(3)]
    prior = EndzonePrior(x_range=(-150, -60), y_range=(-15, 15), z_range=(10, 60),
                         focal_range=(1500, 3500))
    per_play = solve_endzone_identity(plays, (1920, 1080), prior, audit_drop_px=8.0)
    solved = [r for pl in per_play for r in pl if r is not None]
    assert len(solved) >= 30
    C_rec = solved[0].pose.center_world()
    assert np.linalg.norm(C_rec - C_true) < 1.0        # near truth, not the +112 mirror
    assert C_rec[0] < 0                                 # correct side (prior box)


def test_solve_endzone_identity_fails_loud_too_few():
    import pytest
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior, solve_endzone_identity
    from nfl_gsplat.errors import CalibrationError
    prior = EndzonePrior((-150, -60), (-15, 15), (10, 60), (1500, 3500))
    with pytest.raises(CalibrationError, match="usable frames"):
        solve_endzone_identity([{0: (np.zeros((4, 3)), np.zeros((4, 2)))}],
                               (1920, 1080), prior)


def test_prior_center_bounds_and_center0():
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior
    p = EndzonePrior((-150, -60), (-15, 15), (10, 60), (1500, 3500))
    assert p.center_bounds == ((-150, -60), (-15, 15), (10, 60))
    assert np.allclose(p.center0, [-105.0, 0.0, 35.0])


def test_center_bounds_constrains_to_prior_box():
    # Same scene whose true camera is at X=-112, but the prior box is on the
    # WRONG (mirror) side, excluding the truth. center_bounds must keep the solve
    # out of the true region: it either fails loud (no in-box camera fits the
    # data) or returns a center in the wrong box -- never the true -112. If
    # center_bounds were ignored the solve would recover -112 (proven by the
    # correct-side recovery test), so this is a real regression guard.
    from nfl_gsplat.calibration.endzone_multiplay import EndzonePrior, solve_endzone_identity
    from nfl_gsplat.errors import CalibrationError
    import numpy as np
    rng = np.random.default_rng(0)
    C_true = np.array([-112.0, 0.0, 24.0])
    plays = [_play_corrs(C_true, rng) for _ in range(3)]
    prior_wrong = EndzonePrior(x_range=(50, 150), y_range=(-15, 15),
                               z_range=(10, 60), focal_range=(1500, 3500))
    try:
        per_play = solve_endzone_identity(plays, (1920, 1080), prior_wrong, audit_drop_px=8.0)
    except CalibrationError:
        return   # no in-box camera consistent with the data -> acceptable
    solved = [r for pl in per_play for r in pl if r is not None]
    for r in solved:
        assert r.pose.center_world()[0] > 0   # stayed in the wrong box, never found true -112
